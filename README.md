# FacDiggerNN

FacDiggerNN 是面向美股日频横截面选股的 point-in-time 机器学习因子研究 CLI。它把
EODHD 或自备标准 Parquet 数据转换为内容寻址快照，训练 E0—E3 对照模型，并输出可回放、
可审计的因子、评价和 walk-forward 研究结果。

本项目是研究工具，不是交易系统；不负责下单、撮合、仓位或资金管理。

> **当前状态**：数据、训练、评价、回放和研究冻结的工程闭环已实现。监督训练以同日股票
> 横截面排序为目标；final holdout 只能在显著性门禁通过、协议冻结并显式解封后，先登记
> holdout 访问和核对冻结样本键，再以截至 validation 末日的数据重新训练后评价。历史动态
> EODHD bronze 曾在项目机器上完成采集和质量审计，
> 但 `data/` 与 `artifacts/` 不进入 Git，新 clone 必须迁移或重建。由于真实退市收益、点时
> 行业和点时流通市值仍缺失，仓库中的 M6 配置明确属于 **engineering research**，不能据此
> 宣称正式样本外 Alpha。

完整文档从[文档中心](docs/README.md)进入：

- [开发文档](docs/开发文档.md)：架构、契约、模块、CLI、扩展和排错；
- [实验设计文档](docs/实验设计文档.md)：E0—E3、切分、统计、冻结和结论边界；
- [Windows RTX 训练指南](docs/RTX2070_Windows训练指南.md)：WSL2、CUDA、数据迁移和内存门禁；
- [关键问题与修复复盘](docs/项目关键问题与修复复盘.md)：真实故障、设计权衡和验证证据。

## 当前能力

| 领域 | 已实现能力 |
|---|---|
| 数据 | EODHD provider、标准 Parquet 契约、来源质量证明、响应缓存和调用预算 |
| 股票池 | active + delisted 候选、历史日 ADV20 动态 top-1000、交易日历和身份隔离 |
| 数据集 | 七通道特征、Train-only scaler、五日超额收益标签、不可变快照 |
| 模型 | E0 LightGBM/MLP、E1 随机 PatchTST、E2 ETTh1 迁移、E3 金融域预训练 |
| 训练 | 完整日横截面排序目标、显存受限两遍精确梯度、inner selection、resume 和泄漏审计 |
| 评价 | IC/Rank IC、ICIR、分组收益、换手、成本、稳定性和中性化可用性报告 |
| 研究 | 多 fold/seed 矩阵、HAC/非重叠检验、Holm 校正、freeze 和 final refit |
| 推理 | checkpoint 独立回放、无标签因子导出、最新信号和独立 prediction 评价 |

核心数据流：

```text
provider / 标准 Parquet
  -> 标准表 + provider-neutral provenance
  -> 内容寻址 snapshot
  -> E0 / E1 / E2 / E3
  -> 统一 prediction 契约与评价
  -> checkpoint 回放 / signal / walk-forward freeze / final refit
```

## 研究任务

- 市场：Nasdaq、NYSE、NYSE American 普通股日频；
- 决策时点：交易日 `t` 收盘后，最早 `t+1` 开盘执行；
- 输入：最近 512 个市场 session 的 7 个价格/成交量通道；
- 标签：`t+1` 开盘到 `t+5` 收盘的对数收益，减当日 eligible 股票池等权收益；
- 监督目标：同日横截面 `1 - corr(score, target_rank)`；
- 主评价：逐日 Rank IC 及其显著性；20 bps 组合结果是参考评价，不是排序主门禁。

神经模型的监督目标是 `cross_sectional_rank_correlation_surrogate_v2_full_date`。
DataLoader 在 CPU 一次组装一个完整交易日，GPU 内只保留由 `batch_size` 限制的
physical microbatch；两遍回放用完整日统计量计算精确梯度。因此降低 `batch_size`
不会把目标退化为小横截面相关性。但模型使用 train-mode BatchNorm，物理微批大小
仍会影响 BN 统计、吞吐和训练轨迹，正式对照必须固定它。

## 安装

推荐 Python 3.11 和 [uv](https://docs.astral.sh/uv/)：

```bash
git clone git@github.com:wwbotww/FacDiggerNN.git
cd FacDiggerNN
git checkout REVIEWED_COMMIT_SHA
uv sync --frozen --all-extras
uv run facdigger doctor
```

正式实验必须 checkout 审阅后的确切 commit，并记录 `git rev-parse HEAD`；不要
依赖会继续移动的分支名。

`uv.lock` 是首选锁文件。`requirements-lock.txt` 是从同一 lock 导出的全 extras pip
fallback，主要供不能使用 uv 的环境使用。Windows + RTX 2070 Super 正式训练推荐 WSL2，
不要复制 macOS 的 `.venv`；详见[平台指南](docs/RTX2070_Windows训练指南.md)。

## EODHD token

只有 live probe、ingest 或刷新缓存需要 token。仓库不会自动创建 `.env.local`：

```bash
touch .env.local
chmod 600 .env.local
```

在文件中写入：

```dotenv
EODHD_API_TOKEN=your_token_here
```

每个新终端加载一次：

```bash
set -a
source .env.local
set +a
```

`.env.local` 已被 Git 忽略。不要把 token 写进 YAML、命令输出、缓存键或 manifest。

## 运行主线

### 1. 准备和验证历史标准表

若已从另一台机器迁移完整 bronze，只需复制整个目录，不能只复制某一张表：

```text
data/bronze/eodhd_us_historical_liquid/
├── bars_daily.parquet
├── universe_daily.parquet
├── corporate_actions.parquet
├── delistings.parquet
└── eodhd_ingestion_manifest.json
```

随后验证通用来源证明和文件哈希：

```bash
uv run facdigger data validate \
  --config configs/datasets/eodhd_historical_liquid.yaml
```

需要重新采集时先运行只读 probe，确认候选规模、配额、磁盘和时间，再显式 ingest：

```bash
uv run facdigger data probe --config configs/data/eodhd_historical_liquid.yaml
uv run facdigger data ingest --config configs/data/eodhd_historical_liquid.yaml
```

全历史采集会产生大量付费 API 请求；`probe` 不会隐式启动下载。

### 2. 单模型实验

单次 E0—E3 训练先构建普通快照：

```bash
uv run facdigger dataset build \
  --config configs/datasets/eodhd_historical_liquid.yaml

uv run facdigger train e0 \
  --config configs/experiments/e0_lightgbm_paid_pilot.yaml \
  --dataset data/snapshots/<dataset_id>

uv run facdigger train e1 \
  --config configs/experiments/e1_random.yaml \
  --dataset data/snapshots/<dataset_id>
```

E1—E3 支持从同一 dataset、完整配置和来源权重哈希绑定的 `last.pt` 恢复。E2/E3 首次运行
还需要取得锁定 revision 的 IBM PatchTST 权重。

当前 E1—E3 监督 checkpoint 是 schema v3，排序目标是 v2 完整日协议。旧 Huber
或 v1 小块相关性 artifacts 只可留作历史证据，不能 resume 到当前协议，也不能与
新结果混合比较。
E3 reconstruction checkpoint 独立使用 schema v2 和
`patch_alignment_protocol=patchifier_sequence_start_v2`，用来拒绝修复前 mask 错位权重。

### 3. Walk-forward engineering research

M6 runner 会按 fold 自行建立 `data/walk_forward_snapshots/`，不要求先构建上面的普通快照：
当前配置的 `research_id` 是 `m6_eodhd_engineering_full_date_v2`，用于完整日
objective v2 的新一轮 validation；它不继承旧 artifacts，final holdout 仍锁定。

```bash
uv run facdigger research plan \
  --config configs/research/m6_eodhd_engineering.yaml

uv run facdigger research preflight \
  --config configs/research/m6_eodhd_engineering.yaml

uv run facdigger research run \
  --config configs/research/m6_eodhd_engineering.yaml
```

中断后使用研究目录恢复：

```bash
uv run facdigger research run \
  --config configs/research/m6_eodhd_engineering.yaml \
  --resume-run artifacts/research/<research_run_id>
```

只有 validation 全部完成、冻结报告通过统计门禁并经过人工审阅后，才能在同一命令追加
`--unlock-final-holdout`。显式参数不能绕过 `no_go`；通过后系统先永久登记 holdout 已访问，
构建 refit snapshot 并核对冻结的 2025 test 键，再重新训练和一次性评价。2025 test 不进入
训练、scaler 拟合或 checkpoint selection。

## 配置选择

| 配置 | 用途 | 不能代表什么 |
|---|---|---|
| `configs/base.yaml` | 环境、CUDA 和 PatchTST checkpoint 诊断 | 训练实验 |
| `configs/data/eodhd_free.yaml` | 两只股票、低成本 live API smoke | 横截面研究 |
| `configs/data/eodhd_all_world_pilot.yaml` | 当前 active 100 股票资源门禁 | 无存活偏差的研究 |
| `configs/data/eodhd_historical_liquid.yaml` | 历史动态 top-1000 主数据路径 | 自动 research-ready |
| `configs/experiments/*_smoke.yaml` | 快速端到端测试 | 正式模型结论 |
| `configs/experiments/*_paid_pilot.yaml` | 真实规模资源验证 | 多 seed 正式对照 |
| `configs/experiments/e1_random.yaml`、`e2_etth1.yaml`、`e3_financial_pretrain.yaml` | 完整模型配置 | 独立于 M6 的正式结论 |
| `configs/research/m6_eodhd_engineering.yaml` | 当前 walk-forward 主线 | 完成正式中性化后的研究 |

`configs/datasets/us_equities_daily_v1.yaml` 是 provider-neutral 标准表范例；EODHD 历史主线
使用 `configs/datasets/eodhd_historical_liquid.yaml`。

## 产物与安全边界

- `data/bronze/`：标准化来源表，删除后可能需要重新消耗 API 配额；
- `data/cache/`：EODHD 原始响应缓存，用于避免重复请求和离线重建；
- `data/state/`：本地调用预算状态；
- `data/snapshots/`、`data/walk_forward_snapshots/`：可重建的内容寻址快照；
- `artifacts/`：checkpoint、预测、指标、报告和研究冻结；
- `.env.local`：仅本机秘密。

这些路径都不进入 Git。不要手工修改已生成 snapshot 或 run；模型、目标、协议或配置变更后，
旧结果只能作为历史证据，不能与新协议结果混用。

## 当前研究限制

1. EODHD 当前来源没有可靠的真实退市终值/原因，系统使用显式、版本化的保守插值。
2. 点时行业和点时流通市值缺失，正式行业/市值中性化门禁尚不能开启。
3. 历史 bronze、快照和结果是本地资产；每台新机器都必须迁移并重新校验哈希。
4. 16 GB RAM / RTX 2070 Super 已有内存优化和工程 pilot，但当前排序与 final-refit 协议仍需
   在目标机重新完成资源门禁后再跑完整 M6。当前建议是 8 GB GPU / 16 GB
   RAM 使用 `batch_size: 64`、FP16 先验收单个 cell；这是待实测的配置，不是已完成
   CUDA 全流程验收或速度承诺。主配置为首轮可行性注册了 5 个监督 epoch
   上限和 3 个 E3 reconstruction epoch 上限，不代表最优最终预算。
5. 项目不包含交易执行、组合约束优化或生产服务。

## 开发与验证

```bash
uv lock --check
uv run ruff check .
uv run pytest
```

普通测试使用 fake transport 和临时目录，不访问 live EODHD、付费配额、模型网络或正式
holdout。开发规则见[贡献指南](CONTRIBUTING.md)，秘密和研究完整性要求见
[安全策略](SECURITY.md)。
