# FacDiggerNN

FacDiggerNN 是面向美股日频横截面选股的 point-in-time 机器学习因子研究 CLI。它把
EODHD 或自备标准 Parquet 数据转换为内容寻址快照，既保留 E0—E3 历史对照，也提供金融
原生 Transformer 的从头训练/金融预训练配对实验，并输出可回放、可审计的评价和
walk-forward 研究结果。

本项目是研究工具，不是交易系统；不负责下单、撮合、仓位或资金管理。

> **当前状态**：数据、训练、评价、回放和研究冻结的工程闭环已实现。金融原生 Transformer
> 的输入、1/5/20 日标签、完整日 Set Transformer、显存受限 embedding replay、金融预训练、
> Train 内 linear probe、100-update RTX 基准和 3-fold 精简 runner 已实现，尚待 RTX 2070S
> CUDA 实测和完整训练。监督训练以同日股票
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
| 数据集 | 旧七通道；新 14 路个股 + 6 路市场状态、Train-only scaler、1/5/20 日标签、target-free 预训练索引 |
| 模型 | E0—E3；金融原生 local/market PatchTST + 多尺度统计 + 完整日 Set Transformer |
| 训练 | 完整日多期限排序、精确 embedding replay、连续片段金融预训练、Train 内 probe、resume 和泄漏审计 |
| 评价 | IC/Rank IC、ICIR、分组收益、换手、成本、稳定性和中性化可用性报告 |
| 研究 | 新主线 3 folds × 1 seed × 2 初始化；旧 M6 完整矩阵、HAC/非重叠检验、freeze 和 final refit |
| 推理/生产 | checkpoint 回放、ModelRelease、冻结 scaler 的无标签快照、单日/全历史 FactorBatch、Docker 日调度和独立评价 |

核心数据流：

```text
provider / 标准 Parquet
  -> 标准表 + provider-neutral provenance
  -> 训练 snapshot -> E0—E3（历史）/ finance Transformer（当前）
       -> 统一评价 / 精简 walk-forward
  -> ModelRelease -> 无标签 inference snapshot
       -> 单日 signal FactorBatch / 固定模型全历史回测 FactorBatch
```

跨项目交易接入只使用 `facdigger.factor_batch` 严格契约。FactorBatch 是
`factors.parquet + manifest.json` 的内容寻址不可变目录；`delivery_id` 同时绑定因子文件
哈希及来源、模型 release、推理输入、时间和覆盖语义，而不是只对 Parquet 求哈希。
checkpoint、配置和 scaler 不跨项目传递；它们由 FacDiggerNN 内部的 ModelRelease 绑定。
ModelRelease 还绑定 source run 原始 predictions 的文件哈希；`factor-batch
from-predictions` 不会接受修改或重新序列化后的预测文件。全历史回测则直接对无标签
inference snapshot 重新评分，不读取 predictions 或 target。
日常推理快照不读取 label 或 split，也不重新拟合 scaler；`facdigger signal` 接受已支持模型的
ModelRelease，并固定输出单个 as-of 日期的完整候选横截面。`--asof latest` 以候选
universe 的最新交易日为准，不会退回最近仍有可评分股票的旧日期；全体不可评分时输出
当日显式无信号批次（诊断能力，不代表清仓指令；每日生产质量门禁禁止发布全不可评分批次）。

## 当前 Transformer 研究任务

- 市场：Nasdaq、NYSE、NYSE American 普通股日频；
- 决策时点：交易日 `t` 收盘后，最早 `t+1` 开盘执行；
- 输入：最近 512 个市场 session 的 7 个原始价量通道、对应 7 个同日横截面 rank 通道，
  外加每日期 6 个市场状态通道；
- 标签：1/5/20 日未来超额收益，5 日为主任务；
- 监督目标：完整交易日内加权的多期限 `1 - corr(score_h, target_rank_h)`；
- 主评价：逐日 Rank IC 及其显著性；20 bps 组合结果是参考评价，不是排序主门禁。

旧 E1—E3 和新模型都以完整日排序为基础；新模型的 objective 是
`multi_horizon_full_date_rank_correlation`。
DataLoader 在 CPU 一次组装一个完整交易日，GPU 内只保留由 `batch_size` 限制的
physical microbatch；两遍回放用完整日统计量计算精确梯度。因此降低 `batch_size`
不会把目标退化为小横截面相关性。新模型使用 LayerNorm，不依赖 train-mode BatchNorm；
scratch 与 pretrained 仍必须固定相同 physical microbatch、完整数据、epoch 预算和
selection 规则。

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

### 3. 金融原生 Transformer 精简实验（当前主线）

精简矩阵严格固定为 3 个 fold、seed 42、scratch/finance-pretrained 两个监督方法；每个 fold
只做一次金融预训练。因此是 3 次预训练 + 6 个监督 cell，共 9 个长阶段。旧 E0—E3 和完整
3-seed 消融不自动运行。

在 RTX 机器上先为最大的 fold 构建新 snapshot，再做 100 个真实 optimizer update 的资源准入：

```bash
uv run facdigger dataset build \
  --config configs/datasets/eodhd_historical_liquid_transformer.yaml

uv run facdigger train finance-benchmark \
  --supervised-config configs/experiments/finance_patch_transformer_scratch.yaml \
  --pretraining-config configs/experiments/finance_patch_pretrain.yaml \
  --dataset data/snapshots/<largest_fold_dataset_id> \
  --updates 100 \
  --output artifacts/benchmarks/finance-transformer-rtx2070s.json
```

基准只提前停止测量任务，不改变正式模型、股票池、日期、512 日上下文或 epoch 上限。报告只有
在 CUDA/FP16 已实际启用、峰值显存/宿主 RAM 分别不超过 7.2/13 GiB，且保守矩阵投影不超过
14 天时才给出 `admitted=true`。`transformer-run` 会强制校验这份报告的配置哈希、最大 fold
dataset ID 和至少 100 个 update，缺失或不匹配时拒绝启动。随后执行：

```bash
uv run facdigger research transformer-plan \
  --config configs/research/finance_transformer_streamlined.yaml

uv run facdigger research transformer-run \
  --config configs/research/finance_transformer_streamlined.yaml
```

中断后只恢复未完成阶段：

```bash
uv run facdigger research transformer-run \
  --config configs/research/finance_transformer_streamlined.yaml \
  --resume-run artifacts/transformer_comparison/<research_run_id>
```

每个长任务在自身 run 目录持续追加 `progress.jsonl`，可在另一终端查看：

```bash
tail -f artifacts/transformer_comparison/<research_run_id>/runs/<fold>/<stage>/<run_id>/progress.jsonl
```

新监督 checkpoint 是 schema v4。训练快照回放、ModelRelease、target-free 快照、每日和
全历史 FactorBatch 已共用多模型推理入口，支持 E1—E3 与 `finance_patch_transformer`
（scratch / finance_pretrained）。工程可发布不代表因子有效；模型选择与真实部署仍需单独审阅
实验结论、实际交付身份和真实 Git 谱系。

### 4. 旧 Walk-forward engineering research（保留，不作为当前默认矩阵）

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

### 5. 发布日频因子（E1—E3 / Finance Transformer）

先选择完整监督 run，再创建 ModelRelease；默认要求训练与发布工作树 clean，联调可显式
追加 `--allow-dirty`，如实保留 dirty 来源，不影响产物哈希验证。release 固定记录训练 run 的
原始 commit。随后用它的冻结 scaler 构建
无标签推理快照，最后发布一个交易日的 FactorBatch：

```bash
uv run facdigger release create \
  --run artifacts/finance_transformer/<run_id> \
  --output-root artifacts/releases

uv run facdigger dataset build-inference \
  --config configs/datasets/eodhd_historical_liquid_inference.yaml \
  --release artifacts/releases/<release_id>

uv run facdigger signal \
  --release artifacts/releases/<release_id> \
  --dataset data/inference_snapshots/<snapshot_id> \
  --output-root artifacts/factor_batches \
  --delivery-config configs/inference/heyboss_delivery.local.yaml \
  --asof latest
```

发布目录只含 `factors.parquet` 和 `manifest.json`，这是 HeyBoss 的唯一输入。模型类型由
release 自动选择，五列数据格式相同，消费者只校验排序语义，不按模型名称分支。
研究 `predict` 只生成 predictions/metrics/report，不再维护第二种因子格式。
交付前将 [HeyBoss 清单模板](configs/inference/heyboss_delivery.example.yaml) 复制为本地配置，
填写双方确认的 `targets` 与有明确有效期/依据的 `identities`。计算仍使用完整横截面，之后
才筛选和映射交付证券；训练中其他股票缺 ISIN 不阻止 release。目标身份缺失、过期、歧义或
缺交则拒绝整批，不自动猜 ticker。未传 profile 的通用研究导出不声明 HeyBoss 身份已验证。
映射与计算/交付数量的旁路审计保存在 `artifacts/factor_batches_delivery_audits/`，不放入
交给 HeyBoss 的两文件目录；历史回放的清单/审计则保存在父级 resolved config/plan。
Windows 训练后迁移到本机时，可在 `release create` 追加 `--dataset <本机训练快照目录>`；
只覆盖文件定位，仍核对原 snapshot ID 和 manifest/scaler 哈希，不改写原 run 或 snapshot。
研究矩阵中的 run 请使用对应监督 cell 的实际目录；预训练 encoder 本身不能直接发布。
需要把同一 release 已绑定的原始评价 predictions 用于 HeyBoss 隔离回放时：

```bash
uv run facdigger factor-batch from-predictions \
  --predictions artifacts/finance_transformer/<run_id>/predictions.parquet \
  --release artifacts/releases/<release_id> \
  --output-root artifacts/factor_batches \
  --delivery-config configs/inference/heyboss_delivery.local.yaml

uv run facdigger factor-batch verify \
  --bundle artifacts/factor_batches/<delivery_id>
```

适配器输出 `source.kind=evaluation_predictions`，只有 eligible 已评分行，不能用于 paper。
它要求 predictions 文件字节与 ModelRelease 绑定值完全一致；旧 Huber、v1 chunked ranking、
mask 错位或未绑定 predictions/scaler 的 `artifacts2` 会被拒绝，不能借接口通用化绕过门禁。
显式 profile 中当日 active 的目标必须在原始预测中存在；需要区分不合格与缺交、覆盖动态
股票池时，使用下述有完整候选表的历史回放，不能把缺失预测静默当成不合格。

同一个已审 release 需要覆盖 inference snapshot 中全部历史日期时，不要循环调用单日
`signal`，使用固定模型历史回放：

```bash
cp configs/inference/e3_historical_replay.example.yaml \
  configs/inference/e3_historical_replay.local.yaml
# 填写本机路径、日期；HeyBoss 清单嵌入 delivery，且 security_ids 留空

uv run facdigger factor-history plan \
  --config configs/inference/e3_historical_replay.local.yaml
uv run facdigger factor-history run \
  --config configs/inference/e3_historical_replay.local.yaml
uv run facdigger factor-history verify \
  --export artifacts/factor_history/<history_id>
```

目录搬迁后，`factor-history plan/run` 可追加 `--release <本机 release>`、
`--dataset <本机 inference snapshot>`、`--output-root <本机历史输出根目录>`；`verify` 可追加
前两个参数。原 resolved config 不改写，续跑仍核对相同 release/snapshot 身份、交付计划和分片。

该路径不读取 label、target 或原 predictions；它加载一次固定 checkpoint，复用每日推理的
模型构建、窗口和评分代码，并按自然年原子发布、校验和恢复标准 FactorBatch。父目录中的
plan/state/manifest 只用于 FacDigger 续跑和审计，交给 HeyBoss 的仍是每个
`factor_batches/<delivery_id>/` 两文件目录。所有历史分片标记为
`evaluation_predictions/eligible_scored_cross_section`，以便消费者强制限制在 backtest；这里
固定模型、scaler 和当前历史来源修订可能使用了相对早期日期的未来信息，因此不是严格样本外
证据，禁止用于 paper 或据此宣称 Alpha。
`run` 会在 stderr 输出每个年份的 `scoring/published/verified` JSON 进度，最终汇总仍单独写到
stdout，便于终端观察和脚本解析。

Finance Transformer 始终先在快照中该日全部 eligible 股票上运行横截面网络，再按配置中的
`security_ids` 筛选交付。`batch_size` 只控制个股编码分块，不裁小模型的横截面。
示例配置沿用 `e3_historical_replay` 文件名，但内容适用于全部已支持的 release。

HeyBoss 侧的精确验收与流程测试计划见
[HeyBoss 因子联调交接](docs/HeyBoss因子联调交接.md)。

### 5. Docker 每日生产服务

日常生产不依赖 macOS `launchd`。Docker 容器内的 `facdigger production serve` 持有
New York 交易时钟、重试状态和单实例锁，因此同一镜像可部署到 macOS、Linux 或后续的
容器平台。每日事务固定为：

```text
19:00 America/New_York 首次尝试
  -> 持久化采集前的计算池覆盖基准
  -> fresh EODHD 最近 10 session 修订（首次部署会补齐历史 bronze 到 D 的缺口）
  -> 调整因子变化证券的定向 hot-window 回填
  -> 原子切换 production source CURRENT
  -> 计算池 / 实际交付池可用性检查
  -> 只为 D 建无标签 inference snapshot
  -> 实际可评分窗口 / Finance 市场输入检查
  -> 显式固定 release_id 推理
  -> D 的 FactorBatch 原子发布
  -> ledger 记录结果；推理快照保留最近 10 个交易日、每天最近 2 次尝试
```

局部缺数时，该交付股票保留 `eligible=false, score=null` 行，其余可评分股票仍在完整可用
计算横截面上推理，最后投影交付子集，不补价格、不补分数。默认容忍计算池损失不超过 5%、
交付池不可评分不超过 20%，且交付至少 3 只可评分；可在 `quality` 中调整。计算池仍受
`inference.minimum_*` 绝对数量约束，Finance 还检查共享市场输入。

异常缺失或暂时性供应商错误每 30 分钟重新采集，到下一 regular session 09:30 ET 截止。
即使 source 已更新到 D，只要尚未发布，重试也会重新获取修订；不会反复推理同一份缺数源。
只有 D 的可用性与完整性检查都通过才发布，少量不可评分行不等于交付不完整。
超过截止跳过 D，契约/身份/模型错误阻断 D 并告警；常驻服务继续等待后续交易日。
不会使用旧 FactorBatch，也不会改写 `data/snapshots/` 或
`data/walk_forward_snapshots/`。FactorBatch 永久保留。生产 source
只保存约 `context_length + 20` 个 session 的 bars、同窗口逐日 universe 和当前/上一修订，避免
每天复制全历史；内部 source revision 只用不透明 ID 标识原子状态，不扫描全表生成内容哈希。
历史 bronze 与训练 snapshot 保持独立、只读。

若已有旧 production store 只保存单日 universe，新代码会拒绝加载。请将本地生产配置的
`data.store_root` 指向新空目录，再从已验证历史 bronze 执行 `production bootstrap`；不要用今日
成员补写历史，也不要修改训练 snapshot。日常修订会连同修订区间及缺口日期重建成员资格，
保留区间外的历史状态；整体 ADV20 预热不足时停止发布，局部不足明确标记不可评分并计入门禁。

`production status` 的 `latest.quality` 区分 `ready/degraded/insufficient`，记录计算/交付
计数、缺数原因和市场检查；Docker 日志输出状态变化告警并去重。`production health` 判断
心跳存活，同时附带最新业务状态，不能把“容器健康”当作“今日可以调仓”。
HeyBoss 的缺分持仓保护及两侧 XNYS 日历统一代码已完成，826 原 validation 预测交付通过了
离线历史验收；真实每日采集、无标签推理、开盘前接纳和持续 paper 仍待验证，真实运行库未迁移。
见[局部缺分持仓保护交接](docs/HeyBoss局部缺分持仓保护交接.md)，不要把历史验收当成生产已就绪。

首次配置：

```bash
cp configs/production/eodhd_daily.example.yaml \
  configs/production/eodhd_daily.local.yaml
# 编辑 local YAML，将 model.release_id 替换成已验证 release 的 64 位 ID

set -a
source .env.local
set +a

uv run facdigger production plan \
  --config configs/production/eodhd_daily.local.yaml

uv run facdigger production bootstrap \
  --config configs/production/eodhd_daily.local.yaml

docker compose build
docker compose up -d
docker compose logs -f facdigger-production
```

Docker 构建上下文排除本地 `data/`、`artifacts/`、`.env*`、测试和文档；真实资产只通过
运行时 bind mount 进入容器，不会被烘焙进镜像层。默认 Compose 在容器内使用 root，以兼容
macOS Docker Desktop 与已有宿主目录的 bind-mount 写权限；若部署到 Linux 服务器，应在
确认卷 UID/GID 后显式设置 `user:`，不要通过放宽目录到全局可写来解决权限问题。

`eodhd_daily.local.yaml` 被 `.gitignore` 的 `*.local.yaml` 规则覆盖。不要在其中放 token
（token 只在 `.env.local`），并确认提交前 `git status`。容器启动前 shell
必须已有 `EODHD_API_TOKEN`。状态与健康检查：

```bash
docker compose exec facdigger-production facdigger production status \
  --config /app/configs/production/eodhd_daily.local.yaml

docker compose exec facdigger-production facdigger production health \
  --config /app/configs/production/eodhd_daily.local.yaml
```

`production tick` 只用于人工单次诊断；正常运行只启动 `serve`，不要再配置第二个宿主调度器。

## 配置选择

| 配置 | 用途 | 不能代表什么 |
|---|---|---|
| `configs/base.yaml` | 环境、CUDA 和 PatchTST checkpoint 诊断 | 训练实验 |
| `configs/data/eodhd_free.yaml` | 两只股票、低成本 live API smoke | 横截面研究 |
| `configs/data/eodhd_all_world_pilot.yaml` | 当前 active 100 股票资源门禁 | 无存活偏差的研究 |
| `configs/data/eodhd_historical_liquid.yaml` | 历史动态 top-1000 主数据路径 | 自动 research-ready |
| `configs/data/eodhd_daily_production.yaml` | fresh bulk EOD 日常修订 | 全历史重新采集 |
| `configs/datasets/eodhd_historical_liquid_inference.yaml` | 冻结 scaler 的无标签推理快照 | 训练或标签评价 |
| `configs/inference/e3_historical_replay.example.yaml` | 固定 release 的全历史回测分片，适用全部已支持模型 | 严格样本外或 paper 信号 |
| `configs/experiments/*_smoke.yaml` | 快速端到端测试 | 正式模型结论 |
| `configs/experiments/*_paid_pilot.yaml` | 真实规模资源验证 | 多 seed 正式对照 |
| `configs/experiments/e1_random.yaml`、`e2_etth1.yaml`、`e3_financial_pretrain.yaml` | 完整模型配置 | 独立于 M6 的正式结论 |
| `configs/research/m6_eodhd_engineering.yaml` | 当前 walk-forward 主线 | 完成正式中性化后的研究 |
| `configs/production/eodhd_daily.example.yaml` | Docker 每日服务模板；本地副本固定 release ID | 以占位 ID 启动（会失败关闭） |

`configs/datasets/us_equities_daily_v1.yaml` 是 provider-neutral 标准表范例；EODHD 历史主线
使用 `configs/datasets/eodhd_historical_liquid.yaml`。

## 产物与安全边界

- `data/bronze/`：标准化来源表，删除后可能需要重新消耗 API 配额；
- `data/cache/`：EODHD 原始响应缓存，用于避免重复请求和离线重建；
- `data/state/`：本地调用预算状态；
- `data/snapshots/`、`data/walk_forward_snapshots/`：训练/研究的内容寻址快照；
- `data/inference_snapshots/`：绑定 ModelRelease、无标签且复用冻结 scaler 的推理快照；
- `artifacts/releases/`：绑定 checkpoint、配置、scaler、训练/run manifest 与 predictions
  身份的内部 ModelRelease；
- `artifacts/factor_batches/`：单日或原 predictions 适配器生成的独立 FactorBatch；
- `artifacts/factor_history/`：全历史回放的续跑清单及年度 FactorBatch 子目录；
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
5. 项目不包含交易下单、撮合、资金管理或组合约束优化；每日“生产服务”只负责生成因子。

## 开发与验证

```bash
uv lock --check
uv run ruff check .
uv run pytest
```

普通测试使用 fake transport 和临时目录，不访问 live EODHD、付费配额、模型网络或正式
holdout。开发规则见[贡献指南](CONTRIBUTING.md)，秘密和研究完整性要求见
[安全策略](SECURITY.md)。
