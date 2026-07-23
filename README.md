# FacDiggerNN

FacDiggerNN 是面向美股日频数据的、强调 point-in-time 语义和可复现性的机器学习因子研究工具。M0 的 PatchTST 兼容性探针、M1 标准 Parquet 数据闭环、M2 E0 基线评价、M3 E1 随机 PatchTST、M4 E2 跨域迁移、M5 E3 金融域预训练、M6 walk-forward 研究冻结、M7 checkpoint 回放，以及 M8 最新信号/独立评价闭环已经可运行。

完整设计见 [实施设计](docs/IMPLEMENTATION_PLAN.md)。

## 开发环境

推荐 Python 3.11，并使用独立虚拟环境。不要复用已有的全局 Conda 环境。

```bash
uv sync --extra model --extra data --extra eodhd --extra dev
source .venv/bin/activate
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

`uv.lock` 是首选的跨平台锁文件；`requirements-lock.txt` 由同一锁文件导出。当前已在 macOS/CPU 上验证 PyTorch 2.13.0 与 Transformers 4.57.6 的 checkpoint 加载、前反向和恢复。Windows + RTX 2070 Super 仍需补跑 CUDA/FP16 冒烟。

## M0 命令

```bash
# 检查 Python、依赖导入和计算设备
facdigger doctor

# 创建包含配置、Git 和环境信息的运行清单
facdigger manifest --config configs/base.yaml --output artifacts

# 下载并验证 IBM ETTh1 checkpoint；首次运行需要网络
facdigger probe-patchtst --config configs/base.yaml --output artifacts/m0-probe

# 只使用本机缓存，不访问网络
facdigger probe-patchtst --config configs/base.yaml --output artifacts/m0-probe --local-files-only

# 普通单元测试不要求安装 torch/transformers
python -m pytest
```

## M1 标准 Parquet 数据闭环

首版数据入口不绑定供应商。将供应商数据转换成标准 Parquet 后，运行：

```bash
facdigger data validate --config configs/datasets/us_equities_daily_v1.yaml
facdigger dataset build --config configs/datasets/us_equities_daily_v1.yaml
```

用免费版的一年历史做端到端冒烟时，可改用专门的短窗口配置：

```bash
facdigger dataset build --config configs/datasets/eodhd_free_smoke.yaml
```

该配置的 `context_length=60` 只用于验证管线。目标实验仍要求 512 个交易日上下文，因此免费版的一年历史不能直接承担最终 E0—E3 对照训练。

### EODHD 接入

EODHD 只存在于 provider 层，输出仍为同一套 `bars_daily.parquet` 和
`universe_daily.parquet`；特征、标签和模型不导入任何 EODHD 类型。免费版目前只有很低的每日调用额度，因此客户端默认启用响应缓存和本地每日预算保护。

```bash
# 可先使用官方 demo token 验证响应格式
facdigger data probe --config configs/data/eodhd_free.yaml

# 使用自己的 token（不要写进 YAML 或提交仓库）
export EODHD_API_TOKEN='...'
facdigger data ingest --config configs/data/eodhd_free.yaml

# 验证标准化结果并继续构建数据集
facdigger data validate --config configs/datasets/us_equities_daily_v1.yaml
facdigger dataset build --config configs/datasets/us_equities_daily_v1.yaml
```

当前 EODHD EOD 适配有几项明确边界：

- 原始 OHLC 保持不变，`adj_factor = adjusted_close / close`；该因子同时包含拆股和分红影响。
- 免费版优先使用显式小股票列表，避免逐股请求耗尽每日额度；缓存命中不计入本地预算。
- 日线不能可靠恢复历史停牌、历史行业和流通市值，这些字段会保留缺失或带质量标记。
- 退市列表不含可直接用于收益标签的可靠终值，因而不会伪造 `delistings.parquet`；未来窗口超出最后行情时标签自然为空。
- `security_id` 优先使用 ISIN；缺少元数据时退化为供应商 ticker，并在 manifest 中告警。

### EODHD 付费历史数据 pilot

token 放在仓库根目录的 `.env.local` 中，文件权限设为 `600`；该文件已被 Git 忽略。每个新终端先加载环境变量：

```bash
set -a
source .env.local
set +a
```

付费 pilot 会从 US active common-stock metadata 与最新 bulk EOD 联结，只保留 Nasdaq、NYSE 和 NYSE American，再按 `close × avgvol_200d` 选取 100 个标的。bulk 请求按官方的 100 API calls 权重计入本地预算，缓存命中不重复计费；`allow_demo_token: false` 防止真实任务静默退回 demo 数据。

```bash
facdigger data probe --config configs/data/eodhd_all_world_pilot.yaml
facdigger data ingest --config configs/data/eodhd_all_world_pilot.yaml
facdigger data validate --config configs/datasets/eodhd_all_world_pilot.yaml
facdigger dataset build --config configs/datasets/eodhd_all_world_pilot.yaml
```

数据快照会复制并哈希 `source_manifest.json`，使 provider 请求、筛选规则、预算和警告进入训练血缘。当前流动性排名使用“当前仍活跃股票”和当前成交数据，因此存在存活偏差与历史前视偏差，只用于 100 股票工程/资源门禁。评价报告会保留统计指标，但强制标记 `source_research_ready=false`；正式研究股票池仍需纳入历史退市证券并按当日信息定义 eligibility。

## M2 E0 基线与统一评价

数据快照 schema v3 会固化 `sample_metadata.parquet`，并新增完全不含标签或 split 的 `inference_index.parquet`。前者确保行业、市值和 eligible 等评价暴露不需要回读可变的原始数据；后者覆盖所有具备完整上下文的 eligible 日期，包括未来收益尚未形成的最新交易日。默认只允许评价 validation；要读取 test，配置必须同时设置 `evaluation_split: test` 和 `unlock_test: true`。

```bash
facdigger train e0 \
  --config configs/experiments/e0_mlp_smoke.yaml \
  --dataset data/snapshots/<dataset_id>

facdigger train e0 \
  --config configs/experiments/e0_lightgbm_smoke.yaml \
  --dataset data/snapshots/<dataset_id>

facdigger compare \
  --runs artifacts/e0/<lightgbm_run>,artifacts/e0/<mlp_run> \
  --output artifacts/e0/comparison
```

每个 run 包含 checkpoint、`predictions.parquet`、`metrics.json`、`report.html`、resolved config 和 manifest。评价器统一计算逐日 IC/RankIC、ICIR、高低分组收益、换手、0/10/20/50 bps 成本情景、年度/行业/市值稳定性以及行业和点时市值中性化。缺少点时行业或市值时，中性化结果保持为空，不能用原始分数冒充。

## M3 E1 随机 PatchTST

E1 复用同一份不可变快照和 evaluator。窗口按需从列式特征读取，缺失值以零填充并单独传递 observed mask；模型为随机初始化的 PatchTST encoder 加 AlphaHead。训练 checkpoint 包含模型、optimizer、scheduler、GradScaler、epoch/global step、RNG 和按日期 sampler 状态。

```bash
facdigger train e1 \
  --config configs/experiments/e1_random_smoke.yaml \
  --dataset data/snapshots/<dataset_id>

# 仅恢复 status=failed/running 且数据集、完整配置哈希一致的 run
facdigger train e1 \
  --config configs/experiments/e1_random_smoke.yaml \
  --dataset data/snapshots/<dataset_id> \
  --resume artifacts/e1/<run_id>/checkpoints/last.pt
```

CUDA 上可配置 `precision: fp16`；CPU 会使用 FP32。当前免费 EODHD 冒烟配置只覆盖 AAPL/TSLA 和一年历史，目的仅是验证 train → predict → report，不可据此判断因子收益或 PatchTST 相对 E0 的研究优势。正式 E1 应使用设计文档中的 512 日上下文、完整股票池和多 seed/walk-forward 协议。

付费 100 股票、512 日上下文的资源门禁配置为：

```bash
facdigger train e0 \
  --config configs/experiments/e0_lightgbm_paid_pilot.yaml \
  --dataset data/snapshots/<paid_dataset_id>

facdigger train e1 \
  --config configs/experiments/e1_random_paid_pilot.yaml \
  --dataset data/snapshots/<paid_dataset_id>
```

该 E1 配置仅训练两轮并缩小 hidden size/depth，用于证明真实规模下的数据吞吐、checkpoint 和预测完整性，不替代正式的 6-layer、多 seed E1 实验。

## M4 E2 ETTh1 encoder 迁移

E2 固定使用 `ibm-research/patchtst-etth1-pretrain` 的 commit revision。加载分两段执行：原始 checkpoint → 当前 Transformers source backbone → 金融 Alpha backbone；每段都按规范化名称和精确 shape 匹配，并强制 loaded-numel ratio 门槛。任意未列入 allowlist 的 missing、unexpected 或 shape mismatch 都会在训练前阻断。

```bash
facdigger train e2 \
  --config configs/experiments/e2_etth1.yaml \
  --dataset data/snapshots/<512_session_dataset_id>

# 中断后仅允许相同 dataset、完整 config 和 source weight hash 恢复
facdigger train e2 \
  --config configs/experiments/e2_etth1.yaml \
  --dataset data/snapshots/<512_session_dataset_id> \
  --resume artifacts/e2/<run_id>/checkpoints/last.pt
```

训练阶段固定为：

- FT-0：冻结整个 encoder，仅训练 AlphaHead；backbone 保持 eval，BatchNorm buffer 也不得变化。
- FT-1：只解冻最后 `N` 个 encoder blocks，使用独立的 encoder/head learning rate。

每个阶段第一次 optimizer step 前后都会计算 encoder 和 head 的完整参数/缓冲区指纹。FT-0 要求 encoder 不变、head 改变；FT-1 要求两者都改变。run 额外输出 `weight_load_report.json`，checkpoint 保存 source hash、加载报告、阶段审计、optimizer/scheduler/GradScaler、RNG 和 sampler 状态。

`configs/experiments/e2_etth1_smoke.yaml` 只运行一个 FT-0 和一个 FT-1 epoch，用于工程门禁；正式配置为 `configs/experiments/e2_etth1.yaml`。两者都默认只读取 validation。

## M5 E3 金融域 masked-patch 预训练

E3 的初始化链固定为 `ETTh1 encoder → 金融域 masked reconstruction → AlphaHead`。金融预训练只从正式 `train` split 取窗口，并在该区间内部按日期切出尾部 10% 作为重建 checkpoint 选择段；正式 validation 和 test 的使用行数都必须为 0。损失只聚合“随机遮蔽且真实观测”的 patch 元素，不把缺失填充值作为重建目标。

```bash
facdigger train e3 \
  --config configs/experiments/e3_financial_pretrain.yaml \
  --dataset data/snapshots/<512_session_dataset_id>

# 两个阶段都可精确恢复；使用失败 manifest 指向的 last.pt
facdigger train e3 \
  --config configs/experiments/e3_financial_pretrain.yaml \
  --dataset data/snapshots/<512_session_dataset_id> \
  --resume artifacts/e3/<run_id>/pretraining/checkpoints/last.pt
```

预训练 checkpoint 保存重建模型、独立 encoder state、optimizer、scheduler、GradScaler、RNG、sequence sampler、切分泄漏审计和 ETTh1 权重哈希。选中的金融 encoder 以 100% 参数量、零未授权 mismatch 的门槛载入新 Alpha 模型，之后复用 M4 完全相同的 FT-0/FT-1 协议和统一 evaluator。`e3_financial_pretrain_smoke.yaml` 只用于一轮预训练加两轮微调的工程门禁。

## M6 Walk-forward 与研究冻结

M6 在 E0–E3 之上增加编排层，不修改单次实验训练器。每个 fold 都从同一基础数据配置重新生成内容寻址快照，因此特征 scaler 只拟合该 fold 的扩展 train 区间；同一 fold 的四个模型与全部 seed 必须共享 dataset_id 和完全相同的预测样本键。

```bash
# 只校验 3 folds × 3 seeds × 4 models 的协议，不构建数据或启动训练
facdigger research plan --config configs/research/m6_eodhd_engineering.yaml

# 正式实验前检查来源 readiness、日期覆盖和全部配置；不构建、不训练
facdigger research preflight --config configs/research/m6_eodhd_engineering.yaml

# 执行 validation 矩阵；结束后生成 freeze.json，但不会读取 test
facdigger research run --config configs/research/m6_eodhd_engineering.yaml

# 中断后复用已完成 cell，并恢复存在 checkpoint 的失败 cell
facdigger research run \
  --config configs/research/m6_eodhd_engineering.yaml \
  --resume-run artifacts/research/<research_run_id>

# 仅在审阅 validation/research.html 后执行；只解封最后一折的 test
facdigger research run \
  --config configs/research/m6_eodhd_engineering.yaml \
  --resume-run artifacts/research/<research_run_id> \
  --unlock-final-holdout
```

validation 完成后会固化配置、fold 计划、完整 cell 矩阵和研究报告哈希。解封 holdout 前这些哈希必须完全一致。统计报告先按日期对多个 seed 求均值，再在 fold 内计算 Newey–West/HAC 和固定 offset 的非重叠样本推断，并分别回答 E1−E0 架构增量、E2−E1 外域迁移增量、E3−E2 金融预训练增量及 E3 中性化/成本后是否可用。

当前 `m6_eodhd_engineering.yaml` 仍基于 current-active/current-liquidity pilot 股票池。其 source provenance 会明确阻断 research readiness，因此即使统计指标为正，最终结论仍应为 `no_go`；该配置用于完整工程与资源验证，不能替代纳入历史退市证券的正式研究股票池。

## M7 Checkpoint 回放与因子导出

训练结束后的预测不再依赖仍驻留在内存中的模型。`predict` 会从完整 run 中重新校验 resolved config、dataset manifest、checkpoint 和 E0 LightGBM preprocessing sidecar 的哈希，然后独立重建 E0–E3 模型。默认回放 source run 原本的 evaluation split，并要求重新计算的 raw score 与原 `predictions.parquet` 在固定容差内一致。

```bash
# 回放原 validation，默认使用 CPU，并写入 <run>/replays/<replay_id>
facdigger predict --run artifacts/e3/<run_id>

# 快照移动后可指定内容完全相同的 dataset 目录
facdigger predict \
  --run artifacts/e1/<run_id> \
  --dataset data/snapshots/<same_dataset_id> \
  --output artifacts/factor_exports/<export_id>

# test 仍需显式解封；M6 正式研究应优先通过 research holdout 命令执行
facdigger predict \
  --run artifacts/e3/<run_id> \
  --split test \
  --unlock-test
```

每次回放原子生成 `manifest.json`、完整评价用 `predictions.parquet`、不含未来标签的 `factors.parquet`、`metrics.json` 和 `report.html`。因子文件声明 `signal_available=after_close` 与 `earliest_execution=next_session_open`；任何已有输出目录都不会被覆盖。LightGBM checkpoint 在隔离进程中加载，避免 macOS 上与 Polars/PyTorch 的 OpenMP runtime 冲突。

## M8 最新信号与独立评价

`signal` 只读取 schema-v3 的 `features.parquet` 和 `inference_index.parquet`，不会读取 `labels.parquet`、目标值或 test split 归属。它支持最新日期、单个历史日期或闭区间，并且只接受 source run 对应的同一内容寻址快照，避免缩放器或特征定义漂移。

```bash
# 最新一个可推理交易日
facdigger signal --run artifacts/e3/<run_id>

# 指定一个历史日期
facdigger signal \
  --run artifacts/e1/<run_id> \
  --asof 2026-01-30 \
  --output artifacts/signals/2026-01-30

# 在完全不加载模型的进程中复核已有 prediction 表
facdigger evaluate \
  --predictions artifacts/e3/<run_id>/predictions.parquet \
  --dataset data/snapshots/<same_dataset_id> \
  --output artifacts/evaluations/<evaluation_id>
```

`factors.parquet` 保留 raw score、可用时的行业/市值中性分数、模型/checkpoint/dataset 血缘，以及 `after_close → next_session_open` 时点声明，但绝不包含 target。`evaluate` 会逐键核对不可变快照里的 target、强制覆盖率门禁，并独立生成 metrics、HTML report 与输入哈希清单。

最低输入包括：

- `bars_daily`：稳定 `security_id`、当日 ticker、session 日期、OHLCV、美元成交额、调整因子和数据版本；
- `universe_daily`：每个证券—session 的上市、退市、停牌、主上市、证券类型、行业、市值、流动性和 eligible 状态；
- `corporate_actions`（可选）：ex-date、价格/成交量调整因子、现金金额和可知时间；
- `delistings`（可选）：退市日、最后交易日、退市收益或终值。配置了文件时将严格校验，不能静默缺失终值。

输出是以内容哈希命名的不可变目录，包含 `features.parquet`、`labels.parquet`、`sample_index.parquet`、`sample_metadata.parquet`、`inference_index.parquet`、只用 Train 区间拟合的 `scaler.json`、`audit.json` 和 `manifest.json`。移动相同输入文件或更换输出目录不会改变 `dataset_id`。

探针只有同时满足以下条件才成功：

- checkpoint 到当前 Transformers 类的 encoder 参数加载率不低于 80%；
- source encoder 到目标 backbone 的迁移率不低于 80%；
- 不存在未加入 allowlist 的不匹配键；
- `[B, 512, 7]` forward/backward 成功；
- checkpoint 保存和恢复成功。
