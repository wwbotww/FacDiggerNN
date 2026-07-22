# FacDiggerNN

FacDiggerNN 是面向美股日频数据的、强调 point-in-time 语义和可复现性的机器学习因子研究工具。M0 的 PatchTST 兼容性探针、M1 标准 Parquet 数据闭环和 M2 E0 基线评价闭环已经可运行。

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

## M2 E0 基线与统一评价

数据快照 schema v2 会额外固化 `sample_metadata.parquet`，确保行业、市值和 eligible 等评价暴露不需要回读可变的原始数据。默认只允许评价 validation；要读取 test，配置必须同时设置 `evaluation_split: test` 和 `unlock_test: true`。

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

最低输入包括：

- `bars_daily`：稳定 `security_id`、当日 ticker、session 日期、OHLCV、美元成交额、调整因子和数据版本；
- `universe_daily`：每个证券—session 的上市、退市、停牌、主上市、证券类型、行业、市值、流动性和 eligible 状态；
- `corporate_actions`（可选）：ex-date、价格/成交量调整因子、现金金额和可知时间；
- `delistings`（可选）：退市日、最后交易日、退市收益或终值。配置了文件时将严格校验，不能静默缺失终值。

输出是以内容哈希命名的不可变目录，包含 `features.parquet`、`labels.parquet`、`sample_index.parquet`、`sample_metadata.parquet`、只用 Train 区间拟合的 `scaler.json`、`audit.json` 和 `manifest.json`。移动相同输入文件或更换输出目录不会改变 `dataset_id`。

探针只有同时满足以下条件才成功：

- checkpoint 到当前 Transformers 类的 encoder 参数加载率不低于 80%；
- source encoder 到目标 backbone 的迁移率不低于 80%；
- 不存在未加入 allowlist 的不匹配键；
- `[B, 512, 7]` forward/backward 成功；
- checkpoint 保存和恢复成功。
