# HeyBoss 因子联调交接

本文是 FacDiggerNN 向 HeyBoss 交易项目的实施交接单。FacDiggerNN 只交付最终
FactorBatch 目录；HeyBoss 不读取训练数据、ModelRelease、checkpoint、scaler 或
predictions。双方应各自固定 Git commit，并用真实 FacDigger 产出的目录做契约测试，不能
各写一份“看起来相同”的测试 fixture 后就认为联调完成。

## 六项 FacDigger 收尾审计

| 原审计项 | 状态 | 当前实现与证据 |
|---|---|---|
| 1. 固化 ModelRelease | 完成 | `inference/releases.py` 绑定 clean Git、checkpoint protocol、配置、scaler、训练/run manifest 和原始 predictions；篡改与旧协议测试失败关闭 |
| 2. 分离训练/推理快照 | 完成 | `data/inference_snapshots.py` 独立生成无标签 features、inference index 和 delivery universe；训练 snapshot ID 与输入 snapshot ID 分离 |
| 3. 复用冻结 scaler | 完成 | 推理只调用 `apply_robust_scaler`；回归测试禁止 fit，并逐值证明同一原始区间的训练/推理 features 完全一致 |
| 4. 统一因子帧 | 完成 | 旧研究 replay factors 已移除；评价回放和日常 signal 均调用唯一 `build_factor_frame`，严格输出五列 |
| 5. 原子 FactorBatch publisher | 完成 | Parquet 写入后重新读取校验，manifest 最后写，目录原子 rename；幂等、语义/文件篡改和失败清理均有测试 |
| 6. E3 集成回放 | 完成但有数据前置 | `factor-batch from-predictions` 只转换 ModelRelease 已绑定的原始 E3 predictions；旧 `artifacts2` 不满足当前协议，故被有意拒绝 |

第 6 项没有增加 legacy bypass：已有旧结果可保留为研究证据，但实际跨仓联调要等待下一轮
完整日、正确 mask、完整谱系绑定的 E3 结果。

## 1. 当前边界与前置结论

FacDiggerNN 已提供两种来源、同一种文件契约：

- `signal_inference`：单日完整候选横截面；“候选”指 FacDigger 当日 source universe，
  并不等于 HeyBoss 已配置标的集合；允许进入人工触发的 paper 验证；
- `evaluation_predictions`：历史评价样本中 eligible 且已有标签的行；只能用于隔离回测。

目录固定为：

```text
<delivery_id>/
├── factors.parquet
└── manifest.json
```

`factors.parquet` 的列、顺序和类型固定为：

| 列 | Arrow/Parquet 类型 | 规则 |
|---|---|---|
| `security_id` | string | 稳定主身份，EODHD production release 使用 `eodhd:isin:*` |
| `symbol` | string | 只用于显示和审计，不能作为自动映射键 |
| `asof_date` | date32/date | 信息截止交易日 |
| `score` | float64 nullable | eligible 时有限且非空；否则必须为空 |
| `eligible` | bool | 是否进入该日横截面排序 |

主键是 `(security_id, asof_date)`，文件按 `(asof_date, security_id)` 升序排列，不能含
`target`、split、未来收益、模型特征或中性化输入。

旧实施材料中“`delivery_id` 等于 Parquet SHA-256”的约定已经废止。只哈希文件无法阻止同一
文件被搭配不同模型、输入或交易时间语义。当前规则是：

```python
identity = dict(parsed_manifest)
identity.pop("created_at")
identity.pop("delivery_id")
canonical = json.dumps(
    identity,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
delivery_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

`artifact.sha256` 仍单独校验 `factors.parquet` 字节。目录名、manifest.delivery_id、上述语义
哈希三者必须相等。

`input.universe_sha256` 由因子文件四列
`security_id,symbol,asof_date,eligible` 的规范有序行计算。每行编码成无空格 UTF-8 JSON
数组（日期为 ISO 字符串），再追加一个换行并依次送入 SHA-256：

```python
digest = hashlib.sha256()
for security_id, symbol, asof_date, eligible in canonical_rows:
    line = json.dumps(
        [security_id, symbol, asof_date.isoformat(), eligible],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest.update(line.encode("utf-8"))
    digest.update(b"\n")
universe_sha256 = digest.hexdigest()
```

HeyBoss 应从收到的 Parquet 独立重算，不能只检查字段格式。

## 2. HeyBoss 当前实现的 P0 差异

2026-08-12 对 HeyBoss 工作树做只读核查后，已有 importer、`FactorScoreData`、NT Catalog、
`PatchTSTFactorActor`、backtest/paper 分区和相关测试，主链路不需要重写。但当前
`trading_assistant/data/factor.py` 仍实现旧契约，会拒绝所有现行 FacDigger 批次：

1. 仍要求 `delivery_id == artifact.sha256`，应改为上面的完整语义哈希；
2. `model` 缺少必需的 `score_semantics`；
3. `input` 缺少 `snapshot_manifest_sha256`、`universe_sha256`、`identity_policy`；
4. `time` 缺少 `calendar_version`；
5. `coverage` 仍使用 `expected_rows/actual_rows/ratio`，现行字段是
   `candidate_rows/actual_rows/expected_eligible_rows/scored_eligible_rows/`
   `missing_eligible_rows/ratio`；
6. finalized 目录检查不应忽略目录内隐藏文件；目录内必须严格只有两个契约文件。消费者只
   忽略交付根目录下尚未 rename 的 `.tmp-factor-batch-*` 兄弟目录。

现行 manifest 的严格字段如下：

| 节点 | 字段 |
|---|---|
| `source` | `kind, repository, commit, run_id, run_manifest_sha256` |
| `model` | `release_id, model_id, model_type, checkpoint_sha256, training_dataset_id, higher_score_is_better, forecast_horizon_sessions, score_semantics` |
| `input` | `snapshot_id, snapshot_manifest_sha256, universe_semantics, universe_sha256, identity_policy` |
| `time` | `calendar, calendar_version, timezone, minimum_asof_date, maximum_asof_date, signal_available, earliest_execution` |
| `coverage` | `candidate_rows, actual_rows, expected_eligible_rows, scored_eligible_rows, missing_eligible_rows, ratio` |
| `artifact` | `file, sha256, bytes, row_count, date_count` |

HeyBoss 应继续失败关闭并只接受：

- `model_type=financial_pretrained_patchtst`；
- `score_semantics=raw_cross_sectional_rank_score`（中性化上线前）；
- `higher_score_is_better=true`；
- `calendar=US_EQUITIES_REGULAR` 且显式记录非空 `calendar_version`；
- `timezone=America/New_York`；
- `signal_available=after_regular_session_close`；
- `earliest_execution=next_regular_session_open`；
- `candidate_rows == actual_rows`、eligible 两个计数相等、`missing_eligible_rows=0`、
  `ratio=1.0`；
- source kind 与 universe semantics 一致；`signal_inference` 只能包含一个日期。

不要在 HeyBoss 增加 predictions reader、旧格式 fallback 或 ModelRelease loader。

## 3. HeyBoss 项目侧实施任务

建议按以下四个小提交实施，任一步失败都不要继续到 paper：

建议 HeyBoss 侧直接把本文件作为任务输入，并以 FacDigger
`codex/factor-batch-v1-release` 分支最终提交的 commit 为生产者基线；不要继续以原
`docs/factor-integration.md` 中“delivery ID 等于 Parquet hash”的旧段落为实现依据。待本
分支合并到 `develop` 后，可把基线改为对应 merge commit，但双方联调记录仍应保存精确
commit，而不只写分支名。

### H1：同步严格消费者契约

- 更新 `src/trading_assistant/data/factor.py` 的严格 key/type/enum/coverage 校验；
- 实现与 FacDigger 完全一致的 semantic delivery ID；
- 保留 artifact hash/bytes/row/date/schema/null/sort/unique 校验；
- 从 Parquet 独立重算 `input.universe_sha256`；
- `source.commit` 校验为 40 位小写十六进制，所有 SHA-256 为 64 位小写十六进制；
- 更新 `docs/factor-integration.md`，删掉旧 hash 和旧字段说明。

最低测试：真实 FacDigger bundle 可读；修改 model/input/time/coverage 任一语义但不改
delivery ID 时拒绝；修改 Parquet、增加目录内文件、缺字段、错误类型或 source/semantics
组合时拒绝。

FacDigger 侧本阶段没有可提交的真实新协议 E3 bundle；HeyBoss 在 H1 单元测试阶段可以把
本契约表和哈希算法翻译成自己的本地 fixture，但该 fixture 只能验证消费者行为。第一次
跨仓验收必须直接复制 FacDigger 下一轮实验生成的原始 `<delivery_id>/` 目录，并同时在两
边运行 verifier/importer，不能把两个仓库分别生成的 fixture 当作联调证据。

### H2：建立明确的身份映射与价格前置条件

当前 HeyBoss 默认清单主要是 ETF，而 FacDigger 的 production universe 是美国普通股，不能
假设存在交集。先从一个真实 FactorBatch 选择 5–20 只 HeyBoss 可交易普通股：

1. 从 Parquet 读取确切 `security_id/symbol`；
2. 人工核对 ISIN、canonical instrument、IBKR route 和上市生命周期；
3. 在 `config/instruments.yaml` 显式添加唯一 `factor_security_id`；
4. 先把相同 as-of 日期范围的 INTERNAL signal bars 和 execution bars 写入同一 Catalog；
5. 禁止按 ticker 自动猜测 ISIN，也不能把未映射行静默映射到同名证券。

生产批次只要求每个“HeyBoss 已配置且当日 active”的证券在 FacDigger 完整横截面中存在；
HeyBoss 不必一次映射 top-1000 全部股票。eligible 行缺 signal bar 必须停止导入。

### H3：导入与 Actor 隔离回测

先使用 `evaluation_predictions` bundle：

```bash
uv run --frozen --env-file .env python scripts/import_factor_bundle.py \
  /path/to/artifacts/factor_batches/<delivery_id>
```

然后把独立回测配置的 `active_strategy` 切到 `patchtst_e3`，仅在该隔离配置中设置
`allow_evaluation_predictions: true`，并让 `data_start/evaluation_start/end` 覆盖因子日期：

```bash
uv run --frozen --env-file .env python scripts/run_backtest.py
```

验收证据至少包括：

- importer 首次导入行数/日期/标的数正确，重复导入为 no-op；
- 同日期不同 delivery 或不同 score 不覆盖 Catalog，而是失败；
- 每个日期只有完整 batch 才触发 Actor；
- ineligible 行不参加排序，score 最高方向与权重方向一致；
- `TradeSignalEvent`、风险限制、订单/成交和回测报告沿现有统一链路产生；
- `evaluation_predictions` 在默认配置和 paper runner 中均被拒绝。

### H4：单日生产语义 dry run

新实验产生合格 E3 release 后，由 FacDigger 发布一个 `signal_inference` 单日批次。HeyBoss：

1. 先同步当日价格并确认 Catalog 中每个 eligible 映射标的都有 signal bar；
2. 导入 batch，验证完整 mapped cross-section；
3. 重启/启动 TradingNode，用 Catalog bootstrap 读取该 batch；
4. 先让执行停留在 dry-run/人工审批边界，核对目标权重、release ID、as-of 日期、过期时间
   和 IBKR route；
5. 只有上述证据稳定后才在 paper 账户批准一次小规模订单。

任何缺批、陈旧日期、身份缺失、价格缺失、冲突批次或 source kind 错误，都应当天不产生
新信号，不能沿用上一日因子。

FacDigger 的 `signal --asof latest` 以 `delivery_universe` 的最新候选交易日为准，不以“最近
仍有 eligible 模型窗口的日期”为准。若当日完整候选横截面存在但全部不可评分，会发布该日
`eligible=false, score=null` 的无信号批次；不会静默回退到上一交易日。若指定日期根本不在
候选 universe 中则失败关闭。HeyBoss 应把无信号批次当作显式 flat/no-new-target 语义，并仍
执行过期与完整性检查。

## 4. FacDigger 侧交付命令

完成新协议 E3 实验后固定并审阅 source run 记录的 clean commit。发布工具自身也必须位于
clean 提交；命令可以运行在之后的代码提交，但 `source.commit` 始终来自训练 run，不得改写
成发布时 commit：

先从冻结 research 结果中选定准备部署的 E3 run，并把“为何选这个 fold/seed/refit cell”
作为人工发布决策记录下来。当前 `release create` 只负责验证并冻结指定 run，不替代模型选择，
也不会自动从多 seed 矩阵挑最好结果；不要事后按 holdout 或 HeyBoss 回测收益挑 seed。

```bash
uv run facdigger release create \
  --run artifacts/e3/<run_id> \
  --output-root artifacts/releases

uv run facdigger release verify \
  --release artifacts/releases/<release_id>
```

隔离回放：

```bash
uv run facdigger factor-batch from-predictions \
  --predictions artifacts/e3/<run_id>/predictions.parquet \
  --release artifacts/releases/<release_id> \
  --output-root artifacts/factor_batches
```

单日生产语义的底层人工诊断命令仍可用：

```bash
uv run facdigger dataset build-inference \
  --config configs/datasets/eodhd_historical_liquid_inference.yaml \
  --release artifacts/releases/<release_id>

uv run facdigger signal \
  --release artifacts/releases/<release_id> \
  --dataset data/inference_snapshots/<snapshot_id> \
  --output-root artifacts/factor_batches \
  --asof latest

uv run facdigger factor-batch verify \
  --bundle artifacts/factor_batches/<delivery_id>
```

正常每日生产不人工串联上述命令，也不使用 macOS LaunchAgent。FacDigger 由 Docker
容器内的 `production serve` 统一完成 EODHD 修订、目标日 inference snapshot、固定
ModelRelease 推理、截止前原子发布和状态落库：

```bash
cp configs/production/eodhd_daily.example.yaml \
  configs/production/eodhd_daily.local.yaml
# 在 local YAML 中显式写入经审阅的 release_id；token 仍只放 .env.local

set -a && source .env.local && set +a
uv run facdigger production plan \
  --config configs/production/eodhd_daily.local.yaml
uv run facdigger production bootstrap \
  --config configs/production/eodhd_daily.local.yaml
docker compose up -d --build
```

服务在每个交易日 19:00 ET 首次尝试，未就绪时每 30 分钟重试，下一 regular session
09:30 ET 截止。只有 D 完整批次才发布，禁止旧因子回退。推理快照保留最近 10 个交易日，
FactorBatch 永久保留；`data/snapshots/` 和 `data/walk_forward_snapshots/` 永不由服务
修改。HeyBoss 只监视已完成的
`artifacts/factor_batches/<delivery_id>/`，不读取 FacDigger SQLite、source store 或隐藏临时目录。

每次交接记录 FacDigger commit、release ID、delivery ID、source kind、日期范围、Parquet hash、
row/date/eligible counts、被批准的 E3 run/seed 选择依据和 HeyBoss commit。只复制完整
`<delivery_id>/` 目录；不要重新序列化 Parquet 或手改 manifest。

## 5. 本轮不能直接使用 artifacts2

现有 `artifacts2` 来自关键修复前的训练协议：它不能证明完整日 objective、正确 patch-mask
对齐以及当前 scaler/predictions 绑定。ModelRelease 会有意拒绝这类旧 run，FactorBatch 回放
入口也没有 legacy bypass。这不是联调功能缺失，而是避免把已知不可靠权重送进交易系统。

因此真实跨仓流程测试的首个数据前置条件是：完成下一轮新协议 E3 训练，并从记录 clean
commit 的原始 run 创建 ModelRelease。在此之前 HeyBoss 可以完成 H1/H2 和使用 FacDigger
测试生成的小型契约 bundle 做 importer 测试，但不能把该 fixture 当成策略效果证据。

## 6. 联调完成定义

满足以下全部条件才可称为“流程跑通”：

- 两个仓库均固定具体 commit，测试全绿；
- FacDigger 对真实 bundle 的 `factor-batch verify` 通过；
- HeyBoss 用同一原始目录通过严格 semantic/artifact/schema 校验；
- 至少一个 evaluation bundle 完成 Catalog→Actor→TradeSignalEvent→风险→模拟成交→报告；
- 至少一个 signal bundle 完成价格前置检查、Catalog 导入、Actor bootstrap 和人工审批前 dry run；
- 重复导入/重启幂等，冲突输入失败关闭；
- paper 未开启 `allow_evaluation_predictions`，且未自动沿用旧信号；
- 日志能用 delivery ID、release ID、as-of 日期和 rebalance key 跨系统追溯。

这一定义只证明工程链路和语义一致，不证明因子有效，也不替代新一轮 walk-forward 统计门禁、
组合风险评审和 paper 观察期。
