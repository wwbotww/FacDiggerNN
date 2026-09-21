# HeyBoss 因子联调交接

本文是 FacDiggerNN 向 HeyBoss 交易项目的实施交接单。FacDiggerNN 只交付最终的单个
FactorBatch 两文件目录；全历史回放的父 plan/state/manifest 只供 FacDigger 续跑，HeyBoss
不读取它，也不读取训练数据、ModelRelease、checkpoint、scaler 或 predictions。双方应各自固定
Git commit，并用真实 FacDigger 产出的目录做契约测试，不能
各写一份“看起来相同”的测试 fixture 后就认为联调完成。

当前状态（2026-09-20）：两侧日历统一、HeyBoss 持仓保护、空组合 SKIP、预期日期/固定
release 消费及审批恢复代码已完成，826 release 与原 validation 预测的历史交付已完成离线
联合验收，具体产物与证据见本文末尾。无需重新实施这些功能或重复创建同一 release。
FacDigger 已完成 fresh EODHD → Docker 无标签推理 → 2026-09-18 真实十只批次，固定 826
release、XOM 新 ISIN 及 HeyBoss parser 逐值只读校验通过。稳定目录、发布时间、耗时和
部署证据见[826 生产运行交接](826每日生产运行交接.md)。仍待验证的是 HeyBoss 真实配置
下的接纳、联合提前量和连续五日；本轮没有迁移或写入其运行库，没有启动交易，也不代表
因子收益有效。
[`局部缺分持仓保护交接`](HeyBoss局部缺分持仓保护交接.md)保留原设计和回归要求；本文
第 2 节旧消费者差异同样是历史背景，不是当前待重做清单。

## 六项 FacDigger 收尾审计

| 原审计项 | 状态 | 当前实现与证据 |
|---|---|---|
| 1. 固化 ModelRelease | 完成 | `inference/releases.py` 绑定真实 Git 来源、checkpoint protocol、配置、scaler、训练/run manifest 和原始 predictions；dirty 联调需显式开关，篡改与旧协议仍失败关闭 |
| 2. 分离训练/推理快照 | 完成 | `data/inference_snapshots.py` 独立生成无标签 features、inference index 和 delivery universe；训练 snapshot ID 与输入 snapshot ID 分离 |
| 3. 复用冻结 scaler | 完成 | 推理只调用 `apply_robust_scaler`；回归测试禁止 fit，并逐值证明同一原始区间的训练/推理 features 完全一致 |
| 4. 统一因子帧 | 完成 | 旧研究 replay factors 已移除；原 predictions、固定模型历史回放和日常 signal 均复用同一五列构造、评分边界和 publisher |
| 5. 原子 FactorBatch publisher | 完成 | Parquet 写入后重新读取校验，manifest 最后写，目录原子 rename；幂等、语义/文件篡改和失败清理均有测试 |
| 6. 多模型集成回放 | 完成但有数据前置 | E1—E3 / Finance Transformer 统一 release/runtime/FactorBatch；原 predictions 逐字节绑定，target-free 年度回放、续跑与逐分片复核复用每日评分；旧 `artifacts2` 仍被有意拒绝 |

第 6 项没有增加 legacy bypass：已有旧结果可保留为研究证据，但真实模型交付仍要求完整日、
正确 mask 和完整产物绑定。默认 clean Git；联调允许 `--allow-dirty` 并如实记录，不能伪造
干净来源。工程通路通过不等于模型质量通过。

## 1. 当前边界与前置结论

FacDiggerNN 已提供两种来源、同一种文件契约：

- `signal_inference`：单日完整交付候选横截面；使用交付 profile 时指双方事先声明的目标集合，
  不等于模型完整计算池，后者的可评分成员仍全部参与评分；真实每日接纳、行情准备与
  运行库迁移验证通过后，才进入人工触发的 paper 验证；
- `evaluation_predictions`：eligible 且已有分数的历史回测行；既可以来自 release 绑定的原始
  评价 predictions，也可以来自固定 release 对 target-free 历史 snapshot 的重新推理。该枚举
  表达 backtest-only 消费边界，不表示 FactorBatch 含有或读取了标签；只能用于隔离回测。

当前 `financial_pretrained_patchtst`（E3）、`random_patchtst`（E1）、
`etth1_transferred_patchtst`（E2）与 `finance_patch_transformer` 使用相同交付契约。
`model.model_type` 是 `[a-z][a-z0-9_]*` 格式的来源元数据；HeyBoss 导入器不再只接受 E3，
也不根据模型名称加载不同依赖。scratch / finance_pretrained 由各自 release 身份区分。
消费者仍严格校验 score 方向、horizon、可用时间、覆盖率、身份及全部语义/文件哈希。

2026-09-09 的跨仓文件验证已将 FacDigger 小型 CPU 实验实际生成的 8 个 E3/Finance 批次
（536 行，含每日、原评价和年度历史）直接导入 HeyBoss 临时 NT Catalog，并核对原分数与幂等性。
这是合成数据的文件/导入通路验收，不代表真实数据、策略收益或 paper 执行已验收。

Finance 模型的历史子集导出会先在快照完整日横截面上计算，再筛选 HeyBoss 的交付股票。
不得为了匹配 HeyBoss 小股票池先裁掉上游计算成员，也不得让 HeyBoss 重算特征或模型分数。

目录固定为：

```text
<delivery_id>/
├── factors.parquet
└── manifest.json
```

`factors.parquet` 的列、顺序和类型固定为：

| 列 | Arrow/Parquet 类型 | 规则 |
|---|---|---|
| `security_id` | string | 实际交付集合的稳定身份；ISIN 或经双方确认、具有明确有效期的可靠映射 |
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

## 2. 消费者契约与早期差异（旧差异已完成同步）

2026-08-12 对 HeyBoss 工作树做只读核查后，已有 importer、`FactorScoreData`、NT Catalog、
`PatchTSTFactorActor`、backtest/paper 分区和相关测试。当时的
`trading_assistant/data/factor.py` 仍实现旧契约，存在以下差异；现行导入器已经同步，
并通过上文的小型跨仓导入验证。此清单仅解释为何旧实施材料不能继续使用：

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

- `model_type` 为合法来源元数据，不按 E3 名称建立模型白名单；
- `score_semantics=raw_cross_sectional_rank_score`（中性化上线前）；
- `higher_score_is_better=true`；
- `calendar=US_EQUITIES_REGULAR` 且发布来源为 `calendar_version=exchange_calendars:4.13.2:XNYS`；
- `timezone=America/New_York`；
- `signal_available=after_regular_session_close`；
- `earliest_execution=next_regular_session_open`；
- `candidate_rows == actual_rows`、eligible 两个计数相等、`missing_eligible_rows=0`、
  `ratio=1.0`；
- source kind 与 universe semantics 一致；`signal_inference` 只能包含一个日期。

不要在 HeyBoss 增加 predictions reader、旧格式 fallback 或 ModelRelease loader。

## 3. HeyBoss 通路检查顺序与验收边界

以下 H1—H4 保留为部署/回归检查顺序，不是待编码清单。H1 契约、H2 日期身份解析及缺分
保护代码已完成；826 的隔离身份/价格配置与 H3 原 validation 预测回测已验收。更换交付范围
或运行环境须重新核对前置条件，固定模型全历史重新推理与 H4 真实每日/paper 仍需单独验证。
任一步失败都不要继续到 paper：

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

真实 bundle 必须来自通过 release 门禁的监督 run；HeyBoss 在 H1 单元测试阶段可以把
本契约表和哈希算法翻译成自己的本地 fixture，但该 fixture 只能验证消费者行为。第一次
跨仓验收必须直接复制 FacDigger 下一轮实验生成的原始 `<delivery_id>/` 目录，并同时在两
边运行 verifier/importer，不能把两个仓库分别生成的 fixture 当作联调证据。

### H2：建立明确的身份映射与价格前置条件

两个项目的股票池和证券生命周期不能假设相同。先约定 5–20 只 HeyBoss 可交易普通股，
再准备 FacDigger 的交付 profile，不从“本次成功映射的行”倒推目标集合：

1. 从 Parquet 读取确切 `security_id/symbol`；
2. 人工核对 ISIN、canonical instrument、IBKR route 和上市生命周期；
3. 在 `config/instruments.yaml` 显式添加唯一 `factor_security_id`；
4. 先把相同 as-of 日期范围的 INTERNAL signal bars 和 execution bars 写入同一 Catalog；
5. 禁止按 ticker 自动猜测 ISIN，也不能把未映射行静默映射到同名证券。

FacDigger 的 `targets` 定义目标和 active 日期，独立的 `identities` 记录源 ID、交付 ID、
HeyBoss instrument、明确的有效期与证据。模板见
[`heyboss_delivery.example.yaml`](../configs/inference/heyboss_delivery.example.yaml)。有效期是闭区间，
过期会使批次失败，不能静默漏交。ISIN 缺失可以经可靠映射补足；训练中不交付股票的 fallback
不阻止 release，也不会提前从模型计算池删除。

当前消费者的静态 `factor_security_id` 适合不变的身份；跨历史 ISIN 变更应配置已实现的
`factor_identity_periods`，按日期解析并验证唯一映射。代码支持不等于真实标的配置已补齐：
不得把历史 ISIN 改写成当前 ISIN 来通过静态匹配。FacDigger 不替 HeyBoss 猜测上市主体或
交易路由，也不修改其真实标的配置。

每个当日 active 的交付目标必须在候选表中存在；不要求 HeyBoss 映射完整 top-1000 计算池。
合法 ineligible 行仍交付为 null；eligible 行缺 signal/execution bar 必须停止导入。

### H3：导入与 Actor 隔离回测

优先使用 FacDigger `factor-history run` 生成的年度 `evaluation_predictions` 子目录。先导入
一个较短年份做 smoke；通过后按父 manifest 声明的年份升序，把每个原始
`factor_batches/<delivery_id>/` 目录逐个交给同一个 importer。父 manifest 只用于操作人员确认
分片齐全，不进入 HeyBoss Catalog：

```bash
uv run --frozen --env-file .env python scripts/import_factor_bundle.py \
  /path/to/artifacts/factor_history/<history_id>/factor_batches/<delivery_id>
```

也可以先用 `factor-batch from-predictions` 的原评价 split 做更小 smoke，但它通常只覆盖一个
validation/test 区间，不能代替全历史工程回测。随后把独立回测配置的 `active_strategy` 切到
`patchtst_e3`，仅在该隔离配置中设置
`allow_evaluation_predictions: true`，并让 `data_start/evaluation_start/end` 覆盖因子日期：

```bash
uv run --frozen --env-file .env python scripts/run_backtest.py
```

验收证据至少包括：

- importer 首次导入行数/日期/标的数正确，重复导入为 no-op；
- 全部年度 delivery 与 FacDigger 父 manifest 一一对应，缺一年或重复一年失败；
- 同日期不同 delivery 或不同 score 不覆盖 Catalog，而是失败；
- 每个日期只有完整 batch 才触发 Actor；
- ineligible 行不参加排序，score 最高方向与权重方向一致；
- `TradeSignalEvent`、风险限制、订单/成交和回测报告沿现有统一链路产生；
- `evaluation_predictions` 在默认配置和 paper runner 中均被拒绝。

这个回测只证明固定模型因子能够像普通策略数据一样贯通全历史价格、Catalog、Actor、风险与
报告。模型参数、scaler 和历史来源修订对早期日期可能含未来信息，禁止把收益曲线标注为样本外，也禁止按
这条曲线反向挑选 release、seed、日期范围或股票白名单。

### H4：单日生产语义 dry run

实验产生合格且经人工选择的 release 后，由 FacDigger 发布一个 `signal_inference` 单日批次。HeyBoss：

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
候选 universe 中则失败关闭。这是 `signal` 的诊断能力；每日生产服务的数量/比例门禁不会
发布全不可评分批次。HeyBoss 必须将其解释为 **SKIP／保持持仓**，不是 flat／清仓，也不能
发送空目标权重事件。仍需执行新鲜度与完整性检查，详见缺分保护交接。

## 4. FacDigger 侧交付命令

完成监督实验后固定并审阅 source run。默认要求训练和发布工作区 clean；当前联调可以显式
追加 `--allow-dirty` 同时放行两者，source.git_clean 会如实记录。命令可以运行在之后的代码
提交，但 `source.commit` 始终来自训练 run，不得改写
成发布时 commit：

先从 research 结果中选定准备使用的监督 run，并把“为何选这个模型/fold/seed/refit cell”
作为人工发布决策记录下来。当前 `release create` 只负责验证并冻结指定 run，不替代模型选择，
也不会自动从多 seed 矩阵挑最好结果；不要事后按 holdout 或 HeyBoss 回测收益挑 seed。

```bash
uv run facdigger release create \
  --run artifacts/e3/<run_id> \
  --output-root artifacts/releases

uv run facdigger release verify \
  --release artifacts/releases/<release_id>
```

上述 E3 run 路径可替换为 Finance Transformer 监督 cell 目录，其他 CLI 不变。跨 Windows/macOS
迁移时追加 `release create --dataset <本机训练快照目录>`，只改变定位，不更改原始 manifest，
也不放宽原数据 ID 或哈希检查。金融预训练的 encoder export 不是可发布的监督 run。
允许 dirty 只解除 Git 干净状态门禁：必须仍为 complete run，checkpoint/config/scaler/predictions
均保持原绑定。不能通过编辑原 manifest 或把 `git_clean` 写成 true 来使用该通路。

全历史工程回测先用 release 的冻结 scaler 建一次完整 target-free snapshot，再运行可恢复的年度
回放。配置中的 `acknowledge_non_oos: true` 是必需的显式确认：

```bash
uv run facdigger dataset build-inference \
  --config configs/datasets/eodhd_historical_liquid_inference.yaml \
  --release artifacts/releases/<release_id>

cp configs/inference/e3_historical_replay.example.yaml \
  configs/inference/e3_historical_replay.local.yaml
# 填写本机路径和日期；将确认后的 targets/identities 嵌入 delivery，security_ids 留空

uv run facdigger factor-history plan \
  --config configs/inference/e3_historical_replay.local.yaml
uv run facdigger factor-history run \
  --config configs/inference/e3_historical_replay.local.yaml
uv run facdigger factor-history verify \
  --export artifacts/factor_history/<history_id>
```

迁移后 `factor-history plan/run` 可追加本机 `--release/--dataset/--output-root`；`verify`
可追加 `--release/--dataset`。这些只重新定位，不改写已保存的 resolved config，不改变
release/snapshot 内容身份或分片键。复制到 HeyBoss 的仍只是子 FactorBatch，无需原 Windows 目录。

`run` 按自然年生成子 FactorBatch，已完成年份会在继续前重新验证并跳过。不要写 shell 循环逐日
调用 `signal`：那会把历史工程回测标成 `signal_inference` 生产语义，且没有父级完整性和恢复
边界。给 HeyBoss 的是上述 export 内各年度 `<delivery_id>/` 子目录，不是 export 根目录。

原始评价 split 的小范围隔离回放仍可用于 importer smoke：

```bash
uv run facdigger factor-batch from-predictions \
  --predictions artifacts/e3/<run_id>/predictions.parquet \
  --release artifacts/releases/<release_id> \
  --output-root artifacts/factor_batches \
  --delivery-config configs/inference/heyboss_delivery.local.yaml
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
  --delivery-config configs/inference/heyboss_delivery.local.yaml \
  --asof latest

uv run facdigger factor-batch verify \
  --bundle artifacts/factor_batches/<delivery_id>
```

升级前若 production store 仅有最后一天 universe，需使用新的 `data.store_root` 从历史 bronze
重新 bootstrap。金融模型要求热窗口内逐日真实股票池，不允许用今日成员回填。新代码会明确
拒绝旧的缺历史成员存储，不自动改写训练资产。
同时迁移生产配置：完整计算池最低数量从 `factor_batch.minimum_*` 移到
`inference.minimum_*`；`factor_batch.delivery` 内嵌确认后的 profile。交付小股票池按精确覆盖
检查，不受完整计算池 100 行的最低数量限制。旁路审计位于 `factor_batches_delivery_audits/`，
HeyBoss 不读取这些审计文件。

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

发布恢复窗口已补齐：如果目录完成后、ledger 记账前退出，FacDigger 在采集前核验唯一原交付，
恢复同一 delivery ID；源修订不能生成第二份同日交付。原快照、质量记录、release 血缘、目标
及有效期映射须全部匹配，歧义/损坏则阻断。截止后只补记截止前已存在的合法原交付，不补发、
不修改创建时间。HeyBoss 的独立截止、唯一接纳与冲突拒绝不变。隔离故障复验及部署边界见
[826 每日生产运行交接](826每日生产运行交接.md#fd-04-发布恢复窗口复验2026-09-21)。

每次交接记录 FacDigger commit、release ID、history ID/年份（若为历史回放）、delivery ID、
source kind、日期范围、Parquet hash、row/date/eligible counts、`strict_out_of_sample=false`、被批准
的模型/run/seed 选择依据和 HeyBoss commit。只复制完整
`<delivery_id>/` 目录；不要重新序列化 Parquet 或手改 manifest。

## 5. 真实结果的独立发布前置

现有 `artifacts2` 来自关键修复前的训练协议：它不能证明完整日 objective、正确 patch-mask
对齐以及当前 scaler/predictions 绑定。ModelRelease 会有意拒绝这类旧 run，FactorBatch 回放
入口也没有 legacy bypass。这不是联调功能缺失，而是避免把已知不可靠权重送进交易系统。

本次检查的 `artifacts826new` / `data826new` 已采用 Finance Transformer 新训练格式，
其中完整监督 run 记录 `git.dirty=true`，EODHD 来源还包含 provider-symbol fallback 身份。
当前可通过 `--dataset <本机快照>` 和 `--allow-dirty` 做联调发布；训练池非交付股票的 fallback
不再阻塞。是否能生成某个真实 FactorBatch 仍取决于目标日期的数据/窗口和交付清单的可靠
映射，不能仅凭已支持模型格式宣称可以全历史接通。不要选择同目录中残留的 running run，
也不得改写原证据、自动按 ticker 补 ISIN，或把 dirty 来源描述为完全可复现。

真实跨仓测试应从满足门禁的原始 run 创建 ModelRelease。在此之前可使用 FacDigger 小型
CPU 集成测试生成的 FactorBatch 验证 importer/NT 通路，不能把合成输入结果当成策略收益证据。

## 6. 联调完成定义

满足以下全部条件才可称为“流程跑通”：

- 两个仓库均固定具体 commit，测试全绿；
- FacDigger 对真实 bundle 的 `factor-batch verify` 通过；
- HeyBoss 用同一原始目录通过严格 semantic/artifact/schema 校验；
- 至少一个 evaluation bundle 完成 Catalog→Actor→TradeSignalEvent→风险→模拟成交→报告；若验收
  目标是全历史回测，FacDigger 父 manifest 声明的全部年度子 delivery 均已按序导入；
- 至少一个 signal bundle 完成价格前置检查、Catalog 导入、Actor bootstrap 和人工审批前 dry run；
- 重复导入/重启幂等，冲突输入失败关闭；
- paper 未开启 `allow_evaluation_predictions`，且未自动沿用旧信号；
- 日志能用 delivery ID、release ID、as-of 日期和 rebalance key 跨系统追溯。

这一定义只证明工程链路和语义一致，不证明因子有效，也不替代新一轮 walk-forward 统计门禁、
组合风险评审和 paper 观察期。


## 2026-09-16：两侧统一交易日历

本次联合实施已按用户确认统一为 `exchange_calendars==4.13.2` / XNYS。
唯一入口为 `src/facdigger/data/market_calendar.py`；原供应商目录的手写日历已删除。
模块懒加载数据依赖，返回标准库 date 与有时区 UTC datetime；MarketSession 包含实际开收盘，
支持提前收盘。regular_sessions 为闭区间，previous/next 严格跨日，shift(offset=0) 要求交易日。
既有 regular_session_frame 仅做 Polars 转换。没有 Provider 抽象、插件注册或共享运行时包。

来源标识复用外部交付既有 calendar_version，固定为 `exchange_calendars:4.13.2:XNYS`。
发布前验证安装版本与所有 asof_date；不能仅换标签而不校验日期。production_window 继续负责
纽约 19:00 首次尝试、30 分钟重试与下一交易日实际开盘截止；提前收盘不改变 19:00 策略。
HeyBoss 独立计算最近已收盘 D，并以 N 开盘前本地成功验收、N 常规时段执行为消费门禁。

两仓 `tests/fixtures/us_equities_sessions.json` 内容一致；覆盖 2001/2012 特殊休市、2021-12-31、
2025-01-09、2026 夏冬令时、Good Friday 和 11-27/12-24 半日市。HeyBoss 的
`scripts/check_calendar_consistency.py` 使用双方各自解释器比较 2000—2027 完整日期集合、UTC
开收盘与前后交易日。升级来源必须重跑此联合检查；普通单仓测试不依赖另一个仓库。


## 2026-09-16：826 联合验收结果

已按已完成的 finance_patch_transformer_pretrained-20260902T045825Z-72c1db3f 运行发布：
model_release_id=fbd630164624c71fe67c5b7c6637f5be08ef3179bdf93f9c3aa48d208c44d7ef；
delivery_id=02172c408d11f2032da4f08567b3d54659bd5ee2199fad9c0dad62b26ef5a87a。
来源为 evaluation_predictions，2023-02-01—2024-12-02，462 个日期、10 个目标、4,620 行。
原交付全部有效；局部缺分和全 false 使用独立测试样例验证，未篡改真实实验产物。
源运行有 dirty Git 状态，发布按工程联调显式 allow-dirty，未改成 clean、未重新训练或解锁 test。

训练快照 b7ca76a74dbe396c8e157eb7ecc826460931ed66e917d939ab56746cb70d2696 的 features、
market_features、inference_index 日期集合与新 XNYS 日历一致；sample_index 中仅有原协议
purge/embargo 缺口，没有非交易日。XOM 的隔离历史映射绑定 US30231G1022，依据原预测及
SEC 历史披露，未替换为当前 ISIN。其余目标与消费者显式身份匹配。

HeyBoss 以 historical 模式完整导入，重复导入 0 新增行。行情从现有 EODHD 缓存经原有 HeyBoss
解析/公司行动/质量管道进入隔离 Catalog（9,300 根双价格日线，0 质量错误）。最终 NT 回放
run_id=20260916T065922Z-d1035895，462 个工作流、970 笔成交；逐笔执行窗口和下一交易日
开盘价加既定滑点核对均无偏差。开盘使用日线 open 推导的 QuoteTick 与固定流动性假设，不代表
真实盘口或精确开盘成交。初始边界 2023-01-31 缺 D 留下一条 SKIP，不回退旧批次。

验收产物位于 /Users/young/Documents/HeyBoss/reports/facdigger-826-validation/，包含
acceptance.json、delivery.yaml、import-audit.db、backtest-accepted.db、catalog 和最终回测目录；
模型、数据库、Catalog、缓存和报告均不提交 Git。HeyBoss 交易入口仍只消费两个 FactorBatch
文件，不读取冻结模型或训练文件。具体命令与限制见 HeyBoss 的
/Users/young/Documents/HeyBoss/docs/facdigger-heyboss-joint-implementation-plan.md 第八节。

质量结果：FacDigger Ruff 与 lock 检查通过，Python 3.10/3.11/3.12 完整测试各 277 项通过；
两仓 2000—2027 的 10,227 个自然日/7,041 个交易日、开收盘和前后日一致。HeyBoss Ruff、
mypy strict、996 项测试通过（总覆盖率 91.05%）；前端 127 项测试和构建通过。
本次完成离线历史链路验收，未启动 IBKR 下单或 Telegram 对外通知，也未迁移真实运行库。
真实 paper 仍需要 signal_inference、固定生产 release 与开盘前本地成功接纳。
