# HeyBoss 因子联调交接

这是 FacDigger 对外接口和联合验收的唯一说明。HeyBoss 只接收一个最终 FactorBatch 目录，
不加载 ModelRelease、checkpoint、scaler、训练数据或 predictions；历史回放父目录也不是输入。
双方固定精确 commit，使用 FacDigger 真实产出的原始文件校验，不能把各自产生的 fixture
当成跨仓验收。本文不授权修改 HeyBoss 配置、运行库或启动交易。

历史改造与 826 validation 回测结果见[联调档案](历史归档/HeyBoss联调与验收记录_2026-09-16_当前可忽略.md)；
正式单日交付的末次证据见[826 运行交接](826每日生产运行交接.md)。
历史通过不代表当前消费者正在运行、已接纳新批次或完成了连续多日验收。

## 1. 同一格式，两种消费用途

| source.kind / universe_semantics | 生成路径 | 消费范围 |
|---|---|---|
| signal_inference / complete_candidate_cross_section | 固定 release 对单日无标签快照评分；保留所有应交付候选 | 满足预期 D、固定 release、时间和质量检查后，才可用于当日 paper 流程 |
| evaluation_predictions / eligible_scored_cross_section | release 绑定的原始评价预测，或固定模型无标签历史回放 | 仅隔离 backtest，只有 eligible 已评分行，禁止用于 paper |

`complete_candidate_cross_section` 指实际交付集合，不等于模型的全部计算池。
模型始终先在完整可评分横截面上运行，最后才筛选交付；Finance 的跨股网络不能先截成十只。

E1 `random_patchtst`、E2 `etth1_transferred_patchtst`、E3 `financial_pretrained_patchtst`
及 `finance_patch_transformer` 共用接口。
`model_type` 是来源元数据（格式 `[a-z][a-z0-9_]*`），不是消费者模型白名单；
scratch/pretrained 由不同 release 区分。消费者仍严格校验排序含义、horizon 和方向。

## 2. 文件与哈希契约

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

delivery ID 绑定完整交付语义，不等于 Parquet SHA-256。计算规则是：

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

最终目录内不能有额外文件，包括隐藏文件；只忽略交付根目录中尚未 rename 的
`.tmp-factor-batch-*` 兄弟目录。根 manifest 还包含 contract、delivery_id、created_at；
完整字段与校验以 [factor_batch.py](../src/facdigger/inference/factor_batch.py) 为准。

manifest 的严格字段如下：

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

生产约定主 horizon 为 5 sessions；不能把 1/20 日辅助 head 当成同一五日因子。
`created_at` 不参与内容 ID 不代表可以改写发布时间：已经发布的目录和原发布时间不可变，
生产恢复还会校验合法发布窗口。

## 3. 交付集合、身份与局部缺分

[Delivery 模板](../configs/inference/heyboss_delivery.example.yaml)分开配置：

- `targets`：instrument_id 及 active_from/active_to，决定当日应交付集合；
- `identities`：instrument_id、security_id、source_security_id、valid_from/valid_to、evidence。
  两端包含，source_security_id=null 表示源 ID 与交付 ID 相同；
- 单日与原预测导出用 `--delivery-config`；历史配置内嵌 delivery，不能同时指定 security_ids；
  每日服务必须设置 `factor_batch.delivery`。

实际交付身份必须唯一。无 ISIN 时仅接受双方确认、有效期和依据明确的可靠稳定映射，
不按 ticker 自动 fallback；目标映射过期不得静默缩小分母。训练中未交付股票缺 ISIN
不阻止 release，也不能据此缩减完整计算池。未提供 profile 的研究导出不声明已匹配 HeyBoss。

XOM 的历史与当前 ISIN 不应无条件 alias：826 原 validation 使用历史身份
US30231G1022；已验证的 2026-09 生产交付使用 US30233Q1085。
当前十只目标与身份核验日期见运行交接，后续仍要按交付日校验映射，不把有日期证据视为永久保证。

`eligible=false, score=null` 是**已交付且当天不可评分**，不同于缺行、未知身份或篡改。
完整性 ratio=1.0 可以与局部 false/null 同时成立；是否仍足够构建组合是另一层质量判断。
已保存的原 predictions 没有完整候选表，显式 active 目标缺失时不能猜成 ineligible；
需要动态资格和全日期语义时用无标签历史回放。

身份/投影审计保存在两文件目录外：单日/预测导出为相邻
`<output_root.name>_delivery_audits/<delivery_id>.json`，历史为父 resolved config/plan。
旁路记录不能替代合法 FactorBatch，也不能塞进交付目录。

## 4. 交易日与消费者行为

日历唯一来源为 `exchange_calendars==4.13.2` 的 XNYS，FacDigger 入口
[data/market_calendar.py](../src/facdigger/data/market_calendar.py) 返回有时区实际开收盘。
calendar_version 为 `exchange_calendars:4.13.2:XNYS`；发布前检查依赖版本和每个 asof_date，
不能只更改标签。提前收盘、特殊休市和 DST 均按交易日历处理，不用固定 24 小时过期。

D 收盘后生成，N 为下一 regular session。FacDigger 的窗口与重试见
[生产运维](每日生产运维.md)；HeyBoss 按自己的交易时钟求 expected D，
只在 N 开盘前本地成功验收后、N 常规时段执行。生产者按时发布不自动证明消费者按时接纳。

日历来源升级时双方重跑共同 fixtures 与 HeyBoss 的跨解释器一致性检查；
普通单仓测试不能隐式依赖另一仓存在。[历史档案](历史归档/HeyBoss联调与验收记录_2026-09-16_当前可忽略.md)
保留当次 2000–2027 日期、开收盘及前后 session 的联合证据。

HeyBoss 必须维持下列消费语义；这是验收要求，不代表本次核查了其运行状态：

| 条件 | 必须行为 |
|---|---|
| 当日合法有限 score | 排序、正常组合决策；有效低分未入选可以正常退出 |
| 当日 false/null，有现有仓位 | 保持**执行时实际股数**，不按旧权重重新买卖 |
| 当日 false/null，无仓位 | 不开新仓，不把数值占位当有效零分 |
| 缺 D、错误 release、全 false、有效数不足 Top N | SKIP 并告警，不发送可执行的空权重清仓，不回退 D−1 |
| 保护仓缺可靠估值、已超风险或预算无法满足 | 整次因子调仓 SKIP，不假设零市值或伪造报价 |
| 待审批/进程重启 | 保存保护意图，重读实时数量/估值；不能由旧挂单继续误卖 |

保护仓仍占用总敞口：总预算 75%、保护仓 25%，正常候选最多 50%。
保护只作用于此因子决策，不妨碍独立风控/人工操作，也不改变其他策略显式清仓语义。
backtest/paper 复用同一决策/Gateway 逻辑。

## 5. FacDigger 交付操作

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

每日部署、旧 store 迁移和唯一交付恢复统一按[生产运维](每日生产运维.md)执行。
HeyBoss 只监视最终 FactorBatch 子目录，不读取 FacDigger SQLite/source 或临时目录；
生产者按时发布与消费者开盘前验收是独立检查，消费者的唯一接纳和冲突拒绝不变。

每次交接记录 FacDigger commit、release ID、history ID/年份（若为历史回放）、delivery ID、
source kind、日期范围、Parquet hash、row/date/eligible counts、`strict_out_of_sample=false`、被批准
的模型/run/seed 选择依据和 HeyBoss commit。只复制完整
`<delivery_id>/` 目录；不要重新序列化 Parquet 或手改 manifest。

日常生产由唯一 Docker serve 完成，不在其旁边手动 signal/tick 制造第二份同日交付；
新 release 创建与生产切换是两个独立动作，详见生产运维。

## 6. 联合验收矩阵

先验证外部契约、语义/字节哈希、目录名/两文件、排序/主键/有限性/覆盖率、身份有效期；
再在临时 Catalog 和 fake broker/时钟中验证消费。双方每次记录精确代码版本与原始 delivery ID。

| 场景 | 必须断言 |
|---|---|
| 10 个目标中 1 个 false/null，已有该仓 | 导入 10 行；其股数不变，未生成因子卖单，9 个有效分数正常参与组合 |
| 同上，该股票无持仓 | 不新建缺分仓；正常候选继续调仓 |
| 只有 1 个有效分数、Top 3 | SKIP，无空权重可执行事件、无卖单；必须覆盖旧实现会清仓的回归 |
| 全 false 或 D 没有 FactorBatch | SKIP；已有持仓不动，不使用 D−1，后续 D+1 可恢复 |
| 有效低分但未入 Top N | 可以正常卖出，不能被误纳入 preserve |
| 75% 总预算、25% 保护仓 | 新正常目标总权重最多 50%，最终计划含保护仓的总敞口不超上限 |
| 缺少保护仓可靠估值或本身超风险 | 整次因子调仓跳过并告警，不忽略暴露或伪造报价 |
| 等待审批期间数量改变、重启恢复 | preserve 完整 round-trip；按新的实际数量保持，不按旧数量补买/卖出 |
| 本策略旧挂单/待执行信号 | 不能在保护决策后继续误卖；独立风控/人工指令仍有独立处理路径 |
| eligible=true 且 null/NaN，候选缺行或身份区间断档 | importer/决策拒绝；不动态缩小分母，不按 ticker 猜测 |
| Catalog 只有旧日/错误 release | 启动、重启和流式路径均不得产生因子调仓 |
| 周末、休市、DST 与下一开盘 | expected D 和执行时刻正确，不因自然日间隔错误清仓/回退 |
| 其他策略显式空仓事件 | 原有清仓语义不受影响；缺分保护不成为全局交易绕过 |
| 同批重复导入/重复到达、恢复正常批次 | 幂等且只产生一次正常调仓，审计包含恢复事件 |

先跑 HeyBoss 对应单元/集成测试，再按其贡献流程跑完整 pytest、Ruff、strict mypy。
最后使用 **FacDigger 实际生成的原始 FactorBatch** 做临时 Catalog → Actor → Gateway
dry-run，并逐值核对 score/eligible/身份；双方各自生成自洽 fixture 不能替代这一步。

额外覆盖 FacDigger 单日与年度原子发布/重复导出、相同分数在两仓逐值一致、不同模型共用导入、
缺失/歧义映射拒绝、全历史先完整计算再投影。旧 Huber、chunked/mask 错位和被重新序列化
的绑定 predictions 不应通过发布；接口通用化不是 legacy bypass。

## 7. 何时可以宣布通路完成

分别给出证据，不合并成笼统“已跑通”：

1. 契约验证：真实原始 batch 在两边校验通过，篡改、重复与错误语义正确拒绝；
2. 隔离 backtest：正确历史身份、价格、下一 session 执行，分数与成本假设可复验；
3. 当日生产：固定 release、fresh 数据、目标 D、完整计算/交付质量、唯一交付、真实发布时间早于截止；
4. 消费端：预期日和固定 release 接纳，开盘前本地验收，缺分/审批/重启保护及幂等；
5. 连续运行：多个交易日和失败恢复分别验收，不以一个成功 D 代替。

固定模型历史回放承认非 OOS，不用它证明收益或 paper 就绪。
学校新模型与 826 当前部署也不能互相替代验收。若需修改 HeyBoss，由其仓库 agent 按自身规则执行。
