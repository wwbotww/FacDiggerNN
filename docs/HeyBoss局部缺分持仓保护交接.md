# HeyBoss 局部缺分持仓保护交接

日期：2026-09-10。本文交给 **HeyBoss 项目的 agent 会话实施**。
本轮仅修改 FacDigger，未修改 HeyBoss、真实标的配置、Catalog、交易数据库或部署环境。
以下 HeyBoss 行为是待实施要求，不是已验收能力。不要因为 FacDigger 可以发布局部缺数批次，
就直接启用无人值守调仓。

## 1. 目标和两项目边界

- FacDigger：数据局部缺失时，明确表示当天不可评分；其余股票继续真实推理。缺失异常则
  等待重试，截止后跳过当天，服务继续运行；不制造价格/分数，不回退旧因子。
- HeyBoss：只用当天有效分数构建组合；不可评分的已有持仓保持当前数量，不因缺分自动
  卖出。数据不足以构建组合、缺少必要估值或没有当天批次时，明确跳过本次因子调仓并告警。
- 两边仍保留身份、时间、来源、文件和交付完整性检查。未知身份不是“允许缺数”。
- 独立风控、人工平仓或已确认的证券生命周期处置不被缺分保护关闭；这些必须有独立依据，
  不能由“因子缺分”暗中触发。

## 2. FacDigger 已实现的接口

### 2.1 FactorBatch 不增加格式

仍只交付 `<delivery_id>/factors.parquet` 与 `manifest.json`。五列顺序、类型、哈希、
ModelRelease 绑定均不变；E3 和 `finance_patch_transformer` 共用该接口。

| 当天情况 | 行是否交付 | eligible / score | HeyBoss 应有语义 |
|---|---|---|---|
| 可评分，模型输出有效 | 是 | true / 有限 float64 | 参加当日排序；有效低分可以导致正常减仓/退出 |
| 活跃交付目标缺 D bar、窗口不足或不满足模型池条件 | 是 | false / null | 不排序、不新建该仓；已有仓位保持数量 |
| 目标行缺失、身份不唯一/过期、eligible 分数缺失或 NaN | 不发布合法批次 | 仍为错误 | 拒绝，不补零、不缩小覆盖分母 |
| 缺失异常、市场输入失效或超过截止 | D 不发布 | 无新批次 | 跳过 D 的调仓，不寻找 D−1 因子 |

`eligible=false` 目前只表达“不可评分”，**不包含强制退出指令，也不能确定具体原因**。
原因在 FacDigger 运维审计中，不增加第三个交付文件；因此 HeyBoss 对 false 行应保守地保持
持仓。若以后要实现退出流动性池后的主动清仓，需另行设计明确退出策略，不能猜测 false 的原因。

`status=complete` 是文件契约完整，不等于所有候选都有分数。例如 10 个交付目标、其中 1 个
不可评分时，仍应满足：

```json
{
  "candidate_rows": 10,
  "actual_rows": 10,
  "expected_eligible_rows": 9,
  "scored_eligible_rows": 9,
  "missing_eligible_rows": 0,
  "ratio": 1.0
}
```

不能将 ratio 改成 0.9，不能删除第 10 行，更不能把空分数改成有效的 0 分。
全部 false 的 `signal` 诊断批次仍可能存在；生产服务的数量/比例门禁禁止发布这种批次。
消费者不能依赖“通常生产者不会发”而省略自己的 SKIP 判断。

历史 `evaluation_predictions` 仍是 eligible-only、backtest-only 契约；消费者按已有明确
目标/日期补齐内部不可评分状态时，也适用持仓保护。不要为本任务修改历史外部文件格式，
或把历史未交付行当成“应该全部卖掉”。

### 2.2 生产运行规则和诊断

配置模板：[`configs/production/eodhd_daily.example.yaml`](../configs/production/eodhd_daily.example.yaml)。

```yaml
quality:
  max_computational_missing_fraction: 0.05
  max_delivery_unscorable_fraction: 0.20
  minimum_delivery_eligible_rows: 3
```

计算基准在采集前持久化，不随重试缩水；连续异常/缩水日不自动成为新的小基准。
计算池绝对下限仍由 `inference.minimum_candidate_rows/minimum_eligible_rows` 控制，默认 100。
Finance 另检查共享市场窗口和贡献成员覆盖。阈值是初始运维设置，应与实际交付池和 HeyBoss
Top N 一致，不是经过收益调优的参数。

小范围问题通过门禁后发布 D，质量状态是 `degraded`；严重缺失为 `insufficient`，每 30 分钟
fresh 修订，下一美股 regular session 开盘截止。source CURRENT 到 D 不会跳过未发布任务的
重新采集。已发布 D 不会因为后续修订再发冲突批次，也不复用旧日分数。

```bash
facdigger production status --config configs/production/eodhd_daily.local.yaml
facdigger production health --config configs/production/eodhd_daily.local.yaml
docker compose logs -f facdigger-production
```

- `status`：`latest.target_date/status/attempts/next_retry_at/delivery_id/error/quality`。
  quality 包括 `stage/status/computation/delivery/unscorable/market/violations`；没有新评估时
  保留上次报告，须结合运行状态阅读。
- `health`：`healthy` 只表示心跳存活；另有 `production.target_date/status/quality`。
  “服务健康”绝不授权消费旧因子。
- 日志：`event=production_readiness`，按日期及状态/错误类型/门禁变化去重，记录等待、降级、
  恢复、阻断和截止。当前告警渠道是 Docker 日志，没有短信/邮件/webhook。
- 消费者不读取 FacDigger SQLite、snapshot 或 checkpoint。日常交易判断仍基于验证后的
  FactorBatch、明确预期 D、固定 release 和自身组合条件。

## 3. HeyBoss 当前根因（只读代码审计）

以 HeyBoss 当前 `src/trading_assistant/` 下实现为基线，开始修改前应重新检查最新 diff：

1. `signals/factor.py::calculate_factor_weights` 在可用股票不足 Top N 时返回 `{}`。
2. `strategies/patchtst_factor.py::_publish_signal` 仍把这个空权重发送为 `TradeSignalEvent`；
   正常局部缺分时也只保留 eligible 权重，没有持仓保护字段。
3. `execution/gateway.py::_build_plan` 遍历正目标与现有持仓的并集，对缺少目标权重的持仓
   使用 0，计算的目标数量为 0。因而“没有足够分数”会变成卖单，而不只是“不发买单”。
4. importer 已接受 false/null，但当前 `FactorScoreData` 将 null 存成 0.0，真实缺分语义
   依赖 eligible；后续筛选/序列化必须确保它永不被当成有效零分。
5. Actor Catalog bootstrap 选择最新完整批次，没有用预期交易日和固定 release 双重约束；
   未发布 D 时仍有选到旧批次的风险。固定小时数过期也不能代替交易日判断。

日期身份映射已存在（`factor_identity_periods` / `factor_security_id_on`）；不要再建一套
ticker alias，也不要把本任务误当成重写 importer 或增加模型专属接口。

## 4. 建议分三个小提交实施

### H1：明确 REBALANCE / SKIP 与持仓保护意图

修改范围：`signals/factor.py`、`strategies/patchtst_factor.py`、`execution/events.py`、
策略严格配置和对应单元测试。遵守 HeyBoss 自身 AGENTS 的文件/接口确认流程。

- 在现有因子纯函数模块返回明确的决策结果，区分 REBALANCE 与 SKIP，附带可审计原因。
  新结果应替代含糊的旧返回值并迁移调用方，不保留两条相反的权威路径。
- 先收齐、校验 D 的完整目标集合，再划分 eligible 与不可评分；eligible 必须有有限分数。
  不足 Top N、交付不可评分比例超过消费者阈值，或有效数低于组合最低要求时 SKIP。
- SKIP 只写审计/告警，不注册可执行的空权重调仓。不要修改所有策略对空权重的全局含义：
  其他策略显式清仓仍应按原策略意图处理。
- REBALANCE 对 eligible 子集排序和分配目标预算；不可评分集合随不可变事件明确传给 Gateway。
  建议最小字段为 `preserve_positions: tuple[str, ...] = ()`，使用 canonical instrument ID，
  与 target_weights 的标的互斥、无重复且必须已知。
- 该字段表达“执行时保持当前数量”，不是固定旧因子、旧分数、旧权重或 Actor 缓存的股数。
  Actor 不查账户、不构造订单；无仓的不可评分标的不会因此新建仓。
- 由于保护仓位可能额外存在，暂时允许实际持仓名数超过 Top N：Top N 指有有效分数的正常
  组合，另加被保护仓位。若风险限制不允许额外持仓，则跳过该次因子调仓，不强制卖掉保护仓。

建议在现有数据结构可行的前提下将内部 score 保留为 nullable，并同步 serializer/schema/
round-trip 测试；若 NT 存储约束需要继续保留数值占位，必须提供单一转换入口及测试，证明
false 行不会进入任何排序/权重路径，不再把 0.0 当作真实测量。

### H2：Gateway 保持数量、扣除占用预算，贯穿审批与恢复

修改范围：`execution/gateway.py`、`storage/models.py`、仓储 DTO/事件转换、对应显式
数据库迁移、Gateway/审批/恢复测试。不直接改用户真实数据库。

1. Gateway 读取实时持仓，对 preserve 集合中的已有仓位设 `target_quantity=current_quantity`，
   因子调仓对其 delta 恒为 0；不通过旧权重近似，否则价格或权益变化仍会生成交易。
2. 这些仓位仍参与权益、gross、单标的风险和其他已有约束。可分配给正常候选的总预算不超过
   `max(0, min(策略请求总敞口, 风控上限) - 被保护持仓敞口)`，必要时等比例缩小正常目标权重。
   例如策略总预算 75%、被保护仓占 25%，正常候选最多分配 50%，不是再加 75%。
3. 没有可靠且符合现有时效规则的估值，或保护仓本身已超约束时，跳过本次因子调仓并告警；
   不伪造价格，不把保护仓当作零市值，不为了满足预算由因子路径卖掉它。独立风控照常处理。
4. 有有效分数但未进入 Top N 的持仓仍按正常策略退出，不能把所有旧仓都冻结。
5. `SignalWorkflowRecord` 及所有 to/from_event、审批请求、重启恢复保存 preserve 字段。
   审批时重新读取数量/估值并重建计划；审批前后其他来源成交不得让保护语义失效。
   注意已经排队或挂出的本策略旧卖单：应撤销/使其失效或明确拒绝继续调仓，不能让“新计划
   没有卖单”掩盖旧单随后成交；不误撤独立风控或人工订单。
6. 迁移只扩展当前需要的字段，不新建通用状态框架。旧因子待审批信号没有足够保护信息时
   不应直接按默认空集合执行；部署前明确使其失效并重新生成，其他策略按既有语义处理。

backtest 和 paper 必须复用同一个决策/Gateway 路径，不能只给线上加保护、回测继续误清仓。

### H3：新鲜度、审计和跨仓验收

修改范围：Actor、运行配置/装配（backtest/paper）、因子导入及策略/Gateway 集成测试、
`docs/factor-integration.md`。不自动修改真实 instrument 配置或切换 active strategy。

- 明确 expected as-of D 与固定 model release。实时/重启 bootstrap 只消费预期 D 的该 release；
  没有就 SKIP，不能选择“Catalog 中最新的一份”代替。历史回测按模拟交易时钟确定 D，
  不把当前机器日期写死；换 release 需显式配置。
- 尊重 `signal_available=after_regular_session_close` 与
  `earliest_execution=next_regular_session_open`。验证纽约 DST、周末、休市、19:00 首次
  生产和实际 Catalog 时间戳的关系；不能用固定 24 小时过期规则取代交易日就绪/执行边界。
- 每次决策审计记录 D、delivery/release ID、有效/不可评分数量、preserve 集合、跳过原因。
  同一 D 同一原因去重告警，恢复后记录新的状态；没有交易并不意味业务循环停止。
- artifact/semantic hash、完整五列、完整候选数量、有效期身份唯一性和 eligible 价格前置条件
  仍严格校验。FactorBatch 的质量降级不允许绕过 importer 的任何完整性检查。

## 5. 必须通过的验收矩阵

全部用临时数据库/Catalog、fake broker/时钟；以下不授权真实交易。

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

## 6. 826 实验 release 的后续联调

本阶段没有新建真实 release；先完成两侧保护，再返回 FacDigger 会话执行 release 与交付。
已检查的候选是 `artifacts826new` 中 complete 的 `wf3/finance_pretrained` 监督 run
`finance_patch_transformer_pretrained-20260902T045825Z-72c1db3f`，不是仍标记 running 的新目录。
发布前重新核验本机 run、checkpoint、scaler、数据集及状态，不按目录日期自动选模型。

1. 从实际文件核验并记录双方 Git commit，固定一版 release。使用本机 `--dataset` 重定位
   Windows 数据集路径；若来源仍为 dirty，显式 `--allow-dirty` 并如实记录，不改原 manifest。
2. 双方确认实际交付 targets、日期有效期、身份依据和 Catalog 中的 signal/execution bars。
   例如 2024 历史 XOM 不能被自动改成当前 ISIN，也不能绕过 HeyBoss 已有日期身份解析。
3. 先取已具备价格和模型上下文的少量历史日期，生成 backtest-only 批次做隔离回测；完整
   横截面计算后才投影小交付集合。不把 2024/2025 数据标成 2026 最新生产。
4. 用测试输入的副本再覆盖局部缺数/严重缺数场景，不修改原训练 snapshot 或实验结果。
5. 再单独验证 fresh EODHD、Docker 生产时钟、文件传递/导入调度及 paper dry-run。
   当前并未完成跨容器自动导入、paper 订单或真实数据长期运行验收。

工程跑通不代表该模型已通过样本外收益门禁；这里不重新训练、不解锁 holdout，也不依据这次
回测选模型。HeyBoss agent 完成后应交回修改清单、实际测试、迁移注意点和未完成项，再进行
826 release 的真实跨仓通路测试。
