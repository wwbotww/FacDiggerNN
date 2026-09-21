# HeyBoss 局部缺分持仓保护交接

当前状态：2026-09-20。两侧日历统一、HeyBoss 的 SKIP、持仓数量/预算保护、时间与 release
门禁、审批恢复代码已完成；826 release 和原 validation 预测的离线历史交付已验收。
§3–4 保留 2026-09-10 的根因分析与实施设计作为历史参考，§5 用于后续回归，均不表示需要
重新开发。实际实施与产物记录见本文末尾及[因子联调交接](HeyBoss因子联调交接.md)。
真实每日采集和 Docker 无标签推理已产出 2026-09-18 的十只批次，并通过只读 parser 校验，
见[826 生产运行交接](826每日生产运行交接.md)。真实接纳、缺分保护和连续五日仍待联合验证；
本轮未迁移或写入 HeyBoss 运行库、未启动交易。不要仅凭离线
历史验收启用无人值守调仓。

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

## 3. 2026-09-10 根因审计（历史，相关代码已修复）

以下描述当时 HeyBoss `src/trading_assistant/` 的实现，不是当前仍存在的问题清单：

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

## 4. 原实施方案（历史设计参考，不是待办清单）

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

## 5. 持续维护的回归验收矩阵

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

## 6. 已发布的 826 与剩余生产验收

`artifacts826new` 中 complete 的 `wf3/finance_pretrained` 监督 run
`finance_patch_transformer_pretrained-20260902T045825Z-72c1db3f` 已生成 release，并交付
462 天、4,620 行 `evaluation_predictions`。具体 ID 与目录见末尾验收记录；不要重复发布，
也不要改写原 manifest、checkpoint、训练 snapshot 或 dirty 来源状态。

该验收使用原 validation 预测，不是最新日期 `signal_inference`，也不是固定 release 的
target-free 全历史重新推理。下一步只推进尚未验证的生产环节：

1. 显式配置选定的生产 release ID，核对双方部署代码及依赖；已有隔离 release 可以验证加载，
   不能把“已存在”当成已部署到生产配置。
2. 确认生产日期的 targets、有效期身份和 signal/execution bars；历史 XOM 隔离映射不可直接
   套用当前日期。按 HeyBoss 迁移流程审阅、备份并显式迁移真实运行库，不直接复用隔离测试库。
3. 以实际 D 数据构建无标签 snapshot、运行完整可评分横截面，再投影交付，验证新
   `signal_inference` 批次；不把历史数据标成最新日期，不把旧预测文件当作每日模型输出。
4. 验证 fresh EODHD、Docker 时钟、文件传递与导入调度、开盘前本地成功接纳和 paper dry-run；
   用独立测试输入覆盖缺数、迟到、重启，不修改真实训练或交付产物。
5. 单独记录持续运行、告警恢复及人工批准的 paper 验收证据；这些尚未由离线回测证明。

工程跑通不代表该模型已通过样本外收益门禁；这里不重新训练、不解锁 holdout，也不依据这次
回测选模型。上述剩余步骤需在后续任务中执行，本次文档收尾不启动真实生产或交易。


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
