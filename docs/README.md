# FacDiggerNN 文档中心

本页是仓库文档的统一入口。代码、严格配置和测试是可执行行为的最终依据；下列“现行文档”
用于解释当前实现。`历史归档/` 只保存设计演进和特定日期证据，不参与当前决策。

## 第一次使用

建议按这个顺序阅读：

1. [`README.md`](../README.md)：项目定位、当前能力、最短运行主线和已知限制；
2. [`开发文档.md`](开发文档.md)：架构、数据契约、模块、CLI、产物、扩展和排错；
3. [`实验设计文档.md`](实验设计文档.md)：研究问题、E0—E3 对照、切分、统计和结论边界；
4. [`RTX2070_Windows训练指南.md`](RTX2070_Windows训练指南.md)：目标 GPU 机器的安装、迁移和资源门禁。
5. [`HeyBoss因子联调交接.md`](HeyBoss因子联调交接.md)：交易项目应实现的 importer 契约和流程验收。

## 现行文档职责

| 文档 | 回答的问题 | 权威范围 |
|---|---|---|
| [`README.md`](../README.md) | 这是什么、当前能做什么、怎样开始 | 项目入口 |
| [`开发文档.md`](开发文档.md) | 代码怎样组织、数据怎样流动、怎样扩展 | 工程说明 |
| [`实验设计文档.md`](实验设计文档.md) | 比较什么、怎样防泄漏、何时可下结论 | 实验协议说明 |
| [`RTX2070_Windows训练指南.md`](RTX2070_Windows训练指南.md) | 怎样在 WSL2/RTX 2070 Super 上运行 | 平台操作 |
| [`HeyBoss因子联调交接.md`](HeyBoss因子联调交接.md) | HeyBoss 怎样校验、导入和跑通因子链路 | 跨项目交接 |
| [`项目关键问题与修复复盘.md`](项目关键问题与修复复盘.md) | 真实问题如何定位、权衡、修复和验证 | 持续维护的复盘档案 |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | 如何改代码、测试和评审 | 贡献流程 |
| [`SECURITY.md`](../SECURITY.md) | 如何处理 token、外部数据和研究完整性 | 安全规则 |
| [`AGENTS.md`](../AGENTS.md) | coding agent 每次任务必须遵守什么 | Agent 仓库规则 |

当说明与实现冲突时，不要静默兼容：先以 `src/`、`configs/` 和 `tests/` 查明当前行为，再同步
修正文档。README 不承载逐里程碑开发日记；关键 bug 和设计问题进入复盘，特定时点审计进入
历史归档。

## 当前状态摘要

- 工程链路已覆盖标准化、快照、E0—E3、统一评价、回放、信号和 M6 walk-forward；
- 跨项目生产侧已具备不可变 ModelRelease、冻结 scaler 的 target-free inference snapshot
  和单日 FactorBatch；FacDigger 每日 EODHD 修订、指定日期推理、30 分钟重试、截止门禁及
  保留策略由 Docker 常驻服务统一编排，不依赖宿主 `launchd`；HeyBoss 已有
  importer/NT Catalog/Actor 骨架，但消费者仍需按
  [联调交接](HeyBoss因子联调交接.md)同步现行 semantic delivery ID 与 manifest 字段；
- 监督阶段以完整日横截面排序相关性为目标；E1—E3 通过 CPU 整日组装、GPU
  physical microbatch 和两遍回放计算精确整日梯度，LightGBM 使用按完整日分组的
  LambdaRank；
- 当前 E1—E3 监督 checkpoint 是 schema v3，objective 是
  `cross_sectional_rank_correlation_surrogate_v2_full_date`；旧 v1 chunked artifacts 不能恢复或混用；
- M6 决策要求单侧 HAC 显著性、非重叠样本稳健性和 Holm 多重比较控制；
- final holdout 在冻结参数后重新建立截至 validation 末日的训练快照并重新训练；
- 全历史 EODHD bronze 曾在项目机器上完成重建和质量门禁，但真实数据不随 Git 分发；
- 真实退市收益、点时行业和点时流通市值仍缺失，因此当前 M6 是 engineering 模式。
- 新一轮 engineering validation 的 `research_id` 是
  `m6_eodhd_engineering_full_date_v2`，final holdout 仍锁定；RTX 2070 Super / 16 GB
  的 `batch_size=64`、FP16 配置尚待真实 CUDA 单 cell 与完整矩阵验收。

具体机器是否具备数据、来源证明、快照和 checkpoint，必须检查本地目录及 manifest，不能根据
文档中的历史完成记录推断。

## 配置地图

```text
configs/
├── base.yaml                         # 环境/Checkpoint 诊断，不是训练配置
├── data/
│   ├── eodhd_free.yaml               # 两股票 API smoke
│   ├── eodhd_all_world_pilot.yaml    # 当前 active 100 股票资源 pilot
│   ├── eodhd_historical_liquid.yaml  # 历史动态 top-1000 主数据路径
│   └── eodhd_daily_production.yaml   # daily bulk 修订（禁缓存）
├── datasets/
│   ├── us_equities_daily_v1.yaml     # provider-neutral 标准表范例
│   ├── eodhd_free_smoke.yaml         # 短窗口管线 smoke
│   ├── eodhd_all_world_pilot.yaml    # 100 股票工程 snapshot
│   └── eodhd_historical_liquid.yaml  # 历史动态主 snapshot
├── experiments/                      # E0—E3 smoke、pilot 和完整模型配置
├── production/
│   └── eodhd_daily.example.yaml      # Docker 生产模板；本地副本固定 release ID
└── research/
    └── m6_eodhd_engineering.yaml     # 当前 M6 主线；正式门禁未全部开启
```

## 历史归档（当前开发可忽略）

以下文件保留是为了追溯，不应继续被更新成混合的“半历史、半现行”说明：

- [`早期实施计划_当前可忽略.md`](历史归档/早期实施计划_当前可忽略.md)：从空仓库起步的原始实施计划；
- [`早期PatchTST原始设计_当前可忽略.md`](历史归档/早期PatchTST原始设计_当前可忽略.md)：早期模型设计与阶段性同步；
- [`数据质量审计快照_2026-07-24_当前可忽略.md`](历史归档/数据质量审计快照_2026-07-24_当前可忽略.md)：特定日期的修复前后审计证据。

文件名和顶部提示都明确标记“当前可忽略”。若历史文档与现行文档冲突，以现行代码、配置、
测试和上表所列现行文档为准。
