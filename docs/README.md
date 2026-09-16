# FacDiggerNN 文档中心

本页是仓库文档的统一入口。代码、严格配置和测试是可执行行为的最终依据；下列“现行文档”
用于解释当前实现。`历史归档/` 只保存设计演进和特定日期证据，不参与当前决策。

## 第一次使用

建议按这个顺序阅读：

1. [`README.md`](../README.md)：项目定位、当前能力、最短运行主线和已知限制；
2. [`开发文档.md`](开发文档.md)：架构、数据契约、模块、CLI、产物、扩展和排错；
3. [`实验设计文档.md`](实验设计文档.md)：研究问题、E0—E3 对照、切分、统计和结论边界；
4. [`Transformer因子质量优化设计.md`](Transformer因子质量优化设计.md)：当前金融原生
   Transformer 精简实验的结构、训练、资源预算和验收方案；
5. [`RTX2070_Windows训练指南.md`](RTX2070_Windows训练指南.md)：目标 GPU 机器的安装、迁移和资源门禁。
6. [`HeyBoss因子联调交接.md`](HeyBoss因子联调交接.md)：现行 importer 契约、离线验收及生产边界。
   缺分保护设计、回归要求及剩余生产验收见[局部缺分持仓保护交接](HeyBoss局部缺分持仓保护交接.md)。

## 现行文档职责

| 文档 | 回答的问题 | 权威范围 |
|---|---|---|
| [`README.md`](../README.md) | 这是什么、当前能做什么、怎样开始 | 项目入口 |
| [`开发文档.md`](开发文档.md) | 代码怎样组织、数据怎样流动、怎样扩展 | 工程说明 |
| [`实验设计文档.md`](实验设计文档.md) | 比较什么、怎样防泄漏、何时可下结论 | 实验协议说明 |
| [`Transformer因子质量优化设计.md`](Transformer因子质量优化设计.md) | 当前怎样提高 Transformer 单模型因子质量 | 已实现协议与待运行实验 |
| [`RTX2070_Windows训练指南.md`](RTX2070_Windows训练指南.md) | 怎样在 WSL2/RTX 2070 Super 上运行 | 平台操作 |
| [`HeyBoss因子联调交接.md`](HeyBoss因子联调交接.md) | HeyBoss 怎样校验、导入和跑通因子链路 | 跨项目交接 |
| [`HeyBoss局部缺分持仓保护交接.md`](HeyBoss局部缺分持仓保护交接.md) | 局部缺分怎样不误清仓，哪些验收尚未完成 | 已实施设计、回归与生产验收边界 |
| [`项目关键问题与修复复盘.md`](项目关键问题与修复复盘.md) | 真实问题如何定位、权衡、修复和验证 | 持续维护的复盘档案 |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | 如何改代码、测试和评审 | 贡献流程 |
| [`SECURITY.md`](../SECURITY.md) | 如何处理 token、外部数据和研究完整性 | 安全规则 |
| [`AGENTS.md`](../AGENTS.md) | coding agent 每次任务必须遵守什么 | Agent 仓库规则 |

当说明与实现冲突时，不要静默兼容：先以 `src/`、`configs/` 和 `tests/` 查明当前行为，再同步
修正文档。README 不承载逐里程碑开发日记；关键 bug 和设计问题进入复盘，特定时点审计进入
历史归档。

## 当前状态摘要

- 工程链路已覆盖标准化、快照、E0—E3、金融原生 Transformer、统一评价、训练快照回放、
  信号和 walk-forward；
- 跨项目生产侧已具备不可变 ModelRelease、冻结 scaler 的 target-free inference snapshot
  和单日 FactorBatch；固定 release 的 target-free 全历史回放可按年生成 backtest-only
  FactorBatch 并断点续跑。FacDigger 每日 EODHD 修订、指定日期推理、30 分钟重试、截止门禁及
  保留策略由 Docker 常驻服务统一编排，不依赖宿主 `launchd`。E1—E3 与金融原生 Transformer
  共用 release/runtime/FactorBatch；HeyBoss 导入器已按排序语义解耦模型名称，实际交易联调
  仍须按[联调交接](HeyBoss因子联调交接.md)核对身份、价格、日期与消费模式；
- 交付 profile 将目标集合与身份有效期分开；完整计算池不因交付子集/缺 ISIN 被裁剪。
  release/predict 与历史 plan/run/verify 支持本机路径重定位，联调可显式允许 dirty 来源而不
  放宽产物绑定；生产配置的计算池最低数量位于 `inference`，必须提供 `factor_batch.delivery`；
- 每日生产允许小范围不可评分行，固定采集前的覆盖基准，分别检查计算池、交付池及 Finance
  市场输入；异常缺失重新 fresh 采集，到截止跳过 D 而不停服务。质量报告与心跳分离，未放宽
  五列/身份/有限分数门禁。HeyBoss 的 SKIP/保持数量/预算保护及两侧 XNYS 日历统一代码已完成，
  826 release 与原 validation 预测交付已通过离线历史验收；真实每日采集 → 无标签推理 →
  HeyBoss 接纳与持续 paper 仍待验证，真实运行库未迁移，不能宣称无人值守交易链路已完成；
- 监督阶段以完整日横截面排序相关性为目标；E1—E3 通过 CPU 整日组装、GPU
  physical microbatch 和两遍回放计算精确整日梯度，LightGBM 使用按完整日分组的
  LambdaRank；
- 当前 E1—E3 监督 checkpoint 是 schema v3，objective 是
  `cross_sectional_rank_correlation_surrogate_v2_full_date`；旧 v1 chunked artifacts 不能恢复或混用；
- 金融原生 Transformer checkpoint 是 schema v4，主矩阵固定 3 次 Train-only 预训练和
  6 个 scratch/pretrained 配对监督 cell；RTX 2070S 的 100-update CUDA/FP16、显存、RAM 和
  14 天资源门禁尚待运行，runner 会强制绑定报告的配置哈希和最大 fold dataset ID；
- M6 决策要求单侧 HAC 显著性、非重叠样本稳健性和 Holm 多重比较控制；
- final holdout 在冻结参数后重新建立截至 validation 末日的训练快照并重新训练；
- 全历史 EODHD bronze 曾在项目机器上完成重建和质量门禁，但真实数据不随 Git 分发；
- 真实退市收益、点时行业和点时流通市值仍缺失，因此当前 M6 是 engineering 模式。
- 新一轮 engineering validation 的 `research_id` 是
  `m6_eodhd_engineering_full_date_v2`，final holdout 仍锁定；RTX 2070 Super / 16 GB
  的旧配置尚待真实 CUDA 单 cell 与完整矩阵验收；当前精简主线使用独立配置，不再默认运行
  36-cell M6。

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
│   ├── eodhd_historical_liquid.yaml  # 历史动态旧主 snapshot
│   └── eodhd_historical_liquid_transformer.yaml # 14+6 路新主 snapshot
├── experiments/                      # E0—E3 与 finance Transformer 配置
├── inference/
│   ├── e3_historical_replay.example.yaml  # 固定 release 的 backtest-only 全历史回放
│   └── heyboss_delivery.example.yaml     # 目标范围与有明确有效期/依据的身份映射
├── production/
│   └── eodhd_daily.example.yaml      # Docker 生产模板；本地副本固定 release ID
└── research/
    ├── finance_transformer_streamlined.yaml # 当前 9 阶段精简主线
    └── m6_eodhd_engineering.yaml     # 旧完整矩阵；保留但不默认运行
```

## 历史归档（当前开发可忽略）

以下文件保留是为了追溯，不应继续被更新成混合的“半历史、半现行”说明：

- [`早期实施计划_当前可忽略.md`](历史归档/早期实施计划_当前可忽略.md)：从空仓库起步的原始实施计划；
- [`早期PatchTST原始设计_当前可忽略.md`](历史归档/早期PatchTST原始设计_当前可忽略.md)：早期模型设计与阶段性同步；
- [`数据质量审计快照_2026-07-24_当前可忽略.md`](历史归档/数据质量审计快照_2026-07-24_当前可忽略.md)：特定日期的修复前后审计证据。

文件名和顶部提示都明确标记“当前可忽略”。若历史文档与现行文档冲突，以现行代码、配置、
测试和上表所列现行文档为准。
