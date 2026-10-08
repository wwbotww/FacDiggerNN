# FacDiggerNN 文档中心

这里按读者任务导航，不重复维护运行状态。代码、严格配置和测试是可执行行为的最终依据；
发现冲突时应查明并同步修正，不能以旧计划放宽现行门禁。

## 阅读顺序与职责

| 文档 | 唯一职责 |
|---|---|
| [项目 README](../README.md) | 定位、安装入口、能力和结论边界 |
| [开发文档](开发文档.md) | 模块职责、内部契约、数据流、扩展位置 |
| [实验设计文档](实验设计文档.md) | 当前 Finance Transformer 配对协议、质量与资源验收 |
| [因子效果诊断与实验改进方案](因子效果诊断与实验改进方案.md) | 历史证据、完整 F/梯度与三 seed、dropout 结果（6.16、6.19、6.22）；扩展计划（6.24）、完整F/S入口（6.25）与学校执行记录（6.26） |
| [第一批实验正确性修复实施方案](第一批实验正确性修复实施方案.md) | 修改范围、兼容边界与有限验收结果；代码、本地回归、小型 CUDA 与真实 wf3 更新准入通过 |
| [训练运行与恢复](训练运行与恢复.md) | 跨平台数据准备、迁移、命令、进度和恢复 |
| [Windows RTX 指南](RTX2070_Windows训练指南.md) | WSL2、驱动和 RTX 2070S 的平台差异 |
| [ICF 操作说明](../configs/deployment/icf/README.md) | 学校 Slurm、存储、凭据和作业包装 |
| [HeyBoss 因子联调交接](HeyBoss因子联调交接.md) | 唯一外部 FactorBatch 契约、缺分语义、跨仓验收 |
| [每日生产运维](每日生产运维.md) | Docker、发布窗口、质量门禁、FD-04 和受审计恢复 |
| [826 运行交接](826每日生产运行交接.md) | 固定部署资产、十只目标、末次有日期的验收结果 |
| [关键问题与修复复盘](项目关键问题与修复复盘.md) | 按主题检索真实案例、设计取舍和面试素材 |
| [贡献指南](../CONTRIBUTING.md) / [安全规则](../SECURITY.md) / [Agent 规则](../AGENTS.md) | 开发流程、秘密处理与每次任务约束 |

新使用者先读 README → 开发文档 → 实验设计，再按训练或交付分支继续。
只部署已训练模型的读者可直接读 HeyBoss 交接 → 每日生产运维 → 对应 release 的运行记录。

## 配置入口

| 目的 | 配置 |
|---|---|
| 小型 API smoke / 活动股票 pilot | [free](../configs/data/eodhd_free.yaml) / [pilot](../configs/data/eodhd_all_world_pilot.yaml) |
| 历史动态流动性数据 | [采集](../configs/data/eodhd_historical_liquid.yaml) |
| 当前 Finance 数据与 9 阶段矩阵 | [dataset](../configs/datasets/eodhd_historical_liquid_transformer.yaml) / [research](../configs/research/finance_transformer_streamlined.yaml) |
| 完整F/S参照实验，独立于正式矩阵 | [60个组合](../configs/research/finance_complete_reference.yaml) / [通用运行配置](../configs/runtime/finance_complete.example.yaml) |
| 通用训练控制 | [runtime](../configs/runtime/reliable.example.yaml) |
| 旧 E0–E3/M6，可选而非默认 | [旧 dataset](../configs/datasets/eodhd_historical_liquid.yaml) / [M6](../configs/research/m6_eodhd_engineering.yaml) |
| 无标签推理和交付身份 | [inference dataset](../configs/datasets/eodhd_historical_liquid_inference.yaml) / [delivery](../configs/inference/heyboss_delivery.example.yaml) |
| 固定模型历史回放 | [historical replay](../configs/inference/e3_historical_replay.example.yaml)，文件名沿用 E3，接口不限 E3 |
| 每日服务 | [production](../configs/production/eodhd_daily.example.yaml) |

模板不是已经部署的配置；`*.local.yaml`、数据、结果和秘密不进入 Git。

## 历史归档：当前开发可忽略

这些材料保留完整方案或时点证据，不再维护成第二份操作手册。文件名与顶部均标记历史用途。

| 归档 | 保留原因 |
|---|---|
| [早期实施计划](历史归档/早期实施计划_当前可忽略.md) | 原始里程碑与取舍 |
| [早期 PatchTST 原始设计](历史归档/早期PatchTST原始设计_当前可忽略.md) | 最初实现参考 |
| [2026-07-24 数据质量审计](历史归档/数据质量审计快照_2026-07-24_当前可忽略.md) | 修复前后真实证据 |
| [E0–E3 与 M6 完整实验协议](历史归档/E0-E3与M6实验协议_2026-09-28_当前可忽略.md) | 旧矩阵、统计决策、final refit/holdout 的完整说明 |
| [Transformer 优化设计](历史归档/Transformer因子质量优化设计_2026-09-28_当前可忽略.md) | 原始优化理由、暂缓消融和实施过程 |
| [训练可靠性改造方案](历史归档/训练可靠性改造方案_2026-09-25_当前可忽略.md) | 改造前根因、设计与 CPU 故障验收 |
| [ICF 环境调查与验收](历史归档/ICF环境调查与验收记录_2026-09-28_当前可忽略.md) | 节点实测、作业编号、数据准备及基准进度 |
| [2026-09-30 ICF 长训练恢复验收](历史归档/ICF长训练恢复验收记录_2026-09-30_当前可忽略.md) | 基准结果、固定版本恢复、requeue、突杀及长期验证边界 |
| [HeyBoss 联调与验收](历史归档/HeyBoss联调与验收记录_2026-09-16_当前可忽略.md) | 旧契约差异、缺分改造与历史联合证据 |
| [826 回补阻断修复方案](历史归档/826回补阻断修复方案_2026-09-23_当前可忽略.md) | 专项根因与验收矩阵 |
| [826 部署与验收记录](历史归档/826部署与验收记录_2026-09-24_当前可忽略.md) | 首次交付、FD-04、后续修复及完整时序 |

复盘案例不是待办列表；已解决问题的历史描述不应覆盖当前代码行为。
归档中的外部绝对路径、作业号和产物哈希只定位原环境证据，不保证新 clone 存在这些文件。

## 后续维护规则

1. 改接口更新对应职责文档，不向所有 README 复制同一说明。
2. 配置是参数默认值的依据；文档解释动机、使用方法与边界。参数变化同步实验协议。
3. “已实现”“已在某环境验证”“研究有效”“已部署”分开写；验证必须附日期与范围。
4. 持续运维步骤写运维指南，某次发布和失败记录写有日期的交接/复盘。
5. 关键修复在[复盘索引](项目关键问题与修复复盘.md)登记并更新相应主题文件；
   保存根因、验证、残余限制，不仅记录通过数字。
6. 历史文件只补充明确勘误或导航，不把旧方案改写成今天已执行的事实。
7. 移动/合并文档时检查路径、章节链接及命令；不删除不可替代的审计证据或生成资产。
