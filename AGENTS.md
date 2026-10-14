# FacDiggerNN agent guide

FacDiggerNN 是面向美股日频数据的 point-in-time 因子研究 CLI，输出因子供交易系统消费。
本文件保留任务入口和容易误改的项目边界；流程、命令和详细协议按需阅读下列权威文档。

## 开始任务

- 先检查 `git status --short --branch` 和相关 worktree，保留已有修改与成果；新任务按
  [分支规则](CONTRIBUTING.md#分支与发布)开展，其他任务占用的 checkout 使用独立 worktree 隔离。
- 按问题定位实现、调用方和测试；不为每次小改动通读架构、历史归档和实验日志。

| 任务 | 按需阅读 |
|---|---|
| 安装、实现、测试、提交或发布 | [贡献指南](CONTRIBUTING.md) |
| 数据契约、模块边界、CLI 扩展 | [开发文档](docs/开发文档.md) |
| 训练、选择、评价或研究解释 | [共同协议](docs/实验设计文档.md)；具体实验从[研究概览](docs/因子效果诊断与实验改进方案.md)定位冻结设计 |
| 训练恢复或学校部署 | [训练运行与恢复](docs/训练运行与恢复.md)、[ICF 操作说明](configs/deployment/icf/README.md) |
| 推理、FactorBatch 或每日生产 | [联调契约](docs/HeyBoss因子联调交接.md)、[生产运维](docs/每日生产运维.md) |
| 文档、结果与复盘 | [文档职责和维护规则](docs/README.md#后续维护规则) |
| 凭据、外部文件、安全缺陷 | [安全规则](SECURITY.md) |

## 不可绕过的边界

- 特征只使用 `asof_date` 当时可知的信息。scaler、SSL、probe 和监督选择各自遵守冻结
  协议的允许区间，outer valid/test 不参与拟合或 checkpoint 选择；各 fold 按协议分别拟合，
  不把晚期 scaler/encoder 用于早期 fold。读取结果遵守任务与计划允许的 F/S/V/holdout 范围。
- 正式对照复用计划约定的快照、数据可用边界、评价支持集、预测契约和 evaluator；
  旧协议、新协议、原生 S-best 与诊断重评分的证据分开。工程通过不等于因子有效。
- 缺失价格不前值填充，非法 OHLC 不修正为合法值；不放宽来源质量、哈希、覆盖率或
  test/holdout 门禁来跑通流程。领域层保留明确异常及根因，CLI 转换为非零退出码。
- 训练快照、原始实验产物、冻结配置和 manifest 不原地改写。迁移只显式重定位路径，
  不重算哈希掩盖差异；也不切换运行中实验的 checkout/venv 或生产 release/CURRENT。
- 最新信号路径不读 labels、target 或 test membership，不重新拟合 scaler。交付身份仅在
  实际交付集合校验，不因未交付训练股票缺 ISIN 拒绝 release；评分仍使用完整计算横截面。
  过期身份映射不得静默缩小覆盖率分母。
- 联调如需 `--allow-dirty`，必须真实记录 dirty 状态；不能伪造 `git_clean=true` 或绕过
  其他产物验证。数据、权重和秘密不进入 Git；凭据与不可信 checkpoint 按安全规则处理。

## 定位实现

包入口是 `src/facdigger/cli.py` 的 Typer app（`facdigger.cli:app`）。业务逻辑在领域模块：

- `src/facdigger/data/` 管标准表、契约、provenance 和快照；EODHD API、预算、缓存及字段
  映射只在 `data/providers/eodhd/`，下游消费通用标准表和来源证明。
- `features/`、`labels/`、`datasets/` 管样本；`models/`、`training/` 管模型及恢复；
  `evaluation/` 管统一评价；`research/` 复用训练器编排；`inference/`、`production/` 管交付。
  这些目录均位于 `src/facdigger/`；共享逻辑先查 `training/common.py` 及已有领域实现。
- `configs/` 是配置实例，`scripts/icf/` 是学校作业包装；学校路径与调度限制留在部署层。
  通用训练可靠性改动保持本地和其他服务器兼容。测试在 `tests/unit/`、`tests/integration/`。

代码、严格配置和测试说明当前可执行行为；与协议或文档冲突时查明原因，不以任一方为由
静默改科学语义。修复、抽象和兼容策略见[变更原则](CONTRIBUTING.md#变更原则)。

## 验证与交付

- 按[风险矩阵](CONTRIBUTING.md#本地验证)选择检查，覆盖本次完整变更范围；纯文档不跑训练
  测试。使用已验证的开发环境，不为阅读文档安装全部依赖或修改固定实验 venv。
- 授权范围内的实现、临时 fixture 测试和文档检查直接推进。正式采集、模型下载、长训练、
  holdout 解封与生产变更须在任务明确授权范围内；已有授权不逐步重复确认。
- 完成前检查 diff，报告改动、实际验证和未覆盖范围；不把静态检查写成运行验收。
  按[提交与评审](CONTRIBUTING.md#提交与评审)核对并说明文档影响，遵守文档中心的唯一职责划分。
