# FacDiggerNN coding guide

本文件只保存 coding agent 每次任务都需要的仓库规则。开发流程和 CI 见
`CONTRIBUTING.md`，安全与秘密处理见 `SECURITY.md`；文档职责和历史归档见
`docs/README.md`，使用、架构和实验协议分别见 `README.md`、`docs/开发文档.md` 和
`docs/实验设计文档.md`。

## Project context

FacDiggerNN 是面向美股日频数据的 point-in-time 机器学习因子研究 CLI，不是交易系统。

包入口由 `pyproject.toml` 中的 `facdigger = "facdigger.cli:app"` 定义，Typer app 位于
`src/facdigger/cli.py`。总体数据流是：

```text
provider -> 标准 Parquet + provenance 契约 -> 内容寻址快照 -> E0–E3 训练
         -> 统一评价 -> checkpoint 回放/信号 -> walk-forward 研究
```

本项目最重要的设计取向是：可复现、可审计、防止未来信息泄漏。数据或协议不可信时应明确
失败，不要为了跑通流程而静默修补。

## Where code belongs

- `src/facdigger/data/`：标准数据契约、adapter、快照；供应商实现只放在
  `data/providers/<provider>/`。
- `features/`、`labels/`、`datasets/`：特征、标签、切分、索引、窗口和采样。
- `models/`：模型结构、基线、PatchTST 迁移与预训练能力。
- `training/`：E0–E3 配置、训练编排与 checkpoint；E1–E3 支持 resume。共享逻辑优先
  复用 `training/common.py`。
- `evaluation/`：预测契约、中性化、指标、覆盖门禁、报告和比较。
- `inference/`：checkpoint 回放和不读取标签的信号生成。
- `research/`：walk-forward、统计、冻结和 holdout 编排；复用训练器，不复制训练逻辑。
- `configs/`：数据、快照、实验和研究协议实例。
- `tests/unit/`、`tests/integration/`：pytest 单元与小型端到端测试。

代码、严格配置和测试是当前可执行行为的最终依据；文档与实现冲突时不要猜测或静默兼容，
应查明原因并同步修正。

## How to work in this repository

1. 修改前检查 `git status`；保留用户已有改动，不覆盖或混入无关文件。
2. 先用 `rg` 查找实现、调用方和相似测试；沿用最近似的模式。
3. 选择能够完整解决问题的最小改动。范围外问题可以报告，不要顺手重构。
4. 新抽象应来自当前需求；优先扩展领域模块，谨慎新增通用 `utils/helpers`。
5. CLI 只负责参数、调用和用户输出；业务规则进入对应领域模块。
6. Bug 修复应定位根因并增加修复前可失败的回归测试；重构默认保持外部行为。
7. 不通过吞异常、降低断言、跳过检查或放宽契约来解决失败。
8. 新依赖前先确认现有能力不足；不要升级无关依赖或无理由改写 lockfile。

## Project-specific boundaries

- EODHD API、缓存、预算、原始响应和字段映射只放在 `data/providers/eodhd/`。provider
  必须生成 `data.provenance` 定义的通用标准化证明；下游只消费标准表和该通用契约。
- 标准表通过 `data.contracts`，来源证明通过 `data.provenance`，预测通过
  `evaluation.contracts`。契约变化必须考虑 snapshot/manifest 版本和旧产物兼容性。
- 特征只使用 `asof_date` 当时可知的信息；scaler 只拟合 train；outer validation/test
  不参与训练或 checkpoint 选择。
- 缺失价格不前值填充，非法 OHLC 不“修正”为合法值；来源质量、哈希、覆盖率、
  test/holdout 等现有门禁不要绕过。
- 正式对照中 E0–E3 必须复用相同快照、预测契约和 evaluator；最新信号路径不能读取
  labels、target 或 test membership。
- CLI、配置字段、Parquet schema、manifest 和 checkpoint 布局都是需要谨慎维护的接口。
- 领域层使用明确异常并保留根因；CLI 可以转换为非零退出码。不要在输出、缓存或
  manifest 中泄露 token。

关键修复的复盘触发条件和贡献要求见 `CONTRIBUTING.md`。

## Commands and validation

```bash
uv sync --frozen --all-extras
uv lock --check
uv run ruff check .
uv run pytest
uv run pytest path/to/test_file.py -q
uv run pytest path/to/test_file.py::test_name -q
uv run pytest --cov=facdigger --cov-report=term-missing
uv run facdigger doctor
```

先运行最相关的测试，再按 `CONTRIBUTING.md` 的风险矩阵扩大。CI 在 Python 3.10–3.12
运行锁定依赖、Ruff 和完整 pytest。EODHD 测试必须使用 fake transport 和临时目录；
普通测试不得访问 live API、刷新真实缓存或下载模型。只报告实际执行过的检查，并说明未覆盖
的网络、GPU、付费数据或正式研究环境。

## Data, secrets and generated files

`.env.local` 和 `EODHD_API_TOKEN` 不提交，也不写入 YAML、日志、缓存键或 manifest。
`data/`、`artifacts/`、`.venv/`、缓存、Parquet、checkpoint、模型权重、`build/` 和
`dist/` 是本机或生成内容，不直接编辑。内容寻址快照生成后按协议不得修改。数据采集、
模型下载、正式训练和 holdout 解锁只在任务明确要求时执行。

## Finishing a task

完成前检查最终 diff，确认改动位于正确模块、测试和文档已同步、没有无关改动、生成物、
秘密或不必要依赖。最终回复简洁说明改了什么、实际验证、未执行验证和仍存风险。
