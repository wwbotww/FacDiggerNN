# FacDiggerNN coding guide

本文件用于帮助 coding agent 快速进入项目、把代码放对位置，并在持续开发中保持清晰、
克制和可靠。开始修改前先看 `git status`，阅读相关实现与测试，不覆盖用户已有改动。

## Project context

FacDiggerNN 是面向美股日频数据的 point-in-time 机器学习因子研究 CLI，不是交易系统。
项目使用 Python 3.10–3.12（推荐 3.11）、uv/Hatchling、Typer、Pydantic/YAML、
Polars/PyArrow，以及可选的 PyTorch、Transformers 和 LightGBM。

入口是 `src/facdigger/cli.py` 中的 `facdigger` 命令。总体数据流是：

```text
provider -> 标准 Parquet 契约 -> 不可变快照 -> E0–E3 训练
         -> 统一评价 -> checkpoint 回放/信号 -> walk-forward 研究
```

本项目最重要的设计取向是：可复现、可审计、防止未来信息泄漏。数据或协议不可信时应明确
失败，不要为了跑通流程而静默修补。

## Where code belongs

- `src/facdigger/data/`：标准数据契约、adapter、快照；供应商实现只放在
  `data/providers/<provider>/`。
- `features/`、`labels/`、`datasets/`：特征、标签、切分、索引、窗口和采样。
- `models/`：模型结构、基线、PatchTST 迁移与预训练能力。
- `training/`：E0–E3 配置、训练编排、checkpoint/resume；共享训练逻辑优先复用
  `training/common.py`。
- `evaluation/`：预测契约、中性化、指标、覆盖门禁、报告和比较。
- `inference/`：checkpoint 回放和不读取标签的信号生成。
- `research/`：walk-forward、统计、冻结和 holdout 编排；复用训练器，不复制训练逻辑。
- `configs/`：数据、快照、实验和研究协议实例。
- `tests/unit/`、`tests/integration/`：pytest 单元与小型端到端测试。

使用说明以 `README.md` 为准；架构和数据契约见 `docs/开发文档.md`；实验协议见
`docs/实验设计文档.md`。代码、严格配置和测试是当前可执行行为的最终依据。

## How to work in this repository

1. 先用 `rg` 查找现有实现、调用方和相似测试；沿用最近似的模式，避免重复实现。
2. 选择能够完整解决问题的最小改动。范围外问题可以报告，不要顺手重构。
3. 新抽象应来自真实的当前需求。优先扩展已有领域模块，谨慎新增通用 `utils/helpers`。
4. 保持函数职责和数据流清楚；命名表达领域含义，避免隐式状态、魔法修复和过度封装。
5. CLI 保持轻薄，只负责参数、调用和用户输出；业务规则进入对应领域模块。
6. Bug 修复应定位根因，并尽量增加修复前可失败的回归测试。重构默认保持外部行为不变。
7. 不通过吞异常、降低断言、关闭检查或放宽契约来解决失败。
8. 新依赖前先确认标准库和现有依赖无法胜任；不要升级无关依赖或无理由改写 lockfile。

## Project-specific boundaries

- EODHD API、缓存、预算和原始字段语义不得越出 `data/providers/eodhd/`；下游只读取标准表。
- 标准表通过 `data.contracts`，预测通过 `evaluation.contracts`。新字段或语义变化要考虑
  snapshot/manifest 版本和旧产物兼容性。
- 特征只使用 `asof_date` 当时可知的信息；scaler 只拟合 train；outer validation/test
  不参与训练或 checkpoint 选择。
- 缺失价格不前值填充，非法 OHLC 不“修正”为合法值；来源质量、哈希、覆盖率、
  test/holdout 等现有门禁不要绕过。
- E0–E3 复用相同快照、预测契约和 evaluator；最新信号路径不能读取 labels/target。
- CLI、配置字段、Parquet schema、manifest 和 checkpoint 布局都是需要谨慎维护的接口。
- 领域层使用明确异常并保留根因；CLI 可以转换为非零退出码。不要在输出、缓存或
  manifest 中泄露 token。

关键 Bug 或会长期影响架构/数据语义的设计问题，应在同一轮更新
`docs/项目关键问题与修复复盘.md`，记录真实证据、验证结果和仍存限制。

## Commands and validation

```bash
uv sync --extra model --extra data --extra eodhd --extra baseline --extra dev
uv run ruff check .
uv run pytest
uv run pytest path/to/test_file.py -q
uv run pytest path/to/test_file.py::test_name -q
uv run pytest --cov=facdigger --cov-report=term-missing
uv run facdigger doctor
```

先运行最相关的测试，再根据影响范围扩大：

- 叶子模块：对应 unit 测试和 Ruff。
- 契约、数据集、共享训练/评价逻辑：相关 integration 测试后运行完整测试。
- 模型/checkpoint：覆盖 pipeline、resume 和 replay；CUDA/网络验证要说明真实环境。
- EODHD：使用 fake transport 和临时目录。普通测试不访问 live API、不刷新真实缓存。
- 纯文档改动：检查链接、命令和 `git diff --check` 即可。

仓库当前没有单独配置 type checker、formatter、CI 或发布命令。只报告实际执行过的检查；
无法执行网络、GPU 或付费数据验证时，说明未覆盖范围。

## Data, secrets and generated files

`.env.local` 和 `EODHD_API_TOKEN` 不提交，也不写入 YAML、日志、缓存键或 manifest。
`data/`、`artifacts/`、`.venv/`、缓存、Parquet、checkpoint、模型权重、`build/` 和
`dist/` 是本机或生成内容，不直接编辑。数据采集、模型下载、正式训练和 holdout 解锁
可能消耗额度或大量算力，只在任务明确需要时执行。

## Finishing a task

完成前确认：

- 改动解决了实际需求，并位于正确模块；
- 相关测试已添加或更新，执行结果如实记录；
- 没有混入无关改动、生成物、秘密或不必要依赖；
- 用户可见行为、配置、数据语义或实验协议变化已同步对应文档；
- 已检查最终 diff。

最终回复简洁说明改了什么、执行了哪些验证、哪些验证未执行，以及仍存风险。
