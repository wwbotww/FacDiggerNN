# Contributing to FacDiggerNN

本文面向人类贡献者，说明开发环境、变更流程、CI 和提交前检查。项目使用和研究协议分别以
`README.md`、`docs/开发文档.md` 和 `docs/实验设计文档.md` 为准；coding agent 的持久指令
位于 `AGENTS.md`；安全与密钥要求见 `SECURITY.md`；文档职责与历史归档见
`docs/README.md`。

## 开发环境

项目支持 Python 3.10–3.12，推荐 3.11。使用仓库根目录的锁文件安装全部开发能力：

```bash
uv sync --frozen --all-extras
uv run facdigger doctor
```

不要把 `.env.local`、数据、缓存、模型权重或训练产物加入版本控制。

## 变更原则

1. 修改前检查 `git status`，保留现有未提交改动，不混入无关重构。
2. 先定位现有实现、调用方和相似测试，再做能完整解决问题的最小变更。
3. Bug 修复应覆盖根因，并增加修复前能够失败的回归测试。
4. 配置、CLI、Parquet schema、manifest、checkpoint 或研究协议变化，应同步对应文档并
   说明兼容策略。
5. 影响 point-in-time、数据契约、snapshot、checkpoint、holdout 或长期架构的关键修复，
   需要在 `docs/项目关键问题与修复复盘.md` 记录证据、验证和仍存限制。

正式采集、模型下载、长时间训练和 final holdout 解封会消耗额度、算力或研究自由度，除非
任务明确要求，否则不要执行。

## 本地验证

先运行最相关的测试，再按影响范围扩大：

| 变更范围 | 最低验证 |
|---|---|
| 叶子模块 | 对应 unit 测试、`uv run ruff check .` |
| 数据契约、dataset、共享训练/评价 | 相关 integration 测试、完整 pytest、Ruff |
| 模型或 checkpoint | pipeline、resume、replay 测试；说明未覆盖的 CUDA/网络环境 |
| EODHD | fake transport 和临时目录；不得访问 live API 或刷新真实缓存 |
| 纯文档 | 链接和命令检查、`git diff --check` |

完整本地门禁：

```bash
uv lock --check
uv run ruff check .
uv run pytest
uv run pytest --cov=facdigger --cov-report=term-missing
git diff --check
```

只报告实际运行过的命令。无法覆盖网络、GPU、付费数据或正式研究时，应明确说明。

## CI

`.github/workflows/ci.yml` 在 push、pull request 和手动触发时运行，使用只读
`GITHUB_TOKEN` 权限，并将第三方 Actions 固定到完整 commit SHA。CI 在 Python
3.10、3.11 和 3.12 上执行：

```bash
uv lock --check
uv sync --frozen --all-extras
uv run --frozen ruff check .
uv run --frozen pytest
```

CI 不读取秘密、不访问 EODHD live API、不下载模型 checkpoint，也不运行正式训练或
holdout。合并前应确保所有矩阵任务通过；不能通过降低断言、跳过测试或放宽数据契约来修复
CI。

## 提交与评审

提交或 pull request 应保持单一目的，并说明：

- 问题和根因；
- 用户可见、数据或协议语义变化；
- 新增或更新的测试；
- 实际执行的验证及未覆盖环境；
- 兼容性、迁移步骤和剩余风险。

评审重点是正确性、防泄漏、数据和产物完整性、失败模式、测试有效性及文档同步。格式问题由
Ruff 和 CI 执行，不应占用人工评审的主要精力。
