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

## 分支与发布

本节是分支、实验版本与发布流程的唯一维护位置；文档内容的职责划分见
[文档中心](docs/README.md#后续维护规则)。GitHub默认分支为`dev`，新开发从`dev`开始；`main`保留稳定发布职责。

| 分支 | 用途与集成方向 |
|---|---|
| `dev` | 日常开发集成；容纳经过相应检查的成果，不代表已获稳定发布批准 |
| `main` | 经确认具备发布条件的稳定版本；只接收已明确确认的候选 |
| `feat/*`、`fix/*`、`docs/*` | 围绕明确任务的短期分支，通常从`dev`创建，通过PR合并回`dev` |

`develop`是原开发分支的历史名称，统一迁移为`dev`，不长期维护两条开发集成线。
历史文档中的`develop@<commit>`仍描述当时事实，不改写已有提交身份。

### 日常开发

在没有其他未提交工作的开发checkout中，从最新`dev`创建任务分支：

```bash
git fetch origin
git switch dev
git pull --ff-only origin dev
git switch -c feat/short-task
```

首次没有本地`dev`时，先执行`git switch --track origin/dev`；修复、实验结果文档分别
使用`fix/short-task`、`docs/experiment-results`等有明确范围的名称。实现、必要测试和
对应文档在任务分支完成，PR以`dev`为目标，按下方风险矩阵执行检查；代码集成不能因为
末次提交只有文档，就省略整个改动范围所需的回归。

合并后清理完成的短期分支，不把单个功能分支长期当作开发主线。删除前再次确认全部
提交已被`dev`或`main`包含，没有未处理成果、开放PR或已发现的实验、部署、其他checkout
及协作依赖；不确定时保留并注明。普通合并或快进保留祖先关系；已用于实验的提交不做
rebase/squash或其他历史重写，不强推共享分支。

旧开发clone迁移时先核验工作区、独有提交及worktree，再改本地分支名和上游为`dev`/
`origin/dev`；改名本身不应移动HEAD或更新文件。固定实验checkout不参与日常分支迁移。

### 实验运行与结果整理

长实验使用确定的**完整commit SHA、干净的独立checkout、冻结配置和独立输出目录**；
环境准备好后固定venv。日常`dev`继续开发不切换正在运行实验的checkout、依赖或配置。
启动前记录代码、数据与计划身份；恢复或补跑记录实际科学代码和独立部署包装的身份，
不在线替换冻结版本，不改写旧审计以伪装一致。

seed、fold、候选和作业实例由冻结计划及产物身份管理，不逐个建立Git分支。结果整理走
`docs/*`→`dev`，保留正式结论的分析脚本及输入定位；实验设计、结果、当前判断分别更新
[文档中心规定的位置](docs/README.md#后续维护规则)。日常排队进度留在机器记录。

### 稳定发布

1. 从`dev`选定明确候选commit，说明相对`main`的变化、验证范围、兼容性和未解决问题。
   提交最新代码或CI通过，都不等于候选已具备发布条件。
2. 执行与发布范围匹配的回归和运行验收。共享训练/数据/评价改动按下方矩阵完整回归；
   涉及平台、恢复或生产接口时，说明实际运行验收及仍未覆盖的环境，不能用CPU测试替代。
3. **用户确认该具体候选具备发布条件后**，才通过PR将候选集成到`main`。确认绑定完整
   commit；候选变化须重新核对并确认，不能解释为对后续所有`dev`提交的批准。
   若`dev`已继续前进，PR须固定在获确认的候选，可用短期任务分支保留它，不建立长期release线。
4. 按实际发布策略和确认范围创建标签或GitHub Release，不因合并或CI成功自动创建。
   不为每个实验、seed或普通开发提交增加版本标签。

`main`的工程稳定性、模型研究有效性、生产部署授权分别表述。更新`main`不自动切换服务器
代码、正在运行的实验、每日生产模型、release或CURRENT；部署仍须有相应范围的明确授权。
本仓库CI负责检查，不承担自动部署或发布批准。

### 紧急修复

确须针对已发布版本修复时，从最新`main`创建`fix/*`分支。完成相应验证并获得该修复
候选的发布确认后，通过PR合并到`main`；及时通过正常合并将相同修复同步到`dev`，
避免两条线遗漏修复。修复后的部署同样不由分支合并自动触发。

## 变更原则

1. 修改前检查 `git status`，保留现有未提交改动，不混入无关重构。
2. 先定位现有实现、调用方和相似测试，再做能完整解决问题的最小变更。
3. Bug 修复应覆盖根因，并增加修复前能够失败的回归测试。
4. 配置、CLI、Parquet schema、manifest、checkpoint 或研究协议变化，应同步对应文档并
   说明兼容策略。
5. 影响 point-in-time、数据契约、snapshot、checkpoint、holdout 或长期架构的关键修复，
   需要在 `docs/复盘/` 的对应主题文件记录证据、验证和仍存限制，并更新
   `docs/项目关键问题与修复复盘.md` 索引。

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
