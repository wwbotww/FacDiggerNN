# ICF 独立训练部署

本页只维护学校 Slurm、存储、凭据和启动包装。通用接口与恢复语义见
[训练运行与恢复](../../../docs/训练运行与恢复.md)，实验参数见
[实验设计](../../../docs/实验设计文档.md)。不要把站点账户/队列写进核心训练器。
已绑定的长作业保持 checkout、venv、实验配置和输出路径固定；部署配置与通用训练协议分别维护。

环境调查、作业编号与时点结果分别归档到
[截至 2026-09-28 的 ICF 记录](../../../docs/历史归档/ICF环境调查与验收记录_2026-09-28_当前可忽略.md)及
[2026-09-30 的长训练恢复验收](../../../docs/历史归档/ICF长训练恢复验收记录_2026-09-30_当前可忽略.md)。
后者记录最大 fold 基准结果和固定 `e165754` 后的恢复验证；**不表示完整矩阵已经完成，
也不是当前实时作业状态**。升级或换节点须重新核验相应范围。

第二批 R+C 有单独的 `scripts/icf/diagnostics.sbatch`，不走正式矩阵或故障注入入口。
通用协议、命令和预算见[诊断方案6.10](../../../docs/因子效果诊断与实验改进方案.md#batch2-execution)。
启动前显式设置 `FD_DIAG_CODE`、`FD_DIAG_COMMIT`、`FD_DIAG_PYTHON`、`FD_DIAG_ROOT`、
`FD_DIAG_DATASET`、`FD_DIAG_CHECKSUMS`；R 另需 `FD_DIAG_OLD_MATRIX`、`FD_DIAG_OLD_SNAPSHOTS`。
路径不含凭据，code 必须是隔离的干净提交。script 的一个位置参数为 R/cache/候选名。

| phase 参数 | sbatch 时限 | 内存 | GPU |
|---|---|---|---|
| R | `04:00:00` | `32G` | `h200_1g.18gb:1` |
| cache | `02:00:00` | `64G` | 无 |
| statistics_linear | `00:45:00` | `32G` | `h200_1g.18gb:1` |
| statistics_mlp | `00:45:00` | `32G` | `h200_1g.18gb:1` |
| finance | `06:30:00` | `32G` | `h200_1g.18gb:1` |

所有阶段4 CPU；显式传 `--output/--error` 到隔离证据目录。C 的三个作业依赖 cache 成功
并等待 R 结束；C 加 `--signal=USR1@180`，signal 交给 srun step 中的 TrainingControl。
默认禁止 requeue。失败后依据 sacct 已消耗时间核销本轮总预算，不能再次申请整份预算。
这些是学校部署值；普通服务器可以直接调用 research 脚本并保持相同科学配置。

完整 F/固定8日梯度诊断另用 `scripts/icf/fixed_diagnostics.sbatch`。保留上述通用路径
变量，`FD_DIAG_ROOT` 必须改成新证据目录；另设 `FD_DIAG_SOURCE` 为已完成的原 `C/`
父目录、`FD_DIAG_CACHE` 为原 `cache/`。code/commit 绑定新实现，原 R/C checkout 不动。
候选参数仍为 finance/statistics_linear/statistics_mlp。Finance与MLP分别申请2小时/20分钟，
`--signal=USR1@180`；2026-10-03获准扩展的Linear续跑申请45分钟，`--signal=USR1@300`，
内部预算2,100秒、退出余量300秒、累计内部预算上限3,600秒。45分钟与35分钟内部预算
之间另留10分钟覆盖启动和延迟退出，不能保证抢占内核中的磁盘等待。
每个仍为4 CPU、32 GiB、`h200_1g.18gb:1`，独立日志，禁止自动重排队。
Linear设置 `FD_DIAG_REUSE_LINEAR_CHUNKS=/path/to/previous-fixed/statistics_linear`，新输出
目录校验并复用旧的已提交块；原审计、代码身份和部分结果不改写。旧225秒allocation
另计，新45分钟allocation加旧用量不超过本次设定的1小时Linear累计上限。
这里没有训练更新或 S/V 重评分，原来2.5小时是首轮部署预算记录，不是科学协议限制。
通用计算与恢复边界见[固定状态入口](../../../docs/因子效果诊断与实验改进方案.md#batch2-fixed-execution)。

2026-10-03确认的廉价seed复现使用`scripts/icf/seed_diagnostics.sbatch`，两个位置参数
分别是`statistics_linear`或`statistics_mlp`、seed `17`或`73`。保留上述通用路径变量，
`FD_DIAG_ROOT`指向新的seed复现根目录，`FD_DIAG_CACHE`复用原RC缓存；不需要SOURCE
或历史矩阵变量。输出为`$FD_DIAG_ROOT/seed-<seed>/<candidate>`，不重训seed42。
包装固定F/S观察、FP32前向与完整F epoch1/2；训练仍保持原FP16/两轮前缀。

每run45分钟、4 CPU/32 GiB/一个`h200_1g.18gb`，内部及累计入口预算35分钟，其中
预留5分钟安全退出；外部另留10分钟启动/延迟退出余量，`--signal=USR1@300`，禁止
自动requeue。四run合计最多3 GPU小时，续跑前核销实际`sacct` allocation及入口attempt。
显式传日志路径和`--chdir`，固定干净代码提交；不要修改其他checkout、缓存或原C结果。
完整科学条件及跨平台命令见[方案6.17–6.18](../../../docs/因子效果诊断与实验改进方案.md#batch2-seed-execution)。

随后确认的单项正则对照复用该包装，参数为
`statistics_mlp <17|42|73> dropout-0.3`，显式传`--statistics-dropout 0.3`；
`.1`控制和Linear均不重训。`FD_DIAG_ROOT`必须使用新目录；三run每项仍45分钟，
合计最多8,100秒allocation。该参数独立于Finance YAML的`model.dropout`，后者
历史上不控制统计MLP。所有观察/恢复/预算规则相同，见
[方案6.21](../../../docs/因子效果诊断与实验改进方案.md#batch2-dropout-execution)。

## 1. 准备独立环境和稳定资产

固定已审阅提交的完整 clone 到 `/home/$USER/facdigger/code`，保留 `.git`、配置和
`uv.lock`；输出、配置、encoder、基准报告一旦绑定 run 就保持固定绝对路径。
不要从 AFS 启动作业，不复制生产的 CURRENT、ledger 或行情状态；数据 API 凭据按第 2a 节单独配置。
在准备节点安装一次环境，计算作业只运行既定 Python：

```bash
cd "/home/$USER/facdigger/code"
export UV_CACHE_DIR="/home/$USER/facdigger/cache/uv"
uv sync --frozen --python /usr/bin/python3 --extra data --extra model --extra eodhd --extra dev
uv lock --check --offline
mkdir -p "/home/$USER/facdigger/configs" "/home/$USER/facdigger/logs"
cp configs/deployment/icf/resources.example.env "/home/$USER/facdigger/configs/resources.env"
cp configs/deployment/icf/runtime.example.yaml "/home/$USER/facdigger/configs/runtime.yaml"
cp configs/deployment/icf/budget.example.yaml "/home/$USER/facdigger/configs/budget.yaml"
```

编辑自己的 `resources.env`：设置 `FD_CONFIG` 为冻结实验/研究配置，`FD_RUNTIME` 和
`FD_RESOURCE_BUDGET` 为刚复制的文件；替换 `FD_DATASET`、`FD_RUN_DIR` 中占位符。
文件会被 shell 显式 source，只使用自己维护的部署配置。所有路径使用绝对路径。
`FD_MODE` 可选 `finance-transformer`、`finance-pretrain`、`transformer-run`。
新 run 可冻结 `device: cuda`；已经开始的实验不得随意改配置哈希。

首次保持 `FD_STAGING_ROOT=""`，从 Lustre 读输入；输出始终留在 Lustre。
启用搬运可设 `FD_STAGING_ROOT="/disk/scratch/${USER}/facdigger"`，每次 allocation
重新创建唯一目录、检查空间、复制并校验所有文件，正常退出清理自己的临时目录。
硬杀可能留下临时目录；确认对应作业结束后单独清理，不能凭 PID 删除训练锁。
Lustre 和 scratch 均需外部独立备份。

## 2. 准备快照输入

按[通用迁移步骤](../../../docs/训练运行与恢复.md#2-选择数据准备方式)生成外置清单并验证全文件。
单模型完整 snapshot 无需 bronze/token；矩阵需要 wf1/wf2/wf3 的全部快照及对应 runtime。
没有 fold 映射仍需 bronze，不能把“代码已安装”当成训练输入已齐备。

runtime 指向持久输入，复制到 scratch 时包装生成本 allocation 的 dataset_overrides，
不把旧节点 scratch 保存成下次唯一来源。首次保持 FD_STAGING_ROOT 为空，验证从 Lustre
直接读取；启用 staging 后必须逐文件验证，硬杀残留只在确认对应作业结束后清理。
已绑定 output/run/encoder/admission 路径保持稳定，不能借 staging 改实验哈希。

## 2a. 在学校重新下载 EODHD 并用 CPU 预构建三个 fold

如果不迁移旧 bronze，使用独立 CPU 作业。先固定代码提交，在 Lustre 生成三份可审阅配置：

```bash
cd "/home/$USER/facdigger/code"
.venv/bin/python scripts/icf/configure_data.py --root "/home/$USER/facdigger"
cp configs/deployment/icf/data.example.env "/home/$USER/facdigger/configs/data.env"
bash scripts/icf/submit_data.sh "/home/$USER/facdigger/configs/data.env" --test-only
```

生成 `configs/eodhd.yaml`、`dataset.yaml`、`transformer.yaml`，仅替换部署路径；保留日期、
股票池、特征、标签、split 和训练配置。重复执行不覆盖人工改动。审核配置后再开始采集；
如需冻结新的 `device: cuda` 实验副本，先修改 `transformer.yaml` 中三份实验路径，并在
benchmark/训练前完成冻结。生成器不安装环境、不下载数据、不提交训练。

`data.env` 默认 `FD_DATA_MODE=plan`，4 CPU / 32G RAM / 4h，不申请 GPU，也不自动重排队。
这些是起始请求，不代表全量 RAM 和时限已经实测通过。先确认上面的 `--test-only` 成功，
再按以下顺序使用同一提交命令：

```bash
bash scripts/icf/submit_data.sh "/home/$USER/facdigger/configs/data.env"
```

| data.env 中的模式 | 工作与结果 |
|---|---|
| `plan` | 查询实时账户额度、活动/退市股票列表、有效缓存；写 `inputs/download_plan.json` |
| `ingest` | 显式改为此模式才全量采集；写独立 cache/state/bronze，仍执行原质量门禁 |
| `prepare` | 采集完成后改为此模式；CPU 生成三份不可变快照、外置清单和 `inputs/transformer/runtime.yaml` |

凭据为 `/home/$USER/facdigger/secrets/eodhd.env`，内容只包含一条
`export EODHD_API_TOKEN='实际值'`。目录必须本人所有且 `0700`，普通文件必须本人所有且
`0600`；拒绝符号链接。提交脚本只传部署变量，计算节点才读取文件，不执行凭据内容、不把
token 放入 Slurm 保存的环境。不要在 shell profile 中 source 凭据。`prepare` 不读取凭据。

预检在无缓存时需要两次成功的股票列表请求，重试另计；不会下载价格历史。
`max_symbols=1000` 是每日选股上限，
历史下载需要覆盖全部历史候选，不能按 1000 只估算额度。报告按候选数、三个端点与缓存命中
计算剩余 calls，分别列出无重试和所有请求都重试的估计。EODHD 的免费用量查询、UTC 日额度
以及每分钟 HTTP 限制见[官方额度说明](https://eodhd.com/financial-apis/api-limits)。

默认 `FD_RESERVE_API_CALLS=10000`、`FD_REQUESTS_PER_MINUTE=300`。每次付费请求及重试前
读取实时日额度，不使用额外付费包；额度不足保留余量并失败退出。账户查询也参与 HTTP
限速，因此最高数据请求吞吐小于 300/min。预留值需按实际生产消耗审核；这不是跨服务器的
原子配额锁，并发消费者仍可能在查询后花费额度。原生产入口不启用这套研究下载保护。

达到时限/额度或临时网络失败后，检查日志再手动重提 `ingest`。有效缓存避免重复请求；
TTL 到期、`refresh=true` 或更改日期会重新消耗额度。映射/拼接重做，全量聚合仍驻留 RAM，
不能把缓存恢复当成全流程流式恢复。首次全量必须观测 `MaxRSS`、空间和请求量。
三个模式持有同一个 Lustre 数据锁，防止该部署同时采集和构建快照。

`prepare` 每完成一个 fold 就保存进度；中断后复用已经校验的 fold，未完成 fold 重新计算。
恢复时拒绝源文件版本不一致、文件损坏或配置变化，不重新生成清单掩盖损坏。全部完成后重复
执行只验证，既不重新读取 bronze，也不改写快照。新的行情下载应使用新的准备目录和研究 run。
若一个 fold 的 CPU 构建超过申请时限，先在 QoS 允许范围内提高时限/资源；反复提交同一个
过短作业不会推进该 fold。这里的缓存/逐 fold 复用，与训练 update 边界的断点恢复分开验收。

后续训练的 `FD_CONFIG` 指向 `configs/transformer.yaml`，`FD_RUNTIME` 指向生成的
`inputs/transformer/runtime.yaml`；从 `folds.json` 取 `wf3.dataset_path` 作为最大 fold 基准输入。
同一份 runtime 可用于本地或其他服务器 CLI；搬运快照后显式更新路径并保留原外置清单。
CPU 准备还不表示 GPU 环境、订阅权限或正式研究 readiness 已通过。

2026-09-28 的市场日历修复需要重新构建金融快照（schema v5）。若已有 complete 的 v4
准备，保留原目录，复制 prepare env，并把 `FD_PREPARE_OUTPUT` 改为新的绝对路径，例如
`/home/$USER/facdigger/inputs/transformer-calendar-complete`。复用已验收的 bronze，
不重新下载、不改研究 YAML 或旧快照；完成后改用新目录的 runtime/fold 映射并重新 benchmark。
旧 v4 仍可校验和迁移，但重复验证旧准备目录不会自动升级。详情见[市场日历复盘](../../../docs/项目关键问题与修复复盘.md)。

长下载使用独立的 `data-ingest.env` 和 `data-prepare.env`，避免排队期间改写同一配置。
模板 4 CPU / 32G / 4h 只是起点。2026-09-27 全量下载实测超过 7 小时且接近 32 GiB，
同范围重建建议申请 ingest 64G 并核实 QoS 时限；不缩减数据来伪装通过。
CPU prepare 可单独配置资源。原始候选/请求量、作业 MaxRSS 与耗时见归档，不在操作指南滚动追加。

可在 `data-prepare.env` 显式设置 `FD_AFTEROK_JOB_ID` 为已经提交的 ingest 作业 ID，然后用
原提交命令排队。只接受正整数，Slurm 只有在上游退出 0 后才启动准备；上游失败则取消
不可满足的依赖。`FD_AFTEROK_JOB_ID=""` 保持原来的立即提交行为，不自动重试下载。
已提交作业读取的代码、YAML 和源文件保持固定；失败后核对日志、额度和缓存，再人工提交
新的 ingest，并将 prepare 依赖改成新的 ID。不能删除锁或降低来源质量门禁来继续。

## 3. 验证目标环境并测量预算

```bash
set -a
source "/home/$USER/facdigger/configs/resources.env"
set +a
sbatch --chdir="$FD_CODE_ROOT" --output="$FD_LOG_ROOT/env-%j.out" --export=ALL \
  scripts/icf/check_environment.sbatch
```

环境脚本申请单个 H200 MIG，检查包导入、分配设备数、实际容量及小型 FP16 前反向。
它记录实际版本，并不替代锁文件检查，也不证明完整模型/replay 已通过。
资源变更需同步 `check_environment.sbatch` 的请求或通过 sbatch 参数覆盖。
环境检查申请 15 分钟，考虑首次导入与共享存储延迟，不沿用余量不足的旧 5 分钟假设。环境导入和 checkpoint I/O 都要计入实际作业预算；
这项学校时限调整不修改通用训练器或实验配置。

在同规格 GPU allocation 内用 `srun` 执行下面基准；不要在头节点运行模型。
研究配置的 `admission_report` 必须指向该输出，三个实验配置路径在提交前冻结。

```bash
srun "$FD_PYTHON" -m facdigger train finance-benchmark \
  --supervised-config /path/to/frozen/scratch.yaml \
  --pretraining-config /path/to/frozen/pretrain.yaml \
  --dataset /path/to/largest-fold-snapshot \
  --updates 100 \
  --resource-budget "$FD_RESOURCE_BUDGET" \
  --output /path/to/frozen/admission.json
```

示例预算采用可见显存的 85% 上限、15 GiB RAM 和 14 天矩阵计算量；RAM 还会受可读取
的 cgroup-v2 上限约束，未知 cgroup 时用显式预算。不是使用物理整卡或整机容量。
硬件/预算变化须重测，原 RTX 报告不能当作 ICF 报告。当前 benchmark 没有完整测量
加载、selection/probe、保存/恢复和最终评价，因此需另用单 fold 短训练测量这些阶段。
300 秒退出余量及 600 秒 checkpoint 间隔只是起点；按最长完整 update 与写盘耗时调整。
`FD_MIN_OUTPUT_FREE_BYTES` 默认 2 GiB 是最低检查值，必须按实测 checkpoint/临时副本提高。

## 4. 先手动续跑，再启用受限自动续跑

```bash
bash scripts/icf/submit.sh "/home/$USER/facdigger/configs/resources.env" --test-only
bash scripts/icf/submit.sh "/home/$USER/facdigger/configs/resources.env"
```

`--test-only` 只验证调度请求。首次真实演练使用合成或已批准的验证数据、短预算和
`FD_AUTO_REQUEUE=0`；确认暂停后再次提交**同一** env、配置和 `FD_RUN_DIR`。
提交器从 runtime 的 `shutdown_margin_seconds` 向上取整生成 `USR1` 提前量，并导出
`FD_SIGNAL_SECONDS` 供作业核验；不要手填该变量或绕过提交器设置不同预警。修改 runtime
余量后重新提交，已排队作业发现预警与 runtime 不一致时会失败。

通过 `srun` 进入 `job.py --run-step` 学校适配入口，再调用原通用训练/研究实现。包装在
staging 后记录有限截止时间；step 完成依赖导入和配置读取后，重新读取 `squeue %L`，
取调度器剩余时间与原截止时间剩余量的较小值，建立共享控制器后才继续训练。
较短的 runtime 预算不会因启动或进入下一个矩阵阶段而重置。无法获得有限剩余时间、
启动已耗尽预算或不足退出余量均明确失败。

| 结果 | 包装行为 |
|---|---|
| 0 / complete | 正常结束，重复进入同 run 校验后返回 |
| 75 / paused / walltime_budget 或 time_limit_warning | 默认退出待手动重提；允许时请求同一作业 requeue |
| 75 / paused / SIGTERM | 保存后退出，不把人为取消推断为自动续跑许可 |
| 其他失败或无有效 paused 记录 | 停止并保留证据，不自动循环 |
| 调度器抢占后重启 / 硬杀留下 running | 同一 run 获锁后验证 last.pt，恢复最近已提交进度 |

只有真实信号链、锁和跨节点恢复验收通过后，才设 `FD_AUTO_REQUEUE=1`。
`FD_MAX_ATTEMPTS=10`、`FD_MAX_TOTAL_SECONDS=1209600`、`FD_MAX_NO_PROGRESS=2`
均持久化执行；10 次四小时作业最多约 40 小时，完整矩阵需要按基准明确提高次数上限。
每次在断点检查及环境加载前预记尝试和整个 allocation 剩余时长。正常退出和可捕获的
初始化失败按实际耗时结算；硬杀留下的未结算尝试在重入时按顺序核对，保守保留整段预算。
读取断点期间被杀也会计数；首次 checkpoint 尚未产生不能无限重新初始化。
同一已提交进度连续两次未推进会停止，不把 probe/market 阶段名变化当作推进。
负责核对并阻断的后续 allocation 也会登记并结算自身的检查开销，但不启动训练。
所有结果、计数和 scratch 清理先于主动 requeue，避免重排队杀掉旧 batch 后遗失结算。
达到限制需检查原因和预算，
再显式调整配置；不删除 `allocation_state.json` 清零。
`--requeue` 只是允许调度器重排，主动续跑由包装显式执行 `scontrol requeue`，不另发新作业。

## 5. 审计、回滚及生产交接

`FD_RUN_DIR/manifest.json` 是训练状态，matrix 另有 `matrix.json`/`folds.json`；
`checkpoints/last.pt` 是恢复权威，最佳导出用于候选模型。
`allocations/<job-id>-<attempt>/` 保存环境和 staging 清单，以及：

- `runtime.json`、`step.json`：启动前预算、截止时间和本次固定输入；
- `step_runtime.json`、`step_started.json`：扣除启动时间后的实际 runtime 与启动耗时；
- `result.json`：退出/中断/阻断结果、错误阶段、预算结算和前后已提交进度。

`allocation_state.json` 保存受限重试计数，Slurm 日志在 `FD_LOG_ROOT`。
JobID 不作为研究身份，多个作业可继续同一个 `FD_RUN_DIR`。

单模型/子 run 的 `progress.jsonl` 提供阶段耗时、进度、保存大小及当前/峰值内存；
矩阵根目录另记录 comparison。监控按 attempt 区分重启，按 `checkpoint_saved` 查看提交
时间；不能把中途进度当成可恢复断点，也不能累加所有嵌套阶段的耗时。
指标口径及仍需的真实 GPU 验收见[通用训练运行与恢复第 8 节](../../../docs/训练运行与恢复.md#8-长时间实验的验收与阶段观测)。

完整矩阵结束后，生成独立健康报告，输出放在研究 run 外：

```bash
"$FD_PYTHON" -m facdigger research transformer-audit \
  --run-dir "$FD_RUN_DIR" --output "$FD_ROOT/artifacts/transformer-health.json"
```

报告分开列出原 Rank IC acceptance 与六个监督 cell 的后半程健康检查；先验证九阶段产物。
`passed` 仍要求人工比较两组 score 波动、梯度稳定性及数据来源限制，不自动晋级模型。

保持旧代码环境和升级前的断点副本。新 reader 支持旧 epoch checkpoint；旧 reader
不能读取新的 epoch 内恢复 contract。不要回滚代码后继续读新 last.pt，也不要改历史哈希。
最佳 checkpoint 和 ModelRelease/FactorBatch 格式保持兼容，但真实候选仍须在生产固定
版本的隔离副本中执行 release 校验及离线回放。训练结束不自动晋级，不写生产目录。

## 6. 有日期的验收边界

2026-09-30 复核：最大 fold 的 100-update CUDA/FP16 基准通过。随后固定 `e165754`，
七项 CUDA/FP16 连续与中断恢复对照、真实最大 fold 的 USR1 暂停、完整包装自动 requeue、
内核 SIGKILL 后重入及无进展上限阻断均通过；学校 CPU 作业就地校验了断点和优化器状态。
同节点长验证作业 `3666742` 在记录时仍运行于首个 epoch 的 local 阶段，不能推断其当前状态。

完整预训练 local→market→probe、监督 selection/最终输出、至少 24 小时与多段长作业、
自然时限预警、实际抢占、跨节点 GPU 恢复、真实写盘期间突杀及独立故障域备份仍待验收。
该真实长验证使用单独配置；新部署启用自动续跑仍按第 4 节先验收，不能照抄时点配置。
正式九阶段矩阵和 holdout 未启动，来源 `research_ready=false` 未改变。
逐项证据、输入身份、资源测量与仍待项目见[2026-09-30 ICF 记录](../../../docs/历史归档/ICF长训练恢复验收记录_2026-09-30_当前可忽略.md)。
