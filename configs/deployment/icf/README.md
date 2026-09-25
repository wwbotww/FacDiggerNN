# ICF 独立训练部署

本目录只配置学校资源；通用恢复在 `facdigger.training`，其他服务器直接使用同一 CLI。
资源调查与验收门禁见[学校方案](../../../docs/学校GPU训练环境适配方案.md)，
断点与兼容性见[通用方案](../../../docs/训练可靠性与独立部署方案.md)。
以下为实施后的操作步骤，尚未在真实 Slurm 作业中运行这套包装。

**1. 准备独立环境和稳定资产**

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

**2. 准备快照输入**

单模型携带完整 snapshot 即可，不需要 bronze 或数据 API。迁移前生成外置清单：

```bash
.venv/bin/facdigger train snapshot-checksums \
  --dataset /path/to/source/snapshot \
  --output /path/to/source/snapshot.checksums.json
```

清单和快照一起传输；在 runtime 的 `dataset_overrides` 中以真实 dataset ID 配置
`{path, checksums}`，验证目标副本。学校单模型 staging 还会对本次复制做全文件比对；
若使用已有迁移清单，先验证持久输入，再将其作为 staging 来源，不能重建清单掩盖损坏。

矩阵只带快照时，必须准备三份对应 fold 的完整快照，并在 runtime 写入：

```yaml
checkpoint_interval_seconds: 600
max_walltime_seconds: null
shutdown_margin_seconds: 300
handle_signals: true
dataset_overrides: {}
fold_snapshots:
  wf1:
    path: /home/USER/facdigger/inputs/snapshots/WF1_DATASET_ID
    checksums: /home/USER/facdigger/inputs/wf1.checksums.json
  wf2:
    path: /home/USER/facdigger/inputs/snapshots/WF2_DATASET_ID
    checksums: /home/USER/facdigger/inputs/wf2.checksums.json
  wf3:
    path: /home/USER/facdigger/inputs/snapshots/WF3_DATASET_ID
    checksums: /home/USER/facdigger/inputs/wf3.checksums.json
```

替换 USER 和 ID。每次提交保留同一份指向持久输入的 runtime；包装生成本 allocation
的实际映射，不把旧节点 scratch 路径保存为下一次唯一来源。不提供 `fold_snapshots`
时，新矩阵仍需要 bronze；第一次从 scratch 启动矩阵必须显式提供这三份输入。

**2a. 在学校重新下载 EODHD 并用 CPU 预构建三个 fold**

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

**3. 验证目标环境并测量预算**

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

**4. 先手动续跑，再启用受限自动续跑**

```bash
bash scripts/icf/submit.sh "/home/$USER/facdigger/configs/resources.env" --test-only
bash scripts/icf/submit.sh "/home/$USER/facdigger/configs/resources.env"
```

`--test-only` 只验证调度请求。首次真实演练使用合成或已批准的验证数据、短预算和
`FD_AUTO_REQUEUE=0`；确认暂停后再次提交**同一** env、配置和 `FD_RUN_DIR`。
训练命令通过 `srun` 接收 `USR1`，运行预算在 staging 后读取 `squeue %L` 剩余时长，
扣除配置的退出余量；无法获得有限剩余时间或不足余量则失败，不盲目启动。

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
每次先预记整个 allocation 剩余时长，正常退出再按实际耗时退还差额，硬杀则保守计费。
同一进度连续两次未推进会停止，不无限重复过长 probe。达到限制需检查原因和预算，
再显式调整配置；不删除 `allocation_state.json` 清零。
`--requeue` 只是允许调度器重排，主动续跑由包装显式执行 `scontrol requeue`，不另发新作业。

**5. 审计、回滚及生产交接**

`FD_RUN_DIR/manifest.json` 是训练状态，matrix 另有 `matrix.json`/`folds.json`；
`checkpoints/last.pt` 是恢复权威，最佳导出用于候选模型。
`allocations/<job-id>-<attempt>/` 保存环境、实际 runtime 和 staging 清单；
`allocation_state.json` 保存受限重试计数，Slurm 日志在 `FD_LOG_ROOT`。
JobID 不作为研究身份，多个作业可继续同一个 `FD_RUN_DIR`。

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
