# ICF 独立训练部署

本目录只配置学校资源；通用恢复在 `facdigger.training`，其他服务器直接使用同一 CLI。
资源调查与验收门禁见[学校方案](../../../docs/学校GPU训练环境适配方案.md)，
断点与兼容性见[通用方案](../../../docs/训练可靠性与独立部署方案.md)。
以下为实施后的操作步骤，尚未在真实 Slurm 作业中运行这套包装。

**1. 准备独立环境和稳定资产**

固定已审阅提交的完整 clone 到 `/home/$USER/facdigger/code`，保留 `.git`、配置和
`uv.lock`；输出、配置、encoder、基准报告一旦绑定 run 就保持固定绝对路径。
不要从 AFS 启动作业，不复制生产的 CURRENT、ledger、行情状态或 token。
在准备节点安装一次环境，计算作业只运行既定 Python：

```bash
cd "/home/$USER/facdigger/code"
export UV_CACHE_DIR="/home/$USER/facdigger/cache/uv"
uv sync --frozen --python /usr/bin/python3 --extra data --extra model --extra dev
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

保持旧代码环境和升级前的断点副本。新 reader 支持旧 epoch checkpoint；旧 reader
不能读取新的 epoch 内恢复 contract。不要回滚代码后继续读新 last.pt，也不要改历史哈希。
最佳 checkpoint 和 ModelRelease/FactorBatch 格式保持兼容，但真实候选仍须在生产固定
版本的隔离副本中执行 release 校验及离线回放。训练结束不自动晋级，不写生产目录。
