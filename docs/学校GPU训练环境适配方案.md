**FacDiggerNN 学校 GPU 训练环境适配方案**

调查日期：2026-09-25。代码基线：`70390e3b3aa9cdcc9330fd6e0ef275655ba87834`。
适用环境：爱丁堡大学 Informatics DICE / ICF，当前账户 `s2977852`。

本文记录实际环境检查、学校部署配置和执行方案。**2026-09-25 已实现通用训练恢复、独立快照输入和学校包装；尚未把这套新包装部署到 ICF，也未迁移真实数据或启动正式训练。** 文中的实测资源属于此前诊断，不能代替新代码的 GPU/Slurm 验收。直接操作见 [ICF 部署说明](../configs/deployment/icf/README.md)。

方案分为两份独立文档：[训练可靠性与独立部署方案](训练可靠性与独立部署方案.md)定义适用于本地及其他服务器的恢复、限时、资源接口和兼容性要求；本文只定义 ICF 如何配置和使用这些能力。学校部署不成为项目的默认运行方式。

**1. 结论与部署边界**

学校环境可以承担模型训练和因子研究。当前账户已实际通过 Teaching 队列分配到 GPU，无需先假设尚未获得集群技术访问权限。建议保持同一仓库与共享核心代码，分别部署学校训练环境和现有每日生产环境，以审阅后的 ModelRelease 交接。

```mermaid
flowchart LR
    A[历史标准表及来源证明] --> B[CPU 构建训练快照]
    B --> C[ICF GPU 训练与评价]
    C --> D[保留训练记录与验证模型]
    D --> E[固定 ModelRelease]
    E --> F[现有生产机器每日推理]
    F --> G[FactorBatch 交付下游]
```

学校侧负责离线训练、预训练、评价和研究报告。每日行情更新、指定时点因子交付与下游交易保持在现有生产机器。Teaching 是批处理资源，具有时间限制和抢占机制，不适合承担需要固定发布时间的常驻生产服务。[学校 ICF 分区说明](https://computing.help.inf.ed.ac.uk/icf-informatics-compute-facility)也区分教学和研究用途；项目用途是否需要导师登记，应按教学项目流程核对，当前技术可访问性不替代用途登记。

训练不能只提取 `src/facdigger/training/`。其依赖包含 `models/`、`datasets/`、`features/`、数据契约、评价和实验记录。静态依赖检查确认，当前金融 Transformer 训练、预训练及精简研究入口没有引用 `production/` 或 EODHD provider。第一阶段保留完整源码，分离安装环境、配置、启动入口和资产目录即可，不需要复制维护两套模型代码。

**分离原则**：checkpoint、阶段恢复、原子状态和可选时间预算进入通用训练实现；`Teaching`、`teaching`、GPU GRES、Lustre/scratch、Slurm 信号和重排队策略只进入学校部署文件。学校代码、venv 和实验输出使用独立目录，现有生产环境继续固定原提交、依赖、release 和每日任务。学校训练完成只产生候选模型，不自动替换线上模型。

**2. 实际发现的机器、资源和账户限制**

以下空间和节点状态均为调查时快照，不是独占资源或未来容量保证。

| 对象 | 实际检查结果 | 对部署的影响 |
|---|---|---|
| 当前工作主机 | `selby.inf.ed.ac.uk`，MOTD 标识为 `student.compute.inf.ed.ac.uk` | 适合编辑、测试、数据准备；不是本次 GPU 训练入口 |
| 操作系统 | Ubuntu 24.04，x86_64；系统 Python 3.12.3 | 落在项目支持的 Python 3.10–3.12 范围内 |
| selby CPU/RAM | 双路 EPYC 7302，32 物理核 / 64 逻辑 CPU；约 503 GiB RAM | 是整机共享资源，不代表个人可全部占用；MOTD 要求长任务 nice 10–19 |
| selby GPU | 无 NVIDIA 设备或 `nvidia-smi`；PCI 显示管理用 Matrox 显卡 | 不在本机验证 CUDA 训练 |
| selby 本地存储 | `/disk/scratch` 剩余约 328 GiB；`/tmp` 剩余约 208 GiB | 仅属于 selby，不能在 GPU 节点直接使用同一批文件 |
| AFS 配额 | `fs listquota`：19,000,000 KiB，约 18.12 GiB；已用约 0.72 GiB | `df` 的 2 TiB 是 AFS 表观容量，不能作为账户容量 |
| ICF 入口 | `icf.inf.ed.ac.uk` → `hastings`；`icf2` → `stanger` | 已实际 SSH 登录 hastings，可使用 Slurm |
| 本人 Slurm 关联 | cluster=`landoniacluster`，account=`teaching`，QoS/default QoS=`teaching`，MaxJobs=20 | 可采用 `-p Teaching -A teaching --qos=teaching`；20 是关联限制，建议先并发 1 |
| Teaching 时限 | `MaxTime=2-00:00:00` | 单作业最长 48 小时；不能把“矩阵可在 14 天完成”当作单作业准入 |
| 抢占与约束 | `PreemptType=preempt/qos`、`PreemptMode=REQUEUE`、`JobRequeue=1`、`GraceTime=0`、`KillWait=30 sec`、`task/cgroup` | 要支持从脚本开头重新进入、识别既有 run 和可靠恢复；不能期待长时间退出缓冲 |
| Interactive | 调度器上限 4 小时，但头节点 MOTD 要求交互任务不超过 1 小时 | 诊断控制在几分钟；正式任务使用 sbatch |
| ICF-Free | 分区允许账户为 `research`；本人仅查到 teaching 关联 | 不能因为分区名含 Free 就把它作为本账户默认目标 |
| Lustre | `/home/s2977852`，所在共享文件系统约剩余 132 TiB | 是集群共享路径；`lfs quota` 本人 quota/limit 为 0，即此次未显示用户配额上限，不表示个人独占全部空间 |
| 容器工具 | selby、hastings 和诊断节点可运行 Apptainer 1.4.5；当前主机未发现 Docker/Podman 命令 | 首选 venv，Apptainer 可作为后续环境打包方式；尚未验收项目 SIF 镜像 |

头节点用于文件准备和提交任务，不执行训练或大规模特征构建。学校计算服务器的 scratch 是本机目录，且没有备份，见[计算服务器说明](https://computing.help.inf.ed.ac.uk/compute-servers)。

**3. GPU 和网络的实际诊断**

此前调查只运行两个限时 3 分钟的轻量诊断作业，没有项目训练或数据下载。`sacct` 记录两者均已 COMPLETED，最终本人队列为空。

| 诊断 | 实际结果 | 结论范围 |
|---|---|---|
| Job `3657573` | Teaching / teaching，节点 `opencast`，约 2 秒 | GPU 分配、驱动与磁盘查询完成；脚本内部 AFS 路径检查触发 PermissionError，后续网络探测未执行，不能仅凭退出码 0 宣称所有检查通过 |
| RTX 2080 Ti | 11,264 MiB，driver `580.178.04` | 可以作为 11 GiB GPU 候选；该次未验证项目锁定 PyTorch 或模型计算 |
| opencast scratch | `/disk/scratch` 已用 96%，当时剩余约 147 GiB；`/tmp` 约 9.1 GiB 可用 | 必须逐节点检查空间，不能按磁盘标称总容量选择 staging 方案 |
| Job `3657578` | Teaching / teaching，节点 `saxa`，约 5 秒 | 验证 H200 MIG 分配、网络、现有 PyTorch 和小型 FP16 运算 |
| H200 MIG | GRES=`gpu:h200_1g.18gb:1`；PyTorch 可见设备数 1、实际显存 17,179,869,184 bytes，即 **16 GiB** | 预算以进程实际可见显存为准；不能把物理 H200 的约 141 GiB 记为本作业可用显存 |
| saxa 驱动 | `595.71.05`；设备 capability `(9, 0)` | 属于本次节点实测，不代表所有 Teaching 节点 |
| saxa 本地环境 | `/opt/venv-cuda132-pytorch-2.12.1`：torch `2.12.1+cu132`、numpy `2.4.4`，没有 transformers | 可用于基础硬件诊断，不能代替项目锁定环境 |
| GPU 运算 | 32×32 FP16 矩阵乘法成功，元素结果 32.0 | 仅证明该环境的小型 CUDA 运算可用，不是 PatchTST、embedding replay 或正式训练验收 |
| saxa scratch | 当时约 3.5 TiB 可用；`/tmp` 约 42 GiB 可用 | 比此次 opencast 空间充足，但启动每个作业仍需重新检查 |
| saxa HTTPS | GitHub、PyPI、Hugging Face HEAD 返回 200；files.pythonhosted.org 根路径返回 HTTP 404 | 四个站点当时均可到达；404 是站点响应，不是 DNS/TLS 不通。未验证大文件下载、Hub CDN 或其他计算节点 |
| AFS | 两个计算节点访问当前项目 AFS 路径均出现 PermissionError | 所有训练输入与程序必须脱离当前 AFS 路径 |

集群还公布了 `gpu:nvidia_rtx_a6000:8` 和 H200 其他 MIG 类型。A6000 当时处于 allocated 状态，本次没有申请它；H200 `3g.71gb` 也未测量实际显存。公开教学集群页面描述 A6000 为 48 GB，但实际任务仍须读取分配结果。[教学集群说明](https://computing.help.inf.ed.ac.uk/teaching-cluster)。

GPU 选择以本项目基准为准：H200 MIG 的预算是切片资源，不能根据物理卡型号预设吞吐提升；2080 Ti、A6000 和不同 MIG profile 应分别测量。首次使用已确认可分配的单个设备，不申请整台节点或多卡来运行当前单卡代码。

为后续单作业预检，已执行 `sbatch --test-only` 验证 **1 个 H200 1g.18gb、2 CPU、20G RAM、4 小时**的资源请求可被调度器接受；没有实际提交该训练任务。该检查不验证程序、数据或资源峰值。

学校网络文档说明部分计算节点对外连接受地址路由影响。虽然 saxa 本次成功，正式训练仍应预先准备依赖与权重，避免运行中临时下载，见[集群网络说明](https://computing.help.inf.ed.ac.uk/cluster-networking)。

**4. 目录、数据和备份安排**

建议的持久目录如下；这些是拟采用的目录，本次没有在集群创建项目部署。

```text
/home/s2977852/facdigger/
├── code/                    # 固定提交的完整 Git clone，包含 .git
├── environments/            # 锁定环境或已验收 SIF 镜像
├── inputs/
│   ├── bronze/              # 历史标准表和来源 manifest
│   └── snapshots/           # 完整不可变训练快照
├── artifacts/               # run、checkpoint、评价、研究状态、ModelRelease
├── logs/                    # Slurm 日志与环境报告
└── cache/                   # 依赖/可选 HF 缓存，必须可重建

/disk/scratch/s2977852/facdigger/<job-id>/
└── inputs/                  # 当前作业输入副本，可在节点变更后重新生成
```

最小落地方案是：代码、独立 venv、run 和 checkpoint 使用固定 Lustre 路径；默认从 Lustre 读取输入，经测量后可选择把大型只读输入复制到节点 scratch。这样先保证恢复可靠，不把断点唯一副本留在随时可能丢失的节点本地目录。检查点保存频率经测量后可做本地写入加周期持久化，但不能仅依赖正常退出时的 shell trap 回传。

学校建议把频繁 I/O 放到本地 scratch，并提示共享盘和 scratch 均不能作为唯一重要数据副本。备份采用“集群保留可恢复工作集，校外原数据机器或其他已确认的持久存储保存独立副本”，避免把全部大文件塞入 18 GiB AFS。打包环境可减少大量小文件访问，见[GPU 集群使用建议](https://computing.help.inf.ed.ac.uk/cluster-tips)。

容量门禁应基于实际文件体积：输入解包体积、可能保留的压缩包、运行环境、checkpoint 临时写入与最新完整副本，再加余量。三 fold 快照不能假设与单 fold 大小相同。当前 Git clone 和本人 ICF home 中均未发现可用的项目历史行情、训练快照或模型权重，所以尚不能给出真实数据总容量与训练耗时。

训练资产迁移规则：

- 只跑单模型时可迁移完整 snapshot，包含特征、索引、scaler、manifest、来源证明，以及金融模型所需 market/pretraining 文件。
- `transformer-run` 已支持 runtime 中完整的 `fold_snapshots` 映射，携带三个经协议/身份/文件清单校验的 fold 快照即可脱离 bronze。省略映射时沿用原构建行为，缓存命中前仍需要标准 bronze；一个 snapshot 不能冒充完整矩阵。
- 标准 bronze 至少按原目录整体迁移 `bars_daily.parquet`、`universe_daily.parquet`、`corporate_actions.parquet`、`delistings.parquet` 和来源 manifest。
- 迁移后验证已有来源哈希；训练快照另附本次搬运的文件校验清单，用于发现传输损坏，不修改快照原始内容或重写其 manifest。
- 当前金融原生 scratch/finance-pretraining 路径不需要旧 ETTh1 外部权重；只有选用旧 E2/E3 时才准备锁定 revision 的模型资产。
- 使用已有标准数据训练不需要 EODHD token，也不迁移生产 SQLite、CURRENT 或线上生产配置。

**5. 软件环境选择**

首选项目独立的 Python 3.12 venv，按 `uv.lock` 安装 `data` 和 `model` extras；开发验收增加 `dev`，LightGBM 对照增加 `baseline`。核心依赖对照如下：

| 依赖 | 项目当前锁定值 | saxa 共享环境 |
|---|---|---|
| torch | `2.13.0` | `2.12.1+cu132` |
| transformers | `4.57.6` | 未安装 |
| numpy（Python 3.12） | `2.5.1` | `2.4.4` |
| Linux CUDA runtime 依赖 | `nvidia-cuda-runtime 13.0.96`，另有 cu13 库 | PyTorch build 为 CUDA 13.2 |

不要把学校的 `add_pytorch` 链接方案与原项目锁混合后仍声称是锁定环境。若未来确需采用共享环境，应作为独立、明确的依赖变更重新生成受审阅锁文件并跑回归；本阶段没有必要为此改动算法依赖。

按当前 Linux x86_64 / CPython 3.12 的 wheel tags 和依赖 markers，从 lock 静态计算：训练依赖共 66 个 wheel，压缩下载体积约 **2.70 GiB**，全部有匹配 wheel。该数字不包含项目构建工具、解压后的环境、下载缓存、数据或模型权重，不能当作安装后磁盘占用。建议环境与构建缓存先预留 15–20 GiB，再用实际安装结果修正。

两个已测驱动版本均达到 NVIDIA 列出的 CUDA 13.x 最低 major driver 580，但这不证明项目 wheel、目标 SM 或具体算子全部兼容。必须在锁定环境里重新执行导入、前反向和 replay 检查。[NVIDIA CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

依赖准备在头节点或其他可联网的准备机完成；不要在每次 sbatch 启动时执行联网安装。启动已安装环境使用确定的 Python 路径，或 `uv run --frozen --no-sync`，避免任务中修改环境。共享 venv 保持固定绝对路径，不直接复制 venv 到另一位置后假设脚本 shebang 仍正确。

如需进一步减少环境小文件访问，再提供 Apptainer SIF：锁定依赖安装到镜像固定路径，以 `--nv` 暴露已分配 GPU，显式挂载代码、输入、输出；保持 Slurm 设置的 `CUDA_VISIBLE_DEVICES`，包括 MIG UUID。容器能挂载固定逻辑路径，但镜像构建与实际执行还未验证。[Apptainer 1.4 GPU 支持](https://apptainer.org/docs/user/1.4/gpu.html)、[路径挂载](https://apptainer.org/docs/user/1.4/bind_paths_and_mounts.html)。

**6. 学校部署层需要提供的配置与包装**

通用恢复根因、格式设计、默认行为和旧产物兼容统一以[训练可靠性与独立部署方案](训练可靠性与独立部署方案.md)为准，不在学校分支复制训练循环。以下文件均已实现；通用代码不读取学校变量。

| 已交付 | 职责 | 边界 |
|---|---|---|
| `configs/deployment/icf/resources.example.env` | partition/account/QoS、GRES、CPU/RAM、单作业时限、固定代码/输出/日志路径 | 只由提交包装读取，不加进模型配置 |
| `configs/deployment/icf/runtime.example.yaml` | 为通用 runtime 接口提供 checkpoint 间隔、退出余量和信号处理设置 | 实际剩余时间及 scratch 输入映射按本次 allocation 生成；不修改实验协议 |
| `scripts/icf/check_environment.sbatch` | 检查导入、GPU/MIG 和小型 FP16 前后向，记录实际环境 | 不采集数据、不启动研究矩阵；锁一致性另用 uv 检查 |
| `scripts/icf/submit.sh`、`train.sbatch`、`job.py` | staging、空间预检、实际剩余时长、srun 启动、暂停/有限 requeue、作业证据 | 不实现优化器或寻找“最新 run”；仅读取明确绑定的恢复状态核验进展 |
| `configs/deployment/icf/budget.example.yaml` | 为通用 benchmark/runner 提供显存、RAM 和矩阵计算预算 | 原默认预算不变，未自动证明完整作业时限 |

配置示例不包含 token。学校值使用显式配置，不将当前用户名、`Teaching` 或 `/home/s2977852` 写入 `src/facdigger/training/`；其他服务器继续直接调用原 CLI。线程和离线缓存变量仅对该学校作业设置，不在包导入时修改全局环境。

建议首次测量值：

| 设置 | 初值 | 解释 |
|---|---|---|
| partition / account / QoS | `Teaching` / `teaching` / `teaching` | 本账户已核实；提交前按当时集群配置复查 |
| GPU / CPU / RAM / 时限 | `gpu:h200_1g.18gb:1` / 2 / `20G` / 4 小时 | 此组合通过过 `sbatch --test-only`，仍需模型实测 |
| checkpoint 间隔 | 600 秒起步，按写盘开销调整 | 只在安全边界保存；目标回退 10–15 分钟须另测最长边界耗时 |
| 退出余量 | 300 秒起步 | 必须覆盖最长安全边界、Lustre 保存和退出，不能只按经验固定 |
| 自动续跑 | 首次演练关闭，先手动重提同一 run | 验收后才启用有次数/总计算预算上限的续跑策略 |
| 模型设置 | 监督 microbatch=16、预训练 batch=32、FP16、num_workers=0 | 沿用已冻结实验；不放进站点 runtime 中自动覆盖 |

H200 MIG 的显存上限取 `torch.cuda.get_device_properties(0).total_memory`，此次为 16 GiB。可先按最多使用约 85% 显存建立测量预算；RAM 预算取显式申请与实际可见 cgroup 限制中的有效较小值，并给数据加载、checkpoint 副本和系统开销留余量。线程上限采用已分配 CPU 数，而非节点总核数。

benchmark 的可选资源预算属于通用改造，学校文件只提供数值。未选择学校配置时，现有 RTX 2070S 的 7.2 GiB GPU / 13 GiB RAM / 14 天门禁保留。ICF 报告还要记录 MIG profile、实际显存、包/驱动版本、最大 fold、保存/恢复耗时和单作业片段预算；至少 100 updates 的现有要求继续执行，probe 与最终评价需另测。GPU 类型变化后重新测量。

**6.1 Slurm 时限、抢占和重入执行流程**

稳定研究/run 身份必须写在 Lustre 上，与 `SLURM_JOB_ID` 分离。作业 ID 用于日志和诊断；同一研究可以经历多个作业 ID，重排队也可能保留同一个 JobID。

1. 提交入口使用固定代码和冻结配置，指定新研究或一个明确的既有 run。启动前确认不存在同目标活跃作业；通用 run 锁仍作为最终单写者保护。
2. 每次 allocation 都重新检查 GPU、配额和磁盘，按需将不可变 snapshot 复制到本节点 scratch 的临时目录。校验通过后才标记副本就绪；不沿用其他节点的 scratch 路径。
3. 通用运行控制接收本次实际剩余时间，扣除 staging/加载和退出余量。作业剩余时间来自调度器；不能在 staging 花掉一小时后仍给 Python 一个完整四小时预算。获取失败时明确停止，或使用事先确定的更保守固定上限，不猜测有无限时间。
4. 通过 `srun` 启动训练命令，让训练进程处于 job step 中；启用通用停止处理后再配置 `--signal=USR1@300`。不使用 `B:` 时预警发送给 job steps；如选 `B:`，必须另实现并测试 shell 到训练进程的转发。学校文档以第一种方式为首选。
5. 通用层完成安全保存后返回暂停状态。退出码 `75` 时，包装核验 run 确实 paused 且恢复状态已提交；只有原因明确为本次时间预算用尽或时限预警时，才按配置退出待重提，或请求一次受限 requeue。不能仅凭 SIGTERM 推断需要主动续跑；抢占重排队交给调度器，人为取消应停止。正常完成才报告训练完成，使用 `set -e` 的 shell 必须显式接住暂停退出码，其余失败继续失败。
6. 调度器因抢占重新执行脚本时，从第 2 步重新进入同一 run；硬杀未写 paused 时，由通用层检查该 run 的有效断点。已完成子阶段核验后复用。

`--requeue` 表示作业允许被重排队，不会让任意非零退出或主动暂停自动排入下一轮。主动续跑需要学校包装明确处理；先演练手动重提，再核实账户可执行的 requeue 策略。预警可能比指定时刻提前约 60 秒；它也不代表每次抢占都有五分钟缓冲。[Slurm sbatch](https://slurm.schedmd.com/sbatch.html)、[抢占处理](https://slurm.schedmd.com/preempt.html)。

不得对所有非零退出无限重排队。OOM、内容校验失败、协议不符、损坏断点或普通代码异常停止并保留证据；人为取消不自动重新提交。调度器自身的抢占重排队和脚本主动续跑不能同时再提交另一个独立作业。使用尝试次数、累计 allocation 耗时及无进展次数限制，计数保存在稳定路径，重新运行脚本不能清零。

**6.2 路径、部署独立性与后续优化**

研究输出、run、encoder、配置和 benchmark 报告使用固定 Lustre 路径，只有只读输入允许通过通用显式映射改到 scratch。`folds.json` 和完整配置哈希绑定的地址不做全局字符串替换；原 manifest/config/hash 保留。符号链接不等于路径虚拟化，因为现有代码多处使用 `.resolve()`。

首次部署也可以全部从 Lustre 读取，先验证恢复，再用 benchmark 决定是否 staging。如选容器固定逻辑路径，从实验开始即固定挂载约定，仍记录实际存储位置。任意移动整个进行中的研究输出不作为首轮自动恢复能力。

首轮顺序运行原九阶段，允许跨多个作业延续。阶段并行是后续优化：同 fold 的 pretrained 必须等待有效 encoder；汇总只有一个写者。BF16、更大 batch、多卡 DDP 属于独立实验改动，需重新冻结与验收，不在 ICF 启动脚本自动启用。

**7. 执行顺序与交付物**

| 阶段 | 执行内容 | 退出条件 |
|---|---|---|
| R：通用工程改造 | 按通用方案独立完成 R1 身份/原子状态、R2 epoch 内恢复/限时、R3 输入/资源接口 | 普通机器合成故障测试与完整回归通过；不需要学校配置即可验收 |
| A：冻结与资产准备 | 固定已验收提交；把独立 Git clone 和标准数据迁入 Lustre，建立外部备份 | Git 状态可追溯、来源验证通过、容量有余量；生产仍固定原环境 |
| B：学校独立运行环境 | 在固定路径安装锁定 extras；准备离线缓存；检查 Python/torch/driver/SM | doctor、锁检查、CUDA 前反向和 replay 通过；环境与通用代码解耦 |
| C：学校短作业演练 | 用学校包装验证信号、时间预算、单写者、重复提交、重排队和跨节点恢复 | 从同一 run 续跑；无重复更新/子 run；日志能区分暂停、失败与完成 |
| D：资源测量 | 在最大 fold 做至少 100-update 基准；测量 selection/probe、snapshot I/O、checkpoint 持久化和一次短恢复 | 实测 GPU/RAM/磁盘满足分配，片段耗时可落入时限，报告绑定配置、数据和硬件 |
| E：单 fold 演练 | 先完成一个 fold 的预训练及 scratch/pretrained 配对；限制同时 1 个 GPU 作业 | 日志、评价、恢复、模型回放一致，输出可完整迁回 |
| F：完整矩阵 | 执行既定 3 次预训练和 6 次监督训练；稳定后才并行独立 fold | 汇总使用同一协议；不因增加机器而改变数据或 selection；holdout 保持锁定 |
| G：生产交接 | 创建并验证候选 ModelRelease；在生产固定版本的隔离副本中做离线回放 | 兼容既有推理和交付契约；实际模型晋级另行执行，原每日流程持续可用 |

R 与 A/B 的资产和环境准备可以并行，C 依赖 R 的相关能力完成。代码交付拆为“通用训练可靠性”和“学校部署配置”两组，不把学校目录、队列及 signal/requeue 操作混进实验协议。待 D 测得真实性能后再决定正式时限、GPU 类型和是否采用 scratch。

先完成单卡可靠运行，不新增常驻调度服务。学校包装调用通用 CLI，续跑指向固定 run，不能复制实现另一套训练恢复逻辑。

**8. 环境安装与提交入口**

已交付的 [ICF 操作说明](../configs/deployment/icf/README.md) 是安装和提交命令的维护入口，包含以下顺序：

1. 独立固定 clone、锁定 data/model 环境、外部部署配置与持久输入；
2. 生成外置校验清单，选择单模型 snapshot 或三 fold 快照映射；
3. 提交轻量环境检查，再在分配到的 GPU 内完成 benchmark 和单作业阶段耗时测量；
4. 用 `scripts/icf/submit.sh resources.env --test-only` 检查资源请求；
5. 先短作业手动重提同一 `FD_RUN_DIR`，验证后再开启受限自动 requeue；
6. 保存环境、实际 runtime、allocation 计数及 Slurm 日志，再执行单 fold 和完整矩阵。

包装每次预记整个 allocation 剩余预算，正常退出后按实际耗时退还差额，硬杀保守计费，重启不清零。默认限制 10 次尝试、14 天累计预算、连续两次无进展；10 次四小时只提供约 40 小时，应根据基准明确调整次数。硬杀留下的 scratch 临时目录须确认作业结束后再清理；checkpoint 始终保留在持久路径。不要删除锁文件或计数文件规避冲突与限制。

这些是已实现的入口，但本轮没有执行 ICF 部署或提交新的作业。Apptainer、不同 GPU profile 与原生 Windows 文件锁仍按各自目标环境另行验收。

**9. 验收清单与必要回归**

平台无关的故障注入、旧配置/旧 checkpoint、legacy 模型与生产回归见[通用方案第 8 节](训练可靠性与独立部署方案.md)。学校侧在这些门禁之外补充：

| 验收 | 必须得到的证据 |
|---|---|
| 独立且离线的环境 | 锁定依赖与当前设备匹配；无需生产文件、EODHD token 或作业中联网安装；不会退回 CPU |
| 实际 CUDA/FP16 | 真实模型前后向及 embedding replay 通过；容差不因换平台放宽 |
| srun 信号链 | 短作业内确认预警到达训练进程；主动暂停写完整状态；不能只测试 shell trap |
| 重排队/新作业重提 | 两种入口均认领同一 run；活跃写者不会重复启动；人为取消和代码失败不自动循环 |
| Lustre 提交与锁 | 强制结束子进程后能读取最近完整状态；两个节点或两个作业不能同时写一个 run |
| 跨节点与 scratch | 重新 staging、验证全部文件、显式输入重定位；不依赖原节点缓存或 AFS |
| 作业时限 | staging、训练安全边界、selection/probe、最终评价、保存和退出均计入预算；至少能完成一个有进展片段 |
| 资源/空间 | 大 fold ≥100 updates；显存按分配 MIG 计算，RAM/CPU 按申请计算；计入最佳权重副本、临时 checkpoint 和数据副本 |
| 生产兼容 | 新候选 ModelRelease 在生产固定版本的隔离副本中通过 `release verify` 和离线回放；不直接切换现有每日生产 |

跨 GPU 不承诺逐 bit 相同。每组 scratch/pretrained 使用一致硬件和冻结配置，保留部署报告、作业日志及基准证据。MIG 基础矩阵运算成功、`sbatch --test-only` 成功均不能代替这一整套验收。

**10. 本次已完成与尚未验证**

此前调查已完成：本机 CPU/RAM/磁盘/AFS 配额；SSH 访问 ICF；本人 account/QoS、分区、时限、抢占配置和 Lustre 配额；两次短 GPU 作业；H200 MIG 小型 FP16 运算和 HTTPS 探测；调度器资源 dry-run。调查结束时没有留下排队或运行中的诊断作业。

本轮实施已完成：通用运行控制与固定 run、原子 checkpoint/manifest、epoch 内安全边界恢复、best 自足恢复、矩阵预登记与重入、显式快照重定位及三 fold 离线输入、可选资源预算，以及独立 ICF 配置、环境检查、staging 和受限续跑包装。原实验配置和锁文件未改；生产实现、状态和发布流程未部署或切换。

本机已安装项目锁定的 all-extras 环境，完成 CPU 故障回归、完整 pytest、Ruff、离线锁检查和 wheel 构建；最终数量与限制见[修复复盘第 58 项](项目关键问题与修复复盘.md)。脚本 fake Slurm 测试和 Bash 语法检查不等于真实调度器验收。

仍未验证：项目 torch 2.13.0 在学校 GPU 的完整模型/replay、真实数据容量与 100-update 基准、selection/probe/保存恢复的实际耗时、Slurm 抢占与预警链、Lustre 跨节点锁、项目 Apptainer 镜像、生产固定版本的真实候选回放、模型晋级及正式 Alpha。已有共享 PyTorch 的小型 GPU 运算不能替代锁定环境验收；真实退市收益及点时行业/市值限制继续有效。

下一步按 A/B 准备稳定输入和环境，再完成 C 的短作业恢复演练；D 的性能数据决定时限、checkpoint 间隔和资源配置。真实数据缺失不阻止本轮通用工程交付，但不能据 CPU 合成结果启动未经验收的完整矩阵或得出研究结论。
