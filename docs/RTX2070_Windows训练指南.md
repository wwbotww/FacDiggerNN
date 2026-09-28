# Windows RTX 2070 Super / WSL2 训练指南

本页只说明目标平台差异。数据准备、迁移清单、完整矩阵命令、进度观察和恢复统一见
[训练运行与恢复](训练运行与恢复.md)，模型与门禁见[实验设计](实验设计文档.md)。

## 1. 能否只下载 GitHub

不能。Git 只有代码、配置与锁文件，没有行情、快照或模型资产。
可以选择迁移完整 bronze 再构建，或直接携带**已验证的完整 fold snapshots + 外置 checksums**。
后者无需 bronze/token 即可离线训练；当前 Finance 主线也不需要 ETTh1 权重。
不要复制 macOS 的 `.venv`，应在目标系统重建。

RTX 2070 Super 为 8 GB 显存，16 GB RAM 应先跑最大 fold 的实际 CUDA/FP16 基准；
默认准入为 7.2 GiB GPU / 13 GiB 主机 RSS / 14 天矩阵估计。Windows、WSL、其他进程也占 RAM，
不能将系统物理 16 GB 当成全部可用，或通过持续 swap 宣称达到性能要求。
只顺序运行 cell，不并行训练多个 fold。CPU 快照构建与 GPU 训练峰值分别测量；
必要时在另一台机器预构建三个 fold 后搬运，不缩减训练数据。

磁盘应按完整快照、三 fold、原子 checkpoint 临时副本、输出与缓存实测留量，
不能仅用 Parquet 压缩体积推算 RAM/磁盘。学校另一 GPU 的合成测试不替代本机资源验收。

## 2. 安装 WSL2 和驱动

先安装或更新 NVIDIA Windows 驱动。然后用管理员 PowerShell 执行：

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --list --verbose
```

若刚安装 WSL，需要按提示重启。目标发行版的 VERSION 应为 2。进入 Ubuntu：

```powershell
wsl
```

在 WSL 中检查 GPU：

```bash
nvidia-smi
```

WSL 使用 Windows 主机映射进来的 NVIDIA 驱动，不要在 Ubuntu 内安装 Linux NVIDIA
display driver。项目不编译自定义 CUDA 扩展，正常安装 PyTorch wheel 时也不要求额外安装
完整 CUDA Toolkit。

## 3. 从 GitHub 安装代码

在 WSL 的 Linux 文件系统中工作，例如 `~/FacDiggerNN`；不要把仓库长期放在
`/mnt/c`，大量 Parquet I/O 在 Windows 挂载路径上通常更慢。

```bash
sudo apt update
sudo apt install -y git curl build-essential

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

git clone git@github.com:wwbotww/FacDiggerNN.git
cd FacDiggerNN

# checkout 维护者已审阅并冻结的本轮 commit，不依赖会移动的分支头。
git checkout REVIEWED_COMMIT_SHA
git rev-parse HEAD

uv python install 3.11
uv sync --frozen --all-extras
uv lock --check
```

若 WSL 尚未配置 GitHub SSH key，可以先配置 SSH，或在仓库允许 HTTPS 访问时改用 HTTPS
clone URL。正式实验使用已经审阅的当前 Finance 协议提交；不能用旧 M6 的
checkpoint/统计/holdout 能力推断新矩阵已经满足全部验收。

## 4. 验证 CUDA

```bash
uv run python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
uv run facdigger doctor
```

必须同时满足：

- `cuda=True`；
- device 为 RTX 2070 Super；
- `doctor` 中 `cuda_available` 为 true。

任一不满足时不要启动训练。优先检查 Windows NVIDIA 驱动、WSL 版本和是否误在原生
PowerShell 环境中安装了 CPU-only wheel。

## 5. 数据落地与路径

数据复制到 WSL Linux 文件系统后，先验证源端生成的全文件 checksums，再配置 runtime。
`fold_snapshots` 用于全矩阵输入；已有研究 run 输入重定位还要提供 `dataset_overrides`。
保留原 manifest/config/hash，不用文本替换把 Windows 路径写进不可变文件。

新 Finance 训练 snapshot 为 schema v5；旧 v4 需在新目录重建、重新 benchmark，不原地升级。
迁移一个已有 supervised run 回 macOS 发布时，`release create --dataset <本机训练快照>`
可显式覆盖定位，仍验证原始身份。进行中的 run/encoder/output 路径不能任意搬迁后继续冒充同一绑定。

## 6. 启动前检查与停止条件

按通用训练指南依次执行 prepare/迁移验证 → 最大 fold benchmark → 短作业恢复演练 →
九阶段矩阵 → 独立健康审计。不要直接把“CUDA 可见”当成完整模型已验收。

- CUDA 不可用、拿到 CPU-only torch、环境与锁不符：先修环境，不启动正式矩阵。
- 持续 swap、RSS/显存超限或预算超时：停止正式启动，定位瓶颈；不绕过 admission。
- 配置/输入变更、断点损坏、重放误差或非有限梯度：保留证据，不能跳过检查继续。
- 主机休眠/关机可能硬中断；启用显式 runtime 周期保存并预留稳定电源/运行时间。
- WSL2 按 Linux 信号路径演练；原生 Windows 的锁/信号行为不能从 WSL 测试推定。
  不支持 SIGUSR1 时采用通用时间预算，不把信号处理隐式开启给所有进程。

旧 E0–E3/M6 的可选命令留在[旧实验协议](历史归档/E0-E3与M6实验协议_2026-09-28_当前可忽略.md)。
本页不再混列两套“下一步训练”清单，也不声称新的 RTX 完整实验已经通过。
