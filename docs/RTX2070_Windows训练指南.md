# FacDiggerNN Windows RTX 2070 Super 训练指南

本文用于把 macOS 上已经标准化的 EODHD 数据迁移到 Windows + RTX 2070 Super，并在
WSL2 Ubuntu 中运行工程实验。目标环境是 Python 3.11、单张 8 GB NVIDIA GPU、FP16。

## 1. 迁移结论

可以从 GitHub 直接取得项目代码，但不能只下载 GitHub 就开始真实数据训练：

- GitHub 保存源码、配置、文档和 `uv.lock`；
- `data/`、`.env.local`、checkpoint 和训练结果被 Git 忽略；
- 真实训练至少还需要
  `data/bronze/eodhd_us_historical_liquid/`；
- 如果只训练已有 bronze，不需要把 EODHD token 复制到目标机；
- 如果目标机还要更新或重新 ingest，才需要额外复制 cache，并在目标机单独设置 token。

不要从 Mac 复制 `.venv`。虚拟环境包含操作系统、CPU 架构和绝对解释器路径，必须在目标机
由锁文件重建。

## 2. 硬件门槛

建议：

| 资源 | 要求 |
|---|---|
| GPU | RTX 2070 Super 8 GB |
| 系统 RAM | 至少 32 GB，64 GB 更稳妥 |
| 可用磁盘 | 至少 30 GB，建议 50 GB |
| 系统 | Windows 11 或支持 WSL2 的 Windows 10 |

共享 `SecurityFeatureStore` 已消除 E1/E2 三份、E3 五份完整特征块复制，但 snapshot
构建和首次 store 构造仍有一次性内存峰值。16 GB RAM 不建议启动全量 M6。

## 3. 安装 WSL2 和驱动

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

## 4. 从 GitHub 安装代码

在 WSL 的 Linux 文件系统中工作，例如 `~/FacDiggerNN`；不要把仓库长期放在
`/mnt/c`，大量 Parquet I/O 在 Windows 挂载路径上通常更慢。

```bash
sudo apt update
sudo apt install -y git curl build-essential

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

git clone -b develop git@github.com:wwbotww/FacDiggerNN.git
cd FacDiggerNN

uv python install 3.11
uv sync --frozen --all-extras
uv lock --check
```

若 WSL 尚未配置 GitHub SSH key，可以先配置 SSH，或在仓库允许 HTTPS 访问时改用 HTTPS
clone URL。

## 5. 验证 CUDA

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

## 6. 迁移真实数据

从 Mac 复制整个目录，不能只复制其中一个 Parquet：

```text
data/bronze/eodhd_us_historical_liquid/
├── bars_daily.parquet
├── universe_daily.parquet
├── corporate_actions.parquet
├── delistings.parquet
└── eodhd_ingestion_manifest.json
```

可使用移动硬盘、局域网或其他文件传输方式。若文件在 Windows 移动硬盘 `E:`，在 WSL
内通常可从 `/mnt/e` 读取；把它复制到仓库的 Linux 文件系统：

```bash
mkdir -p ~/FacDiggerNN/data/bronze
cp -a /mnt/e/eodhd_us_historical_liquid ~/FacDiggerNN/data/bronze/
```

复制完成后由通用 manifest 重新计算四张表的 SHA-256：

```bash
cd ~/FacDiggerNN
uv run facdigger data validate \
  --config configs/datasets/eodhd_historical_liquid.yaml

uv run facdigger research preflight \
  --config configs/research/m6_eodhd_engineering.yaml
```

预期 `data validate` 成功，并且 preflight 显示：

```text
ready: true
blockers: []
research_mode: engineering
```

`engineering` 是预期状态：当前退市终值为明确标记的插值，且缺少点时行业和流通市值。

## 7. 下载并验证 PatchTST 来源权重

目标机首次运行需要访问 Hugging Face：

```bash
uv run facdigger probe-patchtst \
  --config configs/base.yaml \
  --output artifacts/m0-probe
```

该命令会使用锁定 revision，执行权重兼容性检查和一次前反向。输出必须显示使用 CUDA 和
FP16。若目标机不能联网，可以把 Mac 的对应 Hugging Face model cache 迁移过去，再加
`--local-files-only`。

## 8. 构建快照与启动训练

先构建一个普通快照，记录系统 RAM 峰值：

```bash
uv run facdigger dataset build \
  --config configs/datasets/eodhd_historical_liquid.yaml
```

命令会输出内容寻址的 snapshot 路径。构建成功后，先运行 E0 或单个 E1 实验，不要直接
解锁 holdout。确认 GPU 利用率、显存、RAM 和 checkpoint 正常后，再启动 M6 validation：

```bash
uv run facdigger research run \
  --config configs/research/m6_eodhd_engineering.yaml
```

另开 WSL 终端监控：

```bash
nvidia-smi -l 2
```

训练日志和 checkpoint 位于 `artifacts/`。中断后使用已有 research run 恢复：

```bash
uv run facdigger research run \
  --config configs/research/m6_eodhd_engineering.yaml \
  --resume-run artifacts/research/<research_run_id>
```

不要在 validation 完成前使用 `--unlock-final-holdout`。

## 9. 首轮停止条件

遇到以下任一情况先停止并修复：

- `torch.cuda.is_available()` 为 false；
- `data validate` 或 preflight 失败；
- snapshot build 被系统 OOM 杀死；
- 8 GB 显存 OOM；
- loss 为 NaN、checkpoint 不落盘或 GPU 长时间利用率接近 0。

显存 OOM 时先把对应实验配置的 batch size 从 64 降到 32 或 16，并等比例提高
`gradient_accumulation_steps`，保持有效 batch 尽量一致。系统 RAM OOM 与显存无关，
应增加 RAM，或进一步把 feature store 改为 mmap/lazy reader。
