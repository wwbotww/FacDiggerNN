# FacDiggerNN Windows RTX 2070 Super 训练指南

> **现行平台操作指南**。文档职责和项目状态见[文档中心](README.md)。

本文用于把 macOS 上已经标准化的 EODHD 数据迁移到 Windows + RTX 2070 Super，并在
WSL2 Ubuntu 中运行工程实验。目标环境是 Python 3.11、单张 8 GB NVIDIA GPU、FP16。
历史 bronze 曾在项目机器上完成，但它是本地资产，不会随 Git clone 出现。

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
| 系统 RAM | 16 GB 可做优化版实测；32 GB 以上仍更稳妥 |
| 可用磁盘 | 至少 30 GB，建议 50 GB |
| 系统 | Windows 11 或支持 WSL2 的 Windows 10 |

2026-07-29 的内存优化除共享 `SecurityFeatureStore` 外，还加入训练窗口范围裁剪、
索引列裁剪、snapshot 分阶段释放，以及 E0 逐证券 Float32 统计和 LightGBM mmap 输入。
这移除了已知的 P0 级内存放大，但 16 GB 目标机尚未完成全 M6 峰值验收。16 GB 机器应先
运行单个 E1/E0 cell，确认系统未进入持续 swap 后再运行完整 M6；不要一开始并行训练多个
cell。32 GB 以上仍是更稳妥的正式实验配置。

完整日 objective v2 不会把约一千只股票的 activation 一次全放入 GPU。CPU
DataLoader 每次组装一个交易日的窗口，GPU 内只处理由 `batch_size` 限制的
physical microbatch；两遍回放仍使用整日 moments 和精确 score 梯度。相比旧的单遍
训练，它每日增加约一次不建图 forward；CPU 侧的额外活跃 batch 只是一日窗口，
但全体 feature store 仍然需要遵守前述 RAM 门禁。这一实现已通过 CPU 单元/集成测试，
尚未在本目标 CUDA 机上完成真实峰值或速度验收。

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
clone URL。不要盲目固定使用旧分支；正式实验必须记录并 checkout 已审阅、包含
完整日 objective v2、checkpoint schema v3、统计门禁和 final-refit 的确切
commit。

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

当前 context=512、patch length/stride=12/12 时，Transformers 从
`sequence_start=8` 开始生成 42 个 patch。本轮代码已让监督 AlphaHead 和 E3 reconstruction
的 observed mask 使用相同起点。修复前 artifacts 的 mask 从第 0 个 session 开始，
与真实 patch 错位 8 个 session；这些权重不能用于本轮矩阵，必须重新训练。

## 8. 构建快照与启动训练

先构建一个普通快照，记录系统 RAM 峰值：

```bash
uv run facdigger dataset build \
  --config configs/datasets/eodhd_historical_liquid.yaml
```

命令会输出内容寻址的 snapshot 路径。构建成功后，使用主配置的 FP16、
`batch_size=64` 先运行单个 E1 资源验收：

```bash
uv run facdigger train e1 \
  --config configs/experiments/e1_random.yaml \
  --dataset data/snapshots/<dataset_id>
```

至少完成一个 epoch 并检查：

- 峰值 VRAM 和 RAM 不会让系统进入持续 swap；
- `last.pt` 是 schema v3，objective 是
  `cross_sectional_rank_correlation_surrogate_v2_full_date`；
- `optimization_protocol.unit` 是 `complete_date`，physical microbatch 是 64；
- history 中 first/second-pass microbatch 计数非零，梯度范数为有限值；
- 中断后可从同配置、同 dataset 的 `last.pt` 恢复。

E1 通过后，再用 E2 或 E3 单 cell 验证 source 权重、FT-0/FT-1 和（E3）
reconstruction 路径。不要一开始并行训练多个 cell。确认 GPU 利用率、显存、RAM
和 checkpoint 正常后，再启动 M6 validation：

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

旧 Huber 监督目标、v1 date-chunk 排序目标或旧 holdout 协议生成的 checkpoint/artifacts
不能恢复到当前 run，也不能与当前完整日目标结果混合；迁移机器时只保留它们作为
历史证据。当前 `research_id` 是 `m6_eodhd_engineering_full_date_v2`，final holdout
仍保持锁定。
监督 E1/E2/E3 fine-tuning checkpoint 是 schema v3；E3 reconstruction checkpoint
是 schema v2，且必须包含
`patch_alignment_protocol=patchifier_sequence_start_v2`。旧 reconstruction checkpoint
既不能 resume，也不能作为当前 Alpha 模型的 financial encoder 来源。

为避免 2070 Super 首轮 3-fold × 3-seed 运行数周，main 配置已注册为硬件
可行性预算：E1/E2/E3 监督阶段 max/minimum epochs=5/3、patience=2，E2/E3
前 2 epochs 是 head-only；E3 reconstruction max/minimum epochs=3/1、patience=2、
learning rate=1e-5。E2/E3 head learning rate 为 3e-4。E3 的保守设置来自
artifacts2 中单 epoch 约 1.5—1.9 万个 optimizer steps 且 encoder 相对 L2 漂移约
0.43 的证据，用于降低灾难性遗忘风险。这些都不是已证明最优的参数。
若验证可行后升级硬件并扩大训练预算，必须预注册新配置和新 research ID 全量
重跑，不能把扩预算 cell 补进本轮矩阵。

## 9. 首轮停止条件

遇到以下任一情况先停止并修复：

- `torch.cuda.is_available()` 为 false；
- `data validate` 或 preflight 失败；
- snapshot build 被系统 OOM 杀死；
- 8 GB 显存 OOM；
- loss 为 NaN、checkpoint 不落盘或 GPU 长时间利用率接近 0。

显存 OOM 时先把对应监督配置的 `batch_size` 从 64 降到 32 或 16。它只是
physical microbatch 上限，降低后完整日 moments 和梯度仍然精确；**不要**联动提高
`dates_per_optimizer_step`，否则同时改变了每次参数更新前累积的完整日数。E3 的
`pretraining.gradient_accumulation_steps` 仅用于 masked reconstruction，保留原有语义。

需要注意，PatchTST 使用 train-mode BatchNorm，所以降低 physical `batch_size` 虽不会
改变“完整日目标”，仍会改变 BN batch statistics、吞吐和最终训练轨迹。一旦
确定可运行的数值，同一次正式对照的所有 E1—E3 cells 必须固定相同 physical
`batch_size`；如果从 64 改为 32，应更新配置、使用新 research ID 并整体重跑，不能
和 64 的 cells 混合。系统 RAM OOM 与显存无关，应增加 RAM，或进一步把 feature
store 改为 mmap/lazy reader。
