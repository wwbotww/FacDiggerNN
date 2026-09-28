# FacDiggerNN

FacDiggerNN 是面向美股日频横截面选股的机器学习因子研究 CLI：将 EODHD 或标准 Parquet
转换为可审计快照，训练模型、评价排序质量，再输出供交易系统消费的因子。
它不是交易系统，不下单，也不管理仓位或资金。

当前研究主线是 **Finance Transformer 从头训练与金融预训练的配对对照**：
3 个 walk-forward folds、1 个 seed、3 次预训练和 6 次监督训练。
E0–E3 与 M6 仍保留，但不是默认实验矩阵。完整协议见[实验设计](docs/实验设计文档.md)。

## 能做什么

| 环节 | 能力 |
|---|---|
| 数据 | EODHD 供应商隔离、标准表及来源证明、动态流动性股票池、质量审计 |
| 训练 | 14 路个股 + 6 路市场输入、512-session 上下文、完整日横截面排序、显存受限精确梯度回放 |
| 评价 | IC/Rank IC、稳定性、成本参考、覆盖检查；旧 M6 另有统计决策与冻结/holdout 流程 |
| 运行 | CPU 预构建 fold、离线快照迁移、资源准入、Finance 训练 update 边界保存与恢复 |
| 交付 | 统一 ModelRelease、无标签推理、单日/全历史 FactorBatch、Docker 每日生产 |

```text
供应商 -> 标准表 + 来源证明 -> 不可变训练快照 -> 训练 / 研究 / 评价
                                               -> ModelRelease
独立生产行情 -> 无标签推理快照 + 冻结 scaler -> 完整横截面评分 -> FactorBatch
```

代码能力不等于环境或研究验收。学校 GPU 的合成演练、真实数据准备和基准进度见
[截至 2026-09-28 的记录](docs/历史归档/ICF环境调查与验收记录_2026-09-28_当前可忽略.md)；
826 生产的末次已验证交付见[运行交接](docs/826每日生产运行交接.md)。
这些是有日期的证据，不是实时状态，也不证明新模型已完成训练或存在有效 Alpha。

## 安装与最短入口

支持 Python 3.10–3.12，推荐 3.11。使用固定、已审阅的代码提交和锁文件：

```bash
git clone git@github.com:wwbotww/FacDiggerNN.git
cd FacDiggerNN
# 正式运行前 checkout 已审阅的确切 commit，记录 git rev-parse HEAD
uv sync --frozen --all-extras
uv run facdigger doctor
uv run facdigger research transformer-plan \
  --config configs/research/finance_transformer_streamlined.yaml
```

最后一条只查看实验计划，不启动采集或训练。Git 不包含数据、权重、快照或结果；
仅 clone 代码不能直接开始正式训练。后续按[训练运行与恢复](docs/训练运行与恢复.md)
准备数据、测量最大 fold、通过资源门禁后再运行。

`uv.lock` 是首选依赖锁；`requirements-lock.txt` 是同源的 pip fallback。
不要跨系统复制 `.venv`。RTX 2070S / 16 GB 机器见
[Windows/WSL2 指南](docs/RTX2070_Windows训练指南.md)；
学校集群见 [ICF 操作说明](configs/deployment/icf/README.md)。

## EODHD 凭据

只有在线采集、额度预检和每日生产需要 token；使用已验证快照训练不需要。
自行创建被 Git 忽略的 `.env.local`，权限设为 `600`，内容为：

```dotenv
EODHD_API_TOKEN=your_token_here
```

在自己的终端加载，不在日志中打印：

```bash
set -a
source .env.local
set +a
```

不要写入 YAML、缓存键或 manifest。学校作业使用隔离的凭据文件，不随 Slurm 环境导出；
详见站点说明和[安全规则](SECURITY.md)。

## 按目标阅读

| 目标 | 入口 |
|---|---|
| 理解模块、数据契约和扩展方式 | [开发文档](docs/开发文档.md) |
| 理解模型、数据、对照和验收 | [实验设计文档](docs/实验设计文档.md) |
| 准备/迁移数据、训练、观察进度、恢复 | [训练运行与恢复](docs/训练运行与恢复.md) |
| 给 HeyBoss 提供当日或历史因子 | [HeyBoss 因子联调交接](docs/HeyBoss因子联调交接.md) |
| 部署和排查每日服务 | [每日生产运维](docs/每日生产运维.md) |
| 定位真实故障与设计理由 | [关键问题与修复复盘](docs/项目关键问题与修复复盘.md) |
| 查阅其他材料和历史方案 | [文档中心](docs/README.md) |

## 结论边界

- 缺少可靠的点时行业、市值和真实退市终值，当前实验属于 engineering research。
  不因暂时关闭中性化硬门禁就宣称模型已经中性化；组合收益只是参考评价。
- 固定模型全历史回放用于跑通工程，可能含相对历史日期的未来信息，不是严格样本外证据。
- 训练、无标签推理和外部交付分开；每日任务不修改训练快照，不回退旧因子，
  不以 ticker 猜测交付身份。局部缺分不等于卖出指令。
- 新 Finance 训练快照是 schema v5；旧价量快照 v4 和 Finance 最佳模型 checkpoint v4
  是不同契约。旧不可变产物不原地升级。

开发和最低验证要求见 [CONTRIBUTING.md](CONTRIBUTING.md)；
coding agent 规则见 [AGENTS.md](AGENTS.md)。
