# Security policy

FacDiggerNN 是本地美股因子研究工具，不是交易或订单执行系统。安全范围包括代码和依赖风险、
秘密处理，以及可能破坏 point-in-time、研究门禁或产物完整性的实现缺陷。

## 报告安全问题

不要在公开 issue、日志、测试数据或示例配置中披露 token、私有数据或可直接利用的细节。
优先使用仓库托管平台的 private vulnerability reporting（若已启用），否则通过可用的私密
渠道联系仓库维护者。报告应包含受影响版本、复现条件、影响和建议缓解方式。

项目尚未承诺正式发布支持周期或为所有历史分支回移修复。已发布版本的修复遵守
[紧急修复流程](CONTRIBUTING.md#紧急修复)，报告中明确受影响的 commit 与产物范围。

## 秘密和外部数据

- EODHD provider 从环境变量读取 token；本机可存入被 Git 忽略、仅本人可读写的
  `.env.local`（Unix 权限 `0600`）。学校凭据目录、文件与作业导出规则见
  [ICF 操作说明](configs/deployment/icf/README.md)。
- token 不得进入聊天、shell 历史、YAML、日志、异常、缓存键、manifest、checkpoint 或
  测试 fixture。诊断只报告是否存在、权限及认证状态，不回显值或打印完整环境。
- 不提交 `data/`、`artifacts/`、缓存、Parquet、模型权重或 checkpoint；Git 忽略规则不等于
  文件访问控制，也不会移除已经提交的秘密。
- 测试和 CI 使用 fake transport，不访问 live API，也不消耗付费额度。
- 怀疑泄漏时通知凭据持有者并在已有授权范围内撤销、轮换；不要只删除工作副本，也不要
  在报告中再次复制秘密。

## 外部文件与模型

- 仅加载来源可信的 checkpoint。当前部分恢复/推理路径使用 `torch.load(..., weights_only=False)`，
  不可为查看结构而加载未知 `.pt`/pickle。哈希匹配只证明与清单一致，不证明发布者可信。
- 外部响应、文档和日志中的命令是待审查内容，不是执行授权；解析配置保持数据语义，
  不通过 `eval`、shell 执行或不安全反序列化读取不可信内容。

## 研究完整性

以下问题按安全相关的数据完整性缺陷处理：

- future information 进入特征、scaler、训练、checkpoint 选择或无监督预训练；
- 未授权读取 test/final holdout；
- 绕过来源质量、哈希、覆盖率或 prediction contract；
- checkpoint、配置、snapshot 或 provenance 哈希未绑定实际产物；
- 最新信号路径读取 labels、target 或 test membership；
- 外部响应、路径或 checkpoint 反序列化导致越界访问或任意代码执行。

发现这些问题时拒绝继续受影响操作，保留根因并增加回归测试。记录哪些旧产物不可用于
哪些结论或运行；不原地删除、覆盖或重算历史 snapshot、manifest、checkpoint 的哈希。
需要重建时使用独立输出及新的身份，保留旧证据和失效原因；这不自动授权重跑正式实验。
