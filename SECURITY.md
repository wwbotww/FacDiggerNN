# Security policy

FacDiggerNN 是本地美股因子研究工具，不是交易或订单执行系统。安全范围包括代码和依赖风险、
秘密处理，以及可能破坏 point-in-time、研究门禁或产物完整性的实现缺陷。

## 报告安全问题

不要在公开 issue、日志、测试数据或示例配置中披露 token、私有数据或可直接利用的细节。
优先使用仓库托管平台的 private vulnerability reporting（若已启用），否则通过可用的私密
渠道联系仓库维护者。报告应包含受影响版本、复现条件、影响和建议缓解方式。

当前项目尚无正式发布支持周期；默认只修复当前主分支。历史本地产物可能不兼容修复后的协议，
应按照修复说明重新生成。

## 秘密和外部数据

- EODHD token 只通过环境变量提供；本机可放在 Git 忽略的 `.env.local`。
- token 不得进入 YAML、命令输出、异常、缓存键、manifest、checkpoint 或测试 fixture。
- 不提交 `data/`、`artifacts/`、缓存、Parquet、模型权重或 checkpoint。
- 测试和 CI 使用 fake transport，不访问 live API，也不消耗付费额度。
- 怀疑秘密泄漏时应立即撤销并轮换，不要只删除 Git 工作副本中的文件。

## 研究完整性

以下问题按安全相关的数据完整性缺陷处理：

- future information 进入特征、scaler、训练、checkpoint 选择或无监督预训练；
- 未授权读取 test/final holdout；
- 绕过来源质量、哈希、覆盖率或 prediction contract；
- checkpoint、配置、snapshot 或 provenance 哈希未绑定实际产物；
- 最新信号路径读取 labels、target 或 test membership；
- 外部响应、路径或 checkpoint 反序列化导致越界访问或任意代码执行。

发现这些问题时应 fail closed、保留根因、增加回归测试，并在必要时使旧 snapshot、
manifest 或 checkpoint 明确失效。
