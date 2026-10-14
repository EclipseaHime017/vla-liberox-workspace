# 服务器训练架构

```text
server_pipeline.yaml + CLI
  → 模型家族／variant + BC/IQL 配置
  → 数据校验、保留成员与划分
  → Prepare + 固定奖励快照
  → 只读 mmap replay 缓存
  → 对应模型环境中的 torchrun
  → DDP actor、Q/V（IQL）、ZeRO-1
  → checkpoint、日志、policy overlay
```

server_inputs.py 管理输入；server_pipeline.py 管理任务生命周期；distributed_training.py 管理多进程训练。模型适配器提供原生损失、训练范围和保存接口，训练方法定义如何赋权。

rank 0 发布日志和产物，所有 rank 参与梯度同步及优化器状态汇总。ZeRO 按参数 dtype 分组，支持 π₀.₅ 原生 BF16/FP32 混合参数。mmap 保存训练所需图像和数值，避免每步重复解压 observation。

输入记录和评价数组是真源，训练不需要网页、SQLite 或实体控制器。每次训练固定自己的数据和奖励引用。
