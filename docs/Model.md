# 模型配置与接口

configs/models/registry.yaml 索引家族 YAML。家族定义 adapter、输入输出、训练范围及运行环境；variants 定义权重来源、部署路径、归一化和默认执行步数。训练通过 model.family 和 model.base_id 选择模型。

现有家族 vla_adapter 支持 frozen/LoRA/full，可独立冻结动作头和 proprio projector；pi05 支持 frozen 动作专家训练/full。可在 server_pipeline.yaml 的 overrides.model 配置。

## 新增权重变体

将下载模型放入 models/<模型名>/，在对应家族 YAML 的 variants 下加入：

```yaml
new-base:
  label: New model
  source: publisher/repository
  revision: null
  checkpoint: ../../../models/new-model
  stats_key: normalization-key
  observation:
    image_flip_vertical: true
    image_flip_horizontal: true
  inference:
    open_loop_steps: 5
```

stats_key 对应该权重的归一化资产，已知的 revision 使用固定 commit；inference.open_loop_steps 由各变体独立配置。observation 定义相对原始相机帧的垂直／水平翻转；训练会先还原记录的存储朝向，再统一转换 actor 双视角和 critic 当前／下一观测，不重复翻转。修改视角后缓存自动失效，不能继续恢复视角不同的 checkpoint；已导出的子模型保留自己的视觉配置。该设置不改变原始数据或评价模型输入。

家族通过 io/architecture 声明输入输出。io.replay_horizon 是 actor 监督窗口，io.native_action_horizon 是模型预测长度；当前监督窗口分别为 8 和 10，历史子模型继续继承其保存的窗口。transition 使用记录中的真实预测／执行边界，独立于监督窗口，并在接管、指令切换等边界处中断。

π₀.₅ 的 JAX 权重通过 prepare_pi05.py --base-model <ID> 转换。LIBERO-X 变体仅上下翻转，现有 LIBERO 变体保持双轴翻转；自训练权重应按其训练输入配置对应 variant。图像先还原存储朝向，再使用模型自己的输入预处理；VLA-Adapter 的 JPEG、缩放和中心裁剪与上游推理一致，TensorFlow 仅使用 CPU。

```bash
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_server.py --model new-base --method bc --dry-run
```

## 新增家族

在 configs/models/registry.yaml 索引家族 YAML，YAML 的 adapter 对应 model_families/ 中的轻量适配器。适配器负责配置校验、训练参数签名、权重身份和 backend 选择，不加载模型或 GPU；新增适配器时在 model_families/__init__.py 注册。

运行 backend 实现 load_components、actor_losses、trainable_parameters、parameter_counts、save_actor、restore_actor、export_actor。已有家族的新权重只需新增 variant；新架构需要实现对应 backend 和导出校验，不由训练流程猜测其模型类型。

actor_losses 接收独立的 actor_actions、actor_action_mask 和原始动作，返回逐样本原生损失与可选预测。BC 求平均，IQL 使用优势权重加权；critic 使用 transition 的 actions、action_mask。算法层不需要识别动作头或 flow 模型。

训练子模型保存父权重内容身份、继承的 I/O 和更新参数。部署设备需保留对应基础权重。跨机器路径不同，可生成一个部署副本：

```bash
python vla-adapter-rynn-iql/scripts/export_policy.py --policy "policy-registry/<id>/policy.yaml" --output "/mnt/export/<id>" --base-checkpoint /target/workspace/models/pi05_libero_torch
```

此操作更新路径和对应元数据哈希，父权重内容 revision 保持不变；原训练结果不会修改。
