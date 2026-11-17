# 训练

## 方法

| 方法 | 说明 | 需要奖励 |
| --- | --- | --- |
| BC | 等权行为克隆：直接学习数据集中的全部有效动作 | 否 |
| IQL | 奖励加权后训练：expectile 训练 Q/V，按 `min(exp(β·A), 上限)` 加权 actor 的 BC 损失 | 是，先在数据集配置中准备评价和 Final Reward |

方法由登记表声明，平台、配置校验和训练循环都从登记表读取；新增方法的步骤见 [训练方法](methods.md)。

## 在网页中训练

在「训练」页面选择数据集、基础模型和训练方法，然后创建任务：

- 「模型训练配置」选择训练哪些模型组件；基础参数和高级参数面板调整步数、batch size、梯度累积、学习率和 critic 设置。
- IQL 需要所选奖励已就绪。训练奖励可选 Final Reward 或 RynnValue；数据集缺少时继承同类型的全局结果，不会回退到其他奖励。γ 和 cumulative reward 只影响本次训练。
- 训练进度、日志和队列显示在页面下方；完成后导出的模型出现在「模型」页面，可直接用于仿真和测试。
- 网页创建的训练写入 `dataset-root/projects/<项目>/training/`。

**数据切分**

- 动作监督长度取模型 YAML 中的 `io.replay_horizon`。
- 策略段按实际推理调用边界切分；人工段按该次仿真记录的执行步数切分。
- 遇到接管、指令切换或轨迹终点时截断，并加 mask。
- 旧记录缺少执行步数时，使用来源模型的默认值；无法识别来源时明确报错。

旧奖励与新切片不一致时，需要在数据集配置中重新生成；关键帧无需重新标记。

## 在终端训练

**单卡流水线**：一次完成数据选择、准备、评价和训练。编辑 `configs/pipeline.yaml`：`base_config` 选择 `./methods/iql.yaml` 或 `./methods/bc.yaml`，`selection` 选择数据，`overrides` 覆盖训练参数。BC 会跳过全部奖励评价。

```bash
conda activate vla-liberox
python -m train pipeline --dry-run    # 只检查配置和数据，打印计划
python -m train pipeline              # 显示计划后输入 y 开始；无人值守时加 --yes
```

`--force-prepare`、`--force-annotate` 分别强制重新准备数据和重新评价。

**分步执行**：

```bash
python -m train prepare-dataset --config configs/methods/iql.yaml
python -m train train --config configs/methods/iql.yaml --model <权重变体ID>   # --method bc 覆盖方法
python -m train export-checkpoint --checkpoint <checkpoints/step_N> --output policy-registry/<ID>
python -m train export-policy --policy <policy.yaml> --output <目录> --base-checkpoint <新设备上的基模路径>
```

命令行训练默认输出到 `outputs/training/`（`configs/runtime.yaml` 的 `paths.output_dir`）。`checkpoints/step_*/state.pt` 是训练恢复文件；用于仿真和评测的是导出目录中的 `policy.yaml` 及同目录权重。跨设备使用导出模型时，用 `export-policy` 改写基础权重的位置。

多卡服务器和 PBS 集群见 [服务器](cluster.md)。

## 监控

- TensorBoard 默认开启，平台在「训练」页面提供链接（`configs/studio/ui.yaml` 的 `tensorboard_port`）。
- W&B：先 `wandb login`，再在训练配置中开启。
- 终端日志每隔 `console_interval_steps` 打印损失、Q/V/优势、学习率、显存和预计完成时间。

## 代码

| 位置 | 内容 |
| --- | --- |
| `train/trainer.py` | 单卡训练循环、checkpoint、恢复、导出 |
| `train/distributed.py`、`train/methods/distributed.py` | 多卡训练 |
| `train/replay.py`、`train/replay_cache.py`、`core/data.py`、`core/frame_cache.py` | 回放数据、切片和帧缓存 |
| `train/models/` | 模型家族实现：`vla_adapter.py`、`pi05.py`、`pi05_worker.py`、`adaptation.py` |
| `train/methods/` | 方法实现，见 [训练方法](methods.md) |
| `train/export.py`、`core/artifacts.py` | 导出与策略契约 |
| `train/pipeline.py`、`train/commands/` | 终端流水线和命令入口 |
| `train/monitoring.py`、`train/diagnostics.py`、`train/critic_diagnostics.py` | 进度、指标与 critic 诊断 |
