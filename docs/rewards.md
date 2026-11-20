# 奖励

## 1. 奖励来源

训练奖励按动作块（chunk，一次推理执行的一段动作）计算：块内 L 步按每步折扣 γ 累加成一个奖励，自举用 γᴸ。

| 来源（`reward.source`） | 定义 | 依赖 |
| --- | --- | --- |
| `sparse` | 每步在环境判定成功（`done`）后为 0，之前为 −1 | 成功判定 |
| `rynnvalue` | sparse + κ·PBRS，势函数为 RynnValue 的价值估计 | RynnValue 评价 |
| `stage` | 关键帧插值的进度分数 xᵖ（p = `stage_exponent`），奖励就是分数本身，不取差分 | 人工关键帧 |
| `final` | Final Reward：`(1−α)·sparse + α·stage + κ·PBRS`（`fusion_mode: additive`）；或乘性形式 `multiplicative` | 按 α、κ 需要 Stage 和/或 RynnValue |

- `alpha=0, κ=0` 即 sparse；`alpha=0, κ>0` 即 RynnValue；`alpha=1, κ=0` 即 Stage。
- Final Reward 默认除以 −R0（完整轨迹第一个 chunk 的奖励，要求 R0 < 0），即 `final_normalization: initial_chunk_v1`；`none` 只用于复现历史的未缩放结果。
- γ 是 20 Hz 的每步折扣：一次转移的 L 步内逐步折扣累加奖励，自举用 γᴸ。乘性形式用区间终点的 Stage 分数缩放区间内的 sparse + κ·PBRS 回报。
- 基于势函数的整形（PBRS）理论上不改变最优策略，也不改变优势 Q−V（Q 与 V 同减 Φ(s)），实际效果取决于 critic 的估计误差。

Robometer 给出每条轨迹的进度和成功概率，用于评价展示和数据筛选，不是训练奖励来源。

## 2. 两个阶段：评价与物化

1. **模型评价**（GPU，代价高）：RynnValue 在 rynnvalue-reward 环境运行，Robometer 在 robometer-reward 环境运行，输出保存在各轨迹的评价目录。决定输出的设置（模型、revision、帧数、dtype 等）构成推理契约；契约和评价边界不变时直接复用已有输出，只重跑不兼容或缺失的轨迹。
2. **奖励物化**（CPU，代价低）：从保存的模型输出和关键帧计算 sparse、PBRS、Stage 和 Final Reward。只改公式参数（α、κ、γ、p、融合方式）时不需要再跑模型。

物化结果按"实现代码的文件内容 + 配置"建立指纹缓存：`rewards/manifest.py`、`pbrs.py`，以及按需的 `stage/`、`fusion.py` 变化后会重新物化，RynnValue 原始输出不受影响。

## 3. 全局结果与数据集结果

- 评价和关键帧可以是全局的（属于轨迹），也可以属于某个数据集。
- 数据集缺少某类结果时，继承同类型的全局结果，不会回退到其他奖励类型。
- 数据集专属结果只对该数据集生效；需要同步修改全局结果时，在数据集配置中开启相应选项。
- 冻结数据集保存当时的成功规则和奖励版本，保证历史训练可复现。

## 4. 配置与命令

`configs/rewards/reward.yaml`：前半部分决定 RynnValue 推理（`model`、`revision`、`max_frames`、`dtype`）；后半部分只用于从已有输出推导训练奖励（`fusion_mode`、`final_normalization`、`alpha`、`stage_exponent`、`gamma`、`shaping_weight`）。`configs/rewards/robometer.yaml` 配置 Robometer 评价。

平台在「数据集」页面完成全部操作。终端命令：

```bash
conda run -n rynnvalue-reward python -m rewards rynnvalue-annotate --config <训练配置>   # RynnValue 评价
conda run -n robometer-reward python -m rewards robometer-evaluate --config configs/rewards/robometer.yaml
conda run -n vla-liberox python -m rewards materialize --config <训练配置>             # 物化奖励
conda run -n rynnvalue-reward python -m rewards rynnvalue-verify                       # 检查环境
conda run -n robometer-reward python -m rewards robometer-verify
```

## 5. 代码

| 位置 | 内容 |
| --- | --- |
| `rewards/pbrs.py` | sparse 奖励、PBRS 整形、逐 chunk 奖励分量 |
| `rewards/stage/` | `keyframes.py` 关键帧标注的结构、校验与读取；`reward.py` 关键帧插值的进度奖励 |
| `rewards/fusion.py` | Final Reward 融合与归一化 |
| `rewards/manifest.py` | 奖励版本、缓存身份与物化 |
| `rewards/evaluation_store.py` | 评价结果的存储 |
| `rewards/rynnvalue/` | `model.py` 锁定的官方模型加载与推理；`outputs.py` 推理契约、输出校验与复用；`annotate.py` 批量评价 |
| `rewards/robometer/` | Robometer 运行时校验与评价 |
| `rewards/commands/` | 命令入口 |

RynnValue 和 Robometer 的上游源码与权重版本锁定在 `third_party/lock.yaml`，见 [third_party](../third_party/README.md)。
