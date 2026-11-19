# 训练方法：接口与新增步骤

## 现有方法

| 方法 | actor 训练 | critic | 推理 | 模型家族 |
| --- | --- | --- | --- | --- |
| BC | 等权行为克隆 | 无 | 直接执行 actor 的动作块 | 全部 |
| IQL | 按 `min(exp(β·A), 上限)` 加权的 BC | expectile 双 Q + V | 直接执行 actor 的动作块 | 全部 |
| IFQL | 等权 flow BC | expectile 双 Q + V | 采样 N 个动作块，执行在线 critic `min(Q1, Q2)` 最高的一个 | π₀.₅ |
| VF（Value Flows） | 等权 flow BC | 双 flow 回报分布 critic | 采样 N 个动作块，执行单步 flow Q 最高的一个 | π₀.₅ |

IFQL 是 IDQL 的 flow 策略版本，参考 [seohongpark/fql](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/ifql.py)（[value-flows](https://github.com/chongyi-zheng/value-flows) 中的副本相同）。与参考实现的差异：按动作块建模，每块一次折扣；critic 使用像素编码器；V 与 Q 依次更新、各用一个优化器。默认值见 `configs/methods/ifql.yaml`（expectile 0.9、tau 0.005、N=32）。

Value Flows 参考 [chongyi-zheng/value-flows](https://github.com/chongyi-zheng/value-flows/blob/01833354f547ad842bdaa21d2b761a16069f9724/agents/value_flows.py) 的离线设定（`'rs'` 策略提取）：critic 是两个回报向量场，用 BCFM 与 DCFM 损失训练，并按回报不确定度（flow 对噪声的 jvp）加置信度权重；下一状态的动作块在训练时由当前 actor 采样 `train_num_samples` 个、由在线 critic 选出最优的一个；回报裁剪到训练回放的奖励范围除以 (1−γ)。与参考实现的差异：按动作块建模；critic 使用像素编码器；参考实现中只服务在线 `rpg` 提取的单步蒸馏 actor 及其 Q 损失未移植；采样的下一动作块按该模型默认的执行步数打分。默认值见 `configs/methods/vf.yaml`（Adam 3e-4、tau 0.005、N=16、10 步积分）。训练时每个可自举的转移都要对 π₀.₅ 采样一次，单步耗时明显高于 IFQL。

## 一种方法由哪些部分组成

平台、配置校验和训练循环只读下列声明，不写死方法名：

| 层 | 位置 | 内容 |
| --- | --- | --- |
| 声明 | `core/methods.py` 的 `METHODS` | 方法需要什么、产出什么（不导入 torch，平台进程直接读取） |
| 配置 | `core/config.py` 的 `TRAIN_SCHEMA[<段>]`，`configs/methods/<方法>.yaml` | 方法段的字段与默认值；字段取值规则在 `_SECTION_*` 表中，按字段名共用 |
| 实现 | `train/methods/<方法>.py`，登记在 `train/methods/base.py` 的 `IMPLEMENTATIONS` | 每步更新、actor 损失加权、checkpoint、导出 |
| 模型能力 | `core/families/<家族>.py` 的 `ACTOR_CAPABILITIES` | actor 能提供的能力，例如 `sample`（同一观测多次采样） |
| 平台 | `GET /api/methods`（`studio/api/methods.py`） | 方法列表、参数表单、与各家族的兼容性；训练页按它渲染 |

### 声明：`TrainingMethod`

| 字段 | 含义 |
| --- | --- |
| `name`、`label`、`summary` | ID、页面名称和说明（训练页显示为 `label · summary`） |
| `requires_rewards` | 是否需要数据集奖励。为否时平台不查奖励、不跑评价，并拒绝 `reward_*` 参数 |
| `requires_transitions` | 是否需要 (s, a, r, s') 转移；为是时训练使用 `ReplayDataset`，并传入 `transition_horizon` |
| `config_section` | 方法自己的 YAML 段（例如 `iql`、`ifql`）；其他方法的段在有效配置中为空 |
| `critic` | 与 actor 一起训练的价值模型类型（`"expectile"`），没有则为 `None` |
| `critic_keys` | 平台共用 critic 表单中属于本方法的字段（优化器、学习率、expectile、warmup 等）；其他方法的 critic 字段被隐藏并拒绝 |
| `actor_capabilities` | 要求模型家族提供的能力；家族不满足时配置校验拒绝，方法目录也不把该家族列为可选 |
| `exports_critic` | 导出的策略携带 critic 和推理选择器，推理时采样 N 个动作块再选择 |
| `actor_samples_in_training` | critic 更新时从当前 actor 采样下一状态的动作块；回放随转移提供下一状态的 actor 输入 |
| `parameters` | 共享参数之外的表单字段，每项 `{name, label, type, default, min, max, step}`，保存在 `config_section` 中；平台按此渲染 |

训练任务记录方法的 `critic` 和 `requires_rewards`，任务监视器据此显示 Q/V 指标；advantage 权重只在方法报告它时显示。

### 实现：`TrainingAlgorithm`

```python
class TrainingAlgorithm(Protocol):
    def update(self, batch, step) -> tuple[context, metrics]: ...       # critic / 价值更新，返回本步上下文
    def actor_loss(self, prediction, batch, context) -> Tensor: ...
    def weight_actor_losses(self, losses, context) -> Tensor: ...      # 逐样本 BC 损失 → 加权损失
    def checkpoint(self) -> dict: ...
    def restore(self, state: dict) -> None: ...
```

可选钩子，存在时才被调用：

| 钩子 | 用途 |
| --- | --- |
| `critic_batch_keys` | `update` 读取的回放字段，默认 `base.CRITIC_BATCH_KEYS` |
| `distribute(wrap, zero_optimizer)` | 多卡时包装 critic 网络并换成 ZeRO 优化器，返回这些优化器 |
| `export_critic()` | 推理用 critic，随每个 checkpoint 保存为 `critic.pt`（`exports_critic` 的方法） |
| `selector(config)`（静态方法） | 写入导出策略的 `selector` 段 |
| `bind_actor_sampler(sampler)`、`bind_reward_range(low, high)` | 接收 backend 的下一状态动作采样器和训练回放的奖励范围（`actor_samples_in_training` 的方法；`base.bind_training_inputs` 在两个训练器中调用） |

`build_algorithm(config, device)` 按 `IMPLEMENTATIONS` 中的 `"模块:类"` 惰性导入。可复用的部件：

- `train/methods/critics.py`：`ExpectileCritic`（双 Q + V，target 网络，`advantage = min(Q1, Q2) − V`；单卡与多卡的 checkpoint 格式相同）、`ValueFlowCritic`（双回报向量场 `ValueVectorField`）及对应的推理打分类 `ExpectileScorer`、`ValueFlowScorer`，以及 `expectile_loss`、`chunk_bellman_target`、`advantage_weights`。编码器通过 `encoder_factory` 替换，默认 `PixelEncoder`。
- `train/methods/critic_inputs.py`：critic 的图像、本体状态和动作块编码（`encode_chunks` 处理采样得到的动作块），训练回放、推理打分和 VF 的下一状态采样共用同一份实现。
- `train/methods/selectors.py`：`SELECTORS`（名称在 `core.methods.SELECTOR_KINDS` 中声明）、`load_critic`、`choose`。
- `train/methods/base.py`：`critic_batch`、`save_critic`、`critic_artifacts`、`weighted_masked_l1`。

## 带 critic 的导出与推理

`exports_critic` 的方法导出时，`policy.yaml` 额外包含 `critic: critic.pt` 和 `selector: {kind, num_samples}`，critic 文件计入 `component_sha256`；`selector` 也计入平台的模型内容身份。`critic.pt` 只含张量和普通设置，用 `weights_only=True` 加载。

推理：actor 对同一观测采样 `num_samples` 个动作块（π₀.₅ 的图像与提示词前缀只编码一次，`train/models/pi05.py` 的 `sample_chunks`），每块取实际执行的前 `open_loop_steps` 步，经过与环境相同的动作后处理和训练时相同的编码后由 critic 打分，选择器选出要执行的块。三个入口共用这一流程：

| 入口 | 执行步数来源 |
| --- | --- |
| 平台（`studio/policies/pi05.py` → `train/models/pi05_worker.py`） | 本次仿真或测试的 `open_loop_steps` |
| 无头评测（`sim/runtime.py` → `evaluation_policy`） | 评测配置解析出的 `open_loop_steps`，作为 `predict(observation, prompt, executed)` 的第三个参数 |
| 远程推理（`cluster/remote_inference.py`） | 客户端随每次请求发送 `executed` |

单次推理的耗时随 N 增长（前缀一次，去噪 N 份）。critic 只能为不超过其训练转移长度的执行步数打分，`open_loop_steps` 超出时在采样前报错。

## 新增一种方法

| 位置 | 新建或修改 |
| --- | --- |
| `core/methods.py` | 在 `METHODS` 登记，填写上表的声明字段 |
| `core/config.py` | `TRAIN_SCHEMA` 增加方法段；新字段名补进 `_SECTION_*` 取值规则 |
| `configs/methods/<方法>.yaml` | `extends: ../runtime.yaml`、`presets`、`training.method` 和方法段的默认值；只适用于 π₀.₅ 的方法以 `pi05.yaml` 为 preset |
| `train/methods/<方法>.py` | 实现 `TrainingAlgorithm` 和需要的钩子；移植第三方实现时只移入核心逻辑，文件头注明来源仓库和 commit |
| `train/methods/base.py` | `IMPLEMENTATIONS["<方法>"]` |
| 新的 critic 类型 | `train/methods/critics.py` 增加训练与打分两个类，在 `selectors._scorers` 登记打分类；`train/monitoring.py` 的 `_OBJECTIVE_FORMATS` 增加终端显示；任务监视器（`web/src/features/training/JobMonitor.tsx`）需要时增加指标 |
| 新的选择方式 | `core/methods.py` 的 `SELECTOR_KINDS` 与 `train/methods/selectors.py` 的 `SELECTORS` 同时登记 |
| `tests/` | 方法目录与兼容性、配置校验、更新与导出往返、平台参数校验 |
| `docs/`、README 能力矩阵 | 补充说明 |

平台后端的方法列表、默认参数、参数校验和前端训练页都来自登记表，通常不需要修改。
