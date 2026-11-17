# 训练方法：接口与新增步骤

一种训练方法分四处描述，平台、配置校验和训练循环都只读这些声明，不写死方法名：

| 层 | 位置 | 内容 |
| --- | --- | --- |
| 声明 | `core/methods.py` 的 `METHODS` | 方法需要什么、产出什么（不导入 torch，平台进程直接读取） |
| 实现 | `train/methods/<方法>.py`，登记在 `train/methods/base.py` 的 `IMPLEMENTATIONS` | 每步更新、actor 损失加权、checkpoint |
| 模型能力 | `core/families/<家族>.py` 的 `ACTOR_CAPABILITIES` | actor 能提供的能力，例如 `sample`（同一观测多次采样） |
| 平台 | `studio/api/methods.py`（`GET /api/methods`）、`web/src/pages/TrainingPage.tsx` | 方法列表、额外参数表单、与各家族的兼容性 |

## 声明：`TrainingMethod`

| 字段 | 含义 |
| --- | --- |
| `name`、`label`、`summary` | ID、页面名称和说明（训练页显示为 `label · summary`） |
| `requires_rewards` | 是否需要数据集奖励。为否时平台不查奖励、不跑评价，并拒绝 `reward_*` 参数 |
| `requires_transitions` | 是否需要 (s, a, r, s') 转移；为是时训练构建器传入 `transition_horizon` |
| `config_section` | 方法自己的 YAML 段（例如 `iql`），默认值在 `configs/methods/<方法>.yaml` |
| `critic` | 与 actor 一起训练的价值模型类型（IQL 为 `"expectile"`），没有则为 `None`。为 `None` 时平台隐藏并拒绝 critic 参数 |
| `actor_capabilities` | 要求模型家族提供的能力；家族不满足时配置校验拒绝，方法目录也不把该家族列为可选 |
| `exports_critic` | 导出的策略是否携带 critic 和推理选择器；为是时 `policy.yaml` 必须包含 `critic`、`selector` 段 |
| `parameters` | 共享参数之外的表单字段，每项 `{name, label, type, default, min, max, step}`，保存在 `config_section` 中；平台按此渲染，不需要改前端 |

训练任务记录方法的 `critic` 和 `requires_rewards`，任务监视器据此决定是否显示 Q/V 指标。

## 实现：`TrainingAlgorithm`

```python
class TrainingAlgorithm(Protocol):
    def update(self, batch, step) -> tuple[context, metrics]: ...       # critic / 价值更新，返回本步上下文
    def actor_loss(self, prediction, batch, context) -> Tensor: ...    # 需要预测结果的方法使用
    def weight_actor_losses(self, losses, context) -> Tensor: ...      # 逐样本 BC 损失 → 加权损失
    def checkpoint(self) -> dict: ...
    def restore(self, state: dict) -> None: ...
```

`build_algorithm(config, device)` 按 `IMPLEMENTATIONS` 中的 `"模块:类"` 惰性导入所选实现。可复用的部件：

- `train/methods/critics.py`：`ExpectileCritic`（双 Q + V，target 网络，`advantage = min(Q1, Q2) − V`）、`expectile_loss`、`chunk_bellman_target`、`advantage_weights`。编码器通过 `encoder_factory` 替换，默认 `PixelEncoder`。
- `train/methods/selectors.py`：推理时从 N 个候选动作块中选一个的 `Selector` 接口和 `SELECTORS` 登记表；BC、IQL 不使用选择器。
- `train/methods/base.py`：`weighted_masked_l1` 等公共损失。
- 多卡训练的同步在 `train/methods/distributed.py`。

## 新增一种方法（以 IDQL 为例）

IDQL：critic 复用 IQL 的 expectile Q/V；actor 按普通 BC 训练；推理时对同一观测采样 N 个动作块，用 Q−V 打分后选择。

| 位置 | 新建或修改 |
| --- | --- |
| `core/methods.py` | 在 `METHODS` 登记 `idql`：`requires_rewards=True`、`requires_transitions=True`、`config_section="idql"`、`critic="expectile"`、`actor_capabilities={"sample"}`、`exports_critic=True`，`parameters` 填采样数 N、选择方式和温度 |
| `train/methods/idql.py`（新建） | 实现 `TrainingAlgorithm`：critic 用 `ExpectileCritic`，`weight_actor_losses` 返回等权损失；从官方实现移植核心逻辑，文件头注明来源仓库和 commit |
| `train/methods/base.py` | `IMPLEMENTATIONS["idql"] = "train.methods.idql:ImplicitDiffusionQLearning"` |
| `train/methods/selectors.py` | 增加 IDQL 选择器：按分数取 argmax 或按温度 softmax 采样 |
| `core/families/pi05.py`、`train/models/pi05.py` | π₀.₅ 已声明 `sample`；在模型实现中提供同一观测、不同噪声的批量采样 |
| `train/export.py`、`core/artifacts.py` | 导出 critic 权重和选择器参数，写入 `policy.yaml` 的 `critic`、`selector` 段 |
| `studio/policies/`、`train/models/pi05_worker.py`、`sim/runtime.py`、`cluster/remote_inference.py` | 加载到选择器时，推理改为"采样 N 个 → critic 打分 → 选择" |
| `configs/methods/idql.yaml`（新建） | `extends: ../runtime.yaml`，`training.method: idql`，`idql` 段的默认参数 |
| `tests/train/test_idql.py`、`tests/core/test_methods.py` | 选择器正确性、critic 复用、导出与加载往返、与 VLA-Adapter 组合时被拒绝 |
| `docs/training.md`、README 能力矩阵 | 补充说明 |

后端的方法列表、参数表单和兼容性都来自登记表，通常不需要修改；前端训练页按参数表单渲染，也不需要修改。只有需要专门的展示（例如候选动作的分数）时，才在 `web/src/features/training/` 或 `web/src/features/evaluation/` 增加组件。

分布式 critic 的方法（例如 Value Flows）放在相同位置：`train/methods/vf.py`，在 `train/methods/critics.py` 增加对应 critic，`configs/methods/vf.yaml` 提供默认参数。
