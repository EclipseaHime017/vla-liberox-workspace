# 模型

## 1. 配置结构

```text
模型家族（model_name）：结构、接口、输入输出契约、可训练范围、actor 能力
  └─ 权重变体（variant）：来源、权重版本、归一化、部署路径、观测处理、默认执行步数
       └─ 后训练模型：父模型配置快照 + 更新后的权重
```

每个家族只有一份 YAML，`variants` 列出可选的基础权重；后训练模型继承实际使用的父模型配置。

```text
configs/models/
├─ registry.yaml             # 家族索引：model_name → 家族 YAML
├─ vla_adapter.yaml          # 一个家族及其权重变体
└─ pi05.yaml
models/
├─ <variant_ID>/             # 下载或转换后的基础权重与配套文件
└─ .cache/                   # 下载元数据、共享 tokenizer 等
policy-registry/
└─ <导出模型ID>/policy.yaml  # 平台或命令行训练后导出的模型
```

| 文件 | 内容 |
| --- | --- |
| `configs/models/registry.yaml` | 家族索引 |
| `configs/models/<model_name>.yaml` | 家族接口、默认参数及全部 `variants` |
| `configs/methods/<method>.yaml` | 训练方法与运行参数，以 `presets.model` 引用家族 YAML |
| 导出目录的 `policy.yaml` | 后训练模型的父配置、组件文件、训练信息和内容摘要 |

配置优先级：家族默认值 → 选中的变体 → 显式配置。运行时配置的 `model.family` 对应 `model_name`，`model.base_id` 对应变体 ID。模型声明：

- **输入**：家族 `io` 声明相机、图像尺寸、本体状态及维度；各变体的 `observation` 独立配置图像翻转。
- **输出**：原生动作长度与维度、控制频率和动作编码；训练监督长度（`io.replay_horizon`）与推理执行步数分开管理。
- **结构与训练**：结构参数、可训练组件、训练范围；适配器的 `ACTOR_CAPABILITIES` 声明 actor 能力（π₀.₅ 为 `sample`，VLA-Adapter 为空），决定可用的 [训练方法](methods.md)。
- **资产与运行**：权重来源、版本、归一化文件、部署位置和运行环境（`defaults.environment`）。

YAML 描述契约，适配器实现契约。

## 2. 接口与代码位置

| 边界 | 接口 | 位置 |
| --- | --- | --- |
| 家族适配器 | `ADAPTERS`；每个适配器声明 `BACKEND`、`POLICY_ADAPTER`、`ACTOR_CAPABILITIES`、`EXPORT_PREFIX` | `core/families/` |
| 配置发现与校验 | `model_catalog`、`base_model_config`、`model_config`、`model_backend` | `core/model_config.py` |
| 契约与登记 | `model_contract`、`family_adapter`、`family_capabilities` | `core/registry.py` |
| 资产解析 | `pin_model`：解析资产并固定实际版本与内容身份 | `core/model_assets.py`、`core/model_storage.py` |
| 训练组件 | `load_components`、`actor_losses`、`trainable_parameters`、`parameter_counts`、`save_actor`、`restore_actor`、`export_actor` | `train/models/<家族>.py`（即 `BACKEND`） |
| 仿真推理 | `PolicyCatalog`、`PolicyProvider` | `studio/policies/`（`POLICY_ADAPTER` 指向 `<家族>_catalog.py`） |
| 任务快照 | `snapshot`、`validate_snapshot`、`catalog_from_snapshot` | `studio/policies/snapshots.py` |
| 导出与策略契约 | `validate_policy`、`decode_child` | `core/artifacts.py` |

页面列表只读取轻量登记信息；权重加载和完整校验在实际训练或推理时进行。后训练模型保存父模型的 ID、有效配置、资产版本和输入输出契约，加载时使用这份快照。

## 3. 添加从 Hugging Face 下载的权重

**选择家族**：查看模型发布页和架构配置，确认结构、观测输入和动作输出属于哪个家族。同一结构用不同数据训练的权重通常是该家族的一个变体；发布者和仓库名不决定家族。新架构先按第 4 节实现适配器。

**放置权重**：例如变体 ID 为 `my-policy`，把完整的模型快照放到 `models/my-policy/`。保留发布时的文件名与目录结构，包括权重分片、索引、模型配置，以及图像处理器、tokenizer、动作／状态归一化文件和独立组件。

- `checkpoint` 指向**整个部署目录**，不是某个 `.safetensors` 文件。需要格式转换时，先运行该家族的准备工具（π₀.₅ 为 `python -m train prepare-pi05`），再指向转换后的目录。
- 以仓库 ID 登记的权重会直接下载到 `models/<组织>-<仓库>/`；`.model-source.json` 记录来源、revision 和文件摘要，已有模型不会被其他版本覆盖。
- 已转换并验证的原始权重可以清理；tokenizer 等仍被引用的资产必须保留。

**登记变体**：在 `configs/models/<model_name>.yaml` 的 `variants` 下追加：

```yaml
variants:
  my-policy:
    label: My Policy
    source: organization/model-repository
    revision: null
    checkpoint: ../../models/my-policy
    stats_key: my_dataset
    observation:
      image_flip_vertical: true
      image_flip_horizontal: true
    inference:
      open_loop_steps: 5
```

| 字段 | 填写内容 |
| --- | --- |
| `my-policy` | 变体 ID，跨家族唯一，用于平台选择和测试命令 |
| `label` | 页面显示名称 |
| `source` | 原始权重来源，例如 Hugging Face 的 `组织/仓库` |
| `revision` | 原始仓库的完整 40 位 commit；未知时为 `null` |
| `checkpoint` | 本地部署目录（相对本 YAML，`../../models/` 即仓库根目录的 `models/`）；支持在线加载的家族也可填仓库 ID |
| `stats_key` | 归一化数据中的实际标识，按模型发布说明或统计文件填写 |
| `observation.image_flip_vertical`／`image_flip_horizontal` | 相对原始仿真图像是否上下翻转／左右镜像，必填；两项都为 `true` 等价于旋转 180° |
| `inference.open_loop_steps` | 默认执行步数，每个变体必填；页面可调整，上限为模型的预测长度 |

变体继承家族的输入输出、结构、运行环境和默认训练范围，可通过 `io`、`architecture` 声明支持的结构差异。两路相机在训练和推理中都使用所选变体自己的翻转配置，后训练模型继承该配置。

保存后重启平台，模型列表显示 `label`。在训练 YAML 中选择同一变体：

```yaml
model:
  family: <model_name>
  base_id: my-policy
```

## 4. 添加新家族

1. 在 `core/families/` 新建适配器模块，声明契约校验、默认参数、训练面板字段、资产身份、导出格式和 `ACTOR_CAPABILITIES`，并在 `core/families/__init__.py` 的 `ADAPTERS` 中登记。家族名称与适配器 ID 分开，一个适配器可以服务多个家族配置。
2. 适配器的 `BACKEND` 指向 `train/models/<家族>.py`（组件加载、输入转换、训练损失、保存恢复）；`POLICY_ADAPTER` 指向 `studio/policies/<家族>_catalog.py`（推理 Provider、导出解析、任务快照资产）。发现配置时不加载这些实现。
3. 新建 `configs/models/<model_name>.yaml`：`schema_version: 1`、`model_name`、`label`、`adapter`、`default_variant`、`backbone_modes`、`defaults`、`io`、`architecture`、`variants`；每个变体填写自己的 `observation` 和 `inference.open_loop_steps`。
4. 把家族文件加入 `configs/models/registry.yaml`（键与 `model_name` 一致）。上游源码放在 `third_party/` 并登记到 `third_party/lock.yaml`，安装单元加到 `envs/roles.yaml` 的 `families`。

已有家族增加或重命名权重只修改家族 YAML，不增加型号专用的判断。模型列表、训练面板、任务快照和推理入口都读取适配器声明。

## 5. 测试新模型

平台运行时，在另一个终端执行（ID 换成刚登记的变体）：

```bash
conda activate vla-liberox
python -m studio check-model --model my-policy
```

依次检查仿真推理、短训练和导出模型加载，结束后自动清理本次产生的记录、数据集、训练模型和后台任务。
