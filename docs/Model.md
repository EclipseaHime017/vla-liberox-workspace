# 模型模块

## 1. 配置结构

```text
模型家族（model_name）：结构、接口、输入输出契约、可训练范围
  └─ 权重变体（variant）：来源、权重版本、归一化、部署路径
       └─ 后训练模型：父模型配置快照 + 更新后的权重
```

`model_name` 指模型家族本身。每个家族只有一份 YAML，其中的 `variants` 列出可选的基础权重，后训练模型继承实际使用的父模型配置。

工作区中的模型文件分为注册配置、基础权重和训练产物：

```text
vla-liberox-workspace/
├─ vla-adapter-rynn-iql/configs/models/
│  ├─ registry.yaml                # 家族索引
│  └─ <model_name>.yaml            # 一个家族及其权重变体
├─ models/
│  ├─ <variant_ID>/                # 下载或转换后的基础权重与配套文件
│  ├─ VLA-Adapter-LIBERO-Object-Pro/ # 直接下载的普通模型文件
│  └─ .cache/                      # 下载元数据及共享 tokenizer 等配套资产
└─ policy-registry/
   └─ <导出模型ID>/                # 平台训练后生成的模型
      └─ policy.yaml
```

| 文件 | 管理内容 |
| --- | --- |
| `vla-adapter-rynn-iql/configs/models/registry.yaml` | 轻量目录：`model_name → 家族 YAML` |
| `vla-adapter-rynn-iql/configs/models/<model_name>.yaml` | 家族接口、默认参数及该家族的所有 `variants` |
| `vla-adapter-rynn-iql/configs/training/<method>.yaml` | 选择训练方法及运行参数，引用家族 YAML |
| 导出目录的 `policy.yaml` | 后训练模型的父配置、组件文件、训练信息和内容摘要 |

配置优先级为：家族默认值 → 选中的权重变体 → 显式配置。相对路径以声明文件为起点解析；重复键、未知字段和不支持的接口组合会被拒绝。运行时配置的 `model.family` 对应 `model_name`，`model.base_id` 对应 variant ID。

模型负责声明：

- **输入**：相机列表、图像尺寸与方向、本体状态表示及维度。
- **输出**：原生动作长度与维度、平台执行长度、控制频率和动作编码。
- **结构与训练能力**：结构参数、可训练组件、训练范围和适配器。
- **资产与运行**：权重来源、版本、归一化文件、部署位置和运行环境。

适配器完成模型原生格式与平台观测／控制格式之间的转换。YAML 描述契约，适配器实现契约。

## 2. 接口定义

| 边界 | 接口 | 职责 |
| --- | --- | --- |
| 配置发现 | `model_catalog`、`base_model_config` | 提供模型列表、能力和有效配置，不加载权重 |
| 配置校验 | `model_config`、`model_contract` | 校验属性、家族归属及输入输出契约 |
| 资产解析 | `pin_model` | 解析指定资产并固定实际版本与内容身份 |
| 训练组件 | `model_backend`、`load_components` | 返回所选家族实现并加载组件 |
| 训练目标 | `actor_losses`、`trainable_parameters`、`parameter_counts` | 提供逐样本损失、可选预测结果、可训练参数及参数统计 |
| 保存恢复 | `save_actor`、`restore_actor`、`export_actor` | 保存、恢复并导出模型组件 |
| 仿真推理 | `PolicyCatalog`、`PolicyProvider` | 选择具体模型，执行加载、预测、动作转换与卸载 |
| 任务快照 | `snapshot`、`validate_snapshot`、`catalog_from_snapshot` | 固定并重建一次任务实际使用的模型 |

平台任务负责调度、数据和统计；模型模块负责加载、编码、预测及资产校验。数据与奖励模块不参与模型注册。

后训练模型保存父模型的 ID、类型、有效配置、资产版本和输入输出契约，加载时使用该快照。页面列表只读取轻量注册信息，权重加载与完整校验在显式操作时执行。

## 3. 添加从 Hugging Face 下载的模型

### 选择所属家族

查看模型发布页和架构配置，确认模型结构、观测输入及动作输出所属的家族，然后在 `vla-adapter-rynn-iql/configs/models/registry.yaml` 中找到对应的家族 YAML。同一结构针对不同数据训练的权重通常作为该家族的一个 `variant`；发布者和 Hugging Face 仓库名称不决定家族。

已有家族添加权重只修改该家族的 YAML。新架构则先按下一节实现家族适配器。

### 放置权重

例如给新权重取 ID `my-policy`，将下载的完整模型快照放到工作区的 `models/my-policy/`。保留发布时的文件名与目录结构，包括权重分片、索引、模型配置，以及模型所需的图像处理器、tokenizer、动作／状态归一化文件和独立组件。

`checkpoint` 指向**整个部署目录**，而不是某个 `.safetensors` 文件。权重格式需要转换时，先执行该家族的准备工具，再将 `checkpoint` 指向转换后的目录；已有家族的安装和权重准备命令见 [环境配置](../README_CN.md#2-环境配置)。

基础权重统一放在 `models/`；`policy-registry/` 用于平台导出的后训练模型。Git 保留 `models/` 的占位文件，不跟踪下载的大模型文件。以 Hugging Face 仓库 ID 注册的策略模型直接下载到 `models/<组织>-<仓库>/`，不再使用 `blobs/snapshots` 布局；`.model-source.json` 记录来源、revision 和文件摘要，已有模型不被其他版本静默覆盖。OpenPI 的 tokenizer 等配套资产位于 `models/.cache/openpi/`，使用时按需下载，不要求部署设备预先存在该目录。奖励模型的存储独立管理。

下载缓存是磁盘上的模型文件，可直接用于加载；已转换且验证完成的原始权重可以清理，需要重新转换时再下载。tokenizer 等仍被运行时引用的资产必须保留。推理缓存通常是内存或显存中的模型、特征等临时状态，不是另一份磁盘权重。

### 登记权重变体

在 `vla-adapter-rynn-iql/configs/models/<model_name>.yaml` 已有的 `variants` 下追加一项，例如：

```yaml
variants:
  my-policy:
    label: My Policy
    source: organization/model-repository
    revision: null
    checkpoint: ../../../models/my-policy
    stats_key: my_dataset
```

将示例中的仓库和归一化标识替换为该权重的真实值：

| 字段 | 填写内容 |
| --- | --- |
| `my-policy` | 权重变体 ID，跨家族唯一；用于平台选择和测试命令 |
| `label` | 页面显示名称 |
| `source` | 原始权重来源，例如 Hugging Face 的 `组织/仓库` |
| `revision` | 原始仓库的完整 40 位 commit；已知时填写，未知时为 `null` |
| `checkpoint` | 该家族加载器读取的本地部署目录；支持在线加载的家族也可填写仓库 ID |
| `stats_key` | 权重配套归一化数据中的实际标识，按模型发布说明或统计文件填写 |

`../../../models/my-policy` 从家族 YAML 所在的 `configs/models/` 解析，指向工作区根目录下的 `models/my-policy/`。本地权重加载时另行计算文件内容身份，和来源仓库的 commit 分别记录。

变体继承家族的输入输出、架构、运行环境和默认训练范围。家族支持的差异可在变体下通过 `io`、`architecture` 声明。调整结构或动作表示需要相应的适配器实现，单独下载权重文件并不包含平台所需的完整加载逻辑。

保存后重启平台，模型选择列表显示 `label`。使用训练 YAML 时，选择同一个变体：

```yaml
model:
  family: <model_name>
  base_id: my-policy
```

## 4. 添加新家族

1. 在 `vla-adapter-rynn-iql/src/vla_rynn_iql/` 实现家族适配器，提供配置校验、组件加载、输入转换、动作解码、训练目标、可训练范围及保存恢复接口。
2. 在模型模块注册适配器与资产校验、导出解析，并在 `liberox-vla-adapter-terminal/backend/app/policies/` 接入推理 Provider。
3. 新建 `<model_name>.yaml`，填写 `schema_version: 1`、`model_name`、`label`、`adapter`、`default_variant`、`backbone_modes`、`defaults`、`io`、`architecture` 和 `variants`；默认变体必须属于该家族。
4. 将家族文件加入 `registry.yaml`，重启服务并执行单模型训练／仿真测试。

目录格式：

```yaml
schema_version: 1
models:
  family_a: family_a.yaml
  family_b: family_b.yaml
```

目录键与对应文件的 `model_name` 一致。已有家族增加权重只修改家族文件；新增家族才增加目录项和适配器实现。

注册与契约代码位于 `vla-adapter-rynn-iql/src/vla_rynn_iql/{base_models,models,model_assets,model_artifacts}.py`。业务页面和任务调度读取统一接口，家族适配器负责调用模型实现。家族 YAML 的 `defaults.environment` 指定运行环境；上游模型包统一位于 `third_party/`，提供网络结构和运算实现；权重目录保存参数及配套资产。安装与已有目录迁移见 [第三方依赖](DEPENDENCIES.md)。

## 5. 测试新模型

完成注册后，在工作区根目录启动平台：

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

保持平台运行，在另一个终端进入同一工作区并执行：

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/test_model.py --model my-policy
```

`my-policy` 替换为刚添加到家族 YAML 的变体 ID。
