# 任务配置

任务由场景和完整目标确定。策略推理、奖励评价和训练统一使用所选任务的完整提示词。

## 目录

```text
configs/tasks/
├── registry.yaml         # 家族索引、默认任务 ID
└── my_task.yaml          # 一个任务家族及其场景变体
tasks/
└── my_scene/
    ├── scene.bddl        # 自备场景及成功条件
    └── initial_states.init
```

BDDL 和 `.init` 使用 LIBERO-X 的原有格式，引用已安装的环境、物体和谓词。平台不生成场景或初始状态。`.init` 必须和对应场景的 MuJoCo 状态结构一致。

## 注册任务

在 `configs/tasks/registry.yaml` 的 `families` 中增加家族文件，保留已有条目：

```yaml
schema_version: 1
default_task_id: my-task::scene-a
families:
  my-task: my_task.yaml
```

在 `configs/tasks/my_task.yaml` 填写：

```yaml
schema_version: 1
family_id: my-task
label: place a bowl in the drawer and close it
variants:
  scene-a:
    task_id: my-task::scene-a
    level: LEVEL1
    task_name: my_drawer_scene
    goal_prompt: place the bowl in the drawer and close the drawer
    bddl: ../../tasks/my_scene/scene.bddl
    init: ../../tasks/my_scene/initial_states.init
    optional: false
```

`task_id` 是稳定且全局唯一的场景任务标识，不随显示名称或指令修改而改变。`level` 使用 LEVEL1–4；同一个家族可以配置多个难度和场景。资源路径相对所在 YAML 解析。`optional: true` 的任务缺少资源时显示不可用；默认任务必须可用。

`goal_prompt` 描述完整任务，应与 BDDL 的成功条件一致。环境通过 BDDL 判断成功，模型推理及 RynnValue、Robometer 评价使用该完整目标。

## 使用与检查

保存配置后重启平台：

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

在控制台创建仿真，通过“任务 → 难度 → 提示词”选择新场景；选择初始状态，检查预览后开始。也可以在测试页面对新场景运行小规模测试。

完成的分支可点击“设为主线”，然后从任意已保存帧重新推理或人工接管，包括最后一帧。续跑总步数包含前缀；已到原上限时，需要增大总步数。每次续跑保存新记录，原记录保持不变。

新记录保存任务配置、BDDL 和初始状态的快照。之后修改注册文件不会改变这些历史记录的场景。提升为主线只改变操作角色，不改变人工／策略来源、旧评价或数据集引用。

轨迹按段记录动作出处，训练和奖励均使用完整任务目标。相同血缘保持在同一训练／验证分组，复制前缀按真实动作出处去重。历史记录若使用了不同的自定义指令，会明确拒绝训练或续跑，不改写原始记录。
