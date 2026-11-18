# 架构与代码位置

## 1. 顶层目录

仓库根目录下的六个目录就是六个 Python 包。其中 `studio`、`train`、`rewards`、`cluster` 提供命令，入口为 `python -m <包> <命令>`；`core`、`sim` 只作为库被导入。

```text
vla-liberox-workspace/
├─ core/       共享基础：IO 与哈希、配置组合、目录布局、模型登记与契约、方法目录、上游锁（不导入 torch）
├─ sim/        LIBERO-X 仿真：建环境、控制、录制、子目标与成功判定、无头 rollout
├─ rewards/    奖励：sparse/PBRS、Stage 关键帧、Final Reward 融合、版本与缓存身份；RynnValue、Robometer 接入
├─ train/      训练：模型家族实现、方法（BC/IQL）、回放数据、单卡/多卡训练、导出
├─ cluster/    服务器：多卡训练入口、队列、PBS、闭环评测、远程推理、模型传输
├─ studio/     平台后端（FastAPI）：接口、任务队列、数据管理、策略推理、接管设备
├─ web/        平台前端（React + Vite）
├─ configs/    studio/ devices/ tasks/ models/ methods/ rewards/ cluster/，以及 runtime.yaml 等
├─ envs/       按设备用途的环境定义（roles.yaml、layers/、setup.py）
├─ third_party/  上游锁 lock.yaml 与补丁；上游检出不提交
├─ tests/      core/ sim/ rewards/ train/ cluster/ studio/（前端测试在 web/ 内）
├─ skills/     agent 技能：deploy（部署流程与只读的 preflight 检查）；.claude/skills/ 链接到这里
└─ docs/
```

运行数据不提交，目录名与 0.8.0 相同：`dataset-root/`（平台数据库与录制数据）、`models/`（基础权重）、`policy-registry/`（导出的策略）、`evaluator/`、`outputs/`、`logs/`。本地实验脚本放在 `experiments/`，同样不提交。

## 2. 依赖方向

```text
core  ←  sim, rewards  ←  train  ←  cluster, studio
```

每个包只能导入自身和更低层；同层互不依赖（`sim` 与 `rewards`、`cluster` 与 `studio`）。`tests/core/test_layout.py` 用 AST 扫描所有 import 检查这一规则，同时检查旧名称不再出现、六个包都从本仓库导入、没有与包同名的脚本。

`core` 不导入 torch，供平台进程、奖励环境和集群调度共用。需要 GPU 的实现都在 `train/models/`、`rewards/rynnvalue/`、`rewards/robometer/` 中，按需导入。

## 3. 三处主要代码

| 部分 | 位置 | 内容 |
| --- | --- | --- |
| 后端 | `studio/` | `main.py` 组装应用；`api/` 路由；`services/` 任务队列、数据集、评价与训练任务；`policies/` 平台推理；`devices/` SpaceMouse、FACTR；`recording/`、`storage/` 录制与存储；`workers/` 后台任务进程；`commands/` 命令行入口 |
| 前端 | `web/` | `src/pages/` 页面；`src/features/` 按功能的组件与接口（run-control、dataset、training、evaluation 等）；`src/components/` 通用组件；`src/api/` 请求与 WebSocket |
| 训练 | `train/` | `models/` 家族实现（VLA-Adapter、π₀.₅）；`methods/` 方法接口、BC、IQL、critic、选择器；`replay.py` 回放数据；`trainer.py` 单卡训练；`distributed.py` 多卡；`export.py` 导出；`pipeline.py` 终端流水线 |

与训练配合的还有 `rewards/`（奖励）、`sim/`（仿真与评测 rollout）和 `cluster/`（服务器调度）。

## 4. 数据流

```text
studio 仿真 / 人工接管 (sim)
  → dataset-root/projects/<项目>/runs/          录制：图像、动作、执行区间、接管边界
  → 冻结数据集 datasets/<ID>/                    选择轨迹、成功规则、训练/验证划分
  → 评价 (rewards)                               RynnValue / Robometer 模型输出，Stage 关键帧
  → 物化奖励 (rewards.manifest)                  sparse、PBRS、Stage、Final Reward，按内容指纹缓存
  → 训练 (train)                                 prepare-dataset → 回放 → BC / IQL → checkpoint
  → 导出 policy-registry/<ID>/policy.yaml        父模型快照 + 组件权重 + 训练信息
  → 测试 (studio 测试页 / cluster evaluate)      LIBERO-X 闭环，连续成功判定
```

平台进程运行在 vla-liberox 环境，需要其他环境时以 `conda run -n <环境> python -m <模块>` 启动子进程，工作目录为仓库根目录：RynnValue 评价在 rynnvalue-reward，Robometer 在 robometer-reward，π₀.₅ 推理和训练在 pi05。环境名在 `configs/studio/ui.yaml` 和模型家族 YAML 的 `environment` 中配置。

## 5. 配置

| 目录 | 内容 |
| --- | --- |
| `configs/studio/` | `ui.yaml`（地址、端口、数据目录、各环境名）、`studio.yaml`（仿真默认值） |
| `configs/devices/` | `spacemouse.yaml`、`factr.yaml` |
| `configs/tasks/` | 任务登记，见 [任务](tasks.md) |
| `configs/models/` | 模型家族与权重变体，见 [模型](models.md) |
| `configs/methods/` | `iql.yaml`、`bc.yaml`：训练方法与默认参数，引用模型和奖励配置 |
| `configs/rewards/` | `reward.yaml`（RynnValue 与 Final Reward）、`robometer.yaml` |
| `configs/cluster/` | `pipeline.yaml`、`evaluation.yaml`、`pbs.yaml`；`local/` 放本机部署配置，不提交 |
| `configs/runtime.yaml`、`inference.yaml`、`pipeline.yaml` | 共享训练设置与数据、输出、上游路径；推理设置；终端训练流水线 |

YAML 中的相对路径都以该文件所在目录为起点解析。训练配置的组合方式：方法 YAML 以 `extends: ../runtime.yaml` 继承共享设置，以 `presets` 引用模型家族和奖励 YAML，最后叠加平台或命令行的显式覆盖。重复键、未知字段和不支持的组合（例如方法需要的 actor 能力该模型家族不提供）会被拒绝。
