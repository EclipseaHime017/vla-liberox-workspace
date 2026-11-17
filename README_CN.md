# LIBERO-X 仿真、接管与后训练平台

版本 **0.8.5** · [更新记录](CHANGELOG.md) · [English](README.md)

在 LIBERO-X 仿真中运行 VLA 策略，人工接管并录制轨迹，用模型评价和人工关键帧给轨迹打奖励，再用 BC／IQL 后训练策略并做闭环测试。平台在网页中完成全部操作；训练也可以在服务器、多卡或 PBS 集群上进行。

| | 支持 |
| --- | --- |
| 策略模型 | VLA-Adapter（LIBERO-Object-Pro、LIBERO-X）、π₀.₅（LIBERO、LIBERO-X） |
| 训练方法 | BC（等权行为克隆）、IQL（奖励加权后训练）；方法接口已为 IDQL 等采样类方法预留 |
| 奖励 | sparse、RynnValue（PBRS 整形）、Stage（人工关键帧）、Final Reward 融合；Robometer 进度与成功概率评价 |
| 场景 | LIBERO-X 五类任务、LEVEL1–4 共 21 个场景，可用 YAML 登记自备场景 |
| 接管设备 | SpaceMouse、FACTR 力反馈主臂 |
| 运行位置 | 本机网页平台；单机多卡服务器；PBS 集群；本地仿真 + 远程推理 |

## 目录与代码位置

```text
core/      共享基础：配置、目录布局、模型登记、方法目录、上游锁
sim/       LIBERO-X 仿真与评测 rollout
rewards/   奖励：pbrs、stage/、fusion、manifest；rynnvalue/、robometer/
train/     训练：models/（家族实现）、methods/（BC、IQL、critic）、回放、训练循环、导出
cluster/   服务器：多卡训练、队列、PBS、闭环评测、远程推理、模型传输
studio/    平台后端（FastAPI）
web/       平台前端（React + Vite）
configs/   全部配置          envs/  按设备用途的环境          third_party/  上游锁与补丁
tests/     按包分组的测试    docs/  文档                      skills/  agent 技能（deploy）
```

- **后端**：`studio/`；**前端**：`web/`；**训练**：`train/`，配合 `rewards/`、`sim/`、`cluster/`。
- 依赖方向 `core ← sim, rewards ← train ← cluster, studio`，详见 [架构](docs/architecture.md)。
- 运行数据不提交：`dataset-root/`、`models/`、`policy-registry/`、`outputs/` 等，见 [数据](docs/data.md)。

## 安装

每台设备只装它需要的部分：

```bash
git clone https://github.com/EclipseaHime017/vla-liberox-workspace.git
cd vla-liberox-workspace
python envs/setup.py --list                       # 查看角色
python envs/setup.py --role studio --dry-run      # 打印计划，不安装
python envs/setup.py --role studio                # 执行
```

| 角色 | 用途 |
| --- | --- |
| `dev` | 开发与测试，不下载权重 |
| `studio` | 平台机：仿真、采集、网页；`--family pi05` 另装 π₀.₅ |
| `teleop` | SpaceMouse 与 FACTR，附加到 studio |
| `reward-rynnvalue`、`reward-robometer` | 两个奖励模型的独立环境 |
| `train`、`hpc` | 训练服务器、PBS 节点：只装所选家族 |
| `eval`、`infer` | 无头闭环评测、远程推理服务 |

系统依赖、人工步骤（udev、ROS、Hugging Face 授权）和从 0.8.0 升级见 [安装](docs/install.md)。

## 快速开始

**平台**

```bash
conda activate vla-liberox
python -m studio            # http://127.0.0.1:8000
```

在「控制台」创建仿真并人工接管 → 在「数据集」评价、标关键帧、冻结数据集 → 在「训练」选择数据集、模型和方法 → 在「模型」「测试」查看和测试导出的策略。详见 [平台使用](docs/studio.md)。

**终端训练（单卡）**

```bash
python -m train pipeline --dry-run    # 按 configs/pipeline.yaml 检查数据与计划
python -m train pipeline              # 确认后开始：选择数据 → 准备 → 评价 → 训练
```

**服务器与集群**

```bash
python -m cluster train               # 单机多卡，configs/cluster/pipeline.yaml
python -m cluster pbs submit          # PBS，configs/cluster/pbs.yaml
python -m cluster evaluate            # 无头闭环评测，configs/cluster/evaluation.yaml
```

每个包的命令列表：`python -m studio|train|rewards|cluster --help`。

## 文档

| 文档 | 内容 |
| --- | --- |
| [安装](docs/install.md) | 角色、环境层、人工步骤、升级 |
| [架构](docs/architecture.md) | 包与依赖方向、数据流、配置组合 |
| [平台使用](docs/studio.md) | 页面操作、接管设备、平台命令 |
| [训练](docs/training.md) | 网页与终端训练、导出、监控 |
| [训练方法](docs/methods.md) | 方法接口、模型能力、新增方法（IDQL）的步骤 |
| [模型](docs/models.md) | 模型家族与权重变体、添加权重或新家族 |
| [任务](docs/tasks.md) | 登记场景与任务 |
| [奖励](docs/rewards.md) | 奖励定义、评价与物化、全局与数据集结果 |
| [评测](docs/evaluation.md) | 平台测试、无头评测、远程推理 |
| [服务器与集群](docs/cluster.md) | 多卡训练、队列、PBS、NSCC 说明 |
| [数据与存储](docs/data.md) | 目录布局、整理旧目录、导出 |
| [third_party](third_party/README.md) | 上游锁与检出 |
| [Agent 指南](docs/agent.md) | 给 AI agent：代码地图、验证命令、改动落点、必须保持的约定；部署与环境核对用 [skills/deploy](skills/deploy/SKILL.md) |

## 测试

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q tests
conda run -n vla-liberox npm --prefix web test
```
