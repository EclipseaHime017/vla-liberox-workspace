# 更新记录

## 0.8.5

本版不改变仿真、训练算法和网页的使用方式，重点是把仓库整理为按职责划分的顶层包，并为新增训练方法和按设备安装做准备。

### 一条主干

- 原 `server` 分支（多卡训练、PBS、闭环评测、远程推理、模型传输）以真正的合并并入 `main`，保留原提交的作者和日期；之后只维护 `main`。

### 仓库结构

- 顶层目录即 Python 包：`core/ sim/ rewards/ train/ cluster/ studio/`，前端为 `web/`；`studio`、`train`、`rewards`、`cluster` 的命令入口统一为 `python -m <包> <命令>`，`core`、`sim` 只作为库。依赖方向 `core ← sim, rewards ← train ← cluster, studio` 由测试检查。
- 根目录一个 `pyproject.toml`，替代原来三个子项目。
- 配置集中到 `configs/`，按 studio、devices、tasks、models、methods、rewards、cluster 分组。
- RynnValue 与 Robometer 作为同级的第三方奖励模型：接入代码在 `rewards/rynnvalue/`、`rewards/robometer/`，关键帧稠密奖励在 `rewards/stage/`，PBRS 公式在 `rewards/pbrs.py`。
- 一次性实验工具移到本地的 `experiments/`，不再提交。

### 训练方法接口

- 方法登记表 `core/methods.py`：每种方法声明是否需要奖励、critic 类型、所需的 actor 能力、是否导出 critic 和额外参数。平台后端、配置校验和训练循环都读取登记表，不再写死 `iql`／`bc`。
- `train/methods/` 拆为 `base.py`（接口与登记）、`bc.py`、`iql.py`、`critics.py`（`ExpectileCritic`，编码器可替换）、`selectors.py`（推理时动作选择）。
- 模型家族声明 actor 能力（π₀.₅：`sample`；VLA-Adapter：无），方法与家族不兼容时配置校验拒绝。
- 新增 `GET /api/methods`，训练页的方法列表和额外参数表单由它提供；选项文字不变。训练任务记录方法的 `critic` 与 `requires_rewards`。
- 新增方法（如 IDQL）的步骤见 [docs/methods.md](docs/methods.md)。

### 按设备用途安装

- `envs/roles.yaml` 定义 dev、studio、teleop、reward-rynnvalue、reward-robometer、train、hpc、eval、infer 九种角色；`python envs/setup.py --role <角色> [--family <家族>] [--dry-run]` 只安装该角色需要的环境、上游和权重。
- 本仓库只安装进 vla-liberox、pi05；rynnvalue-reward、robometer-reward 不安装本仓库，由主环境在仓库根目录下启动。
- 上游锁合并为 `third_party/lock.yaml`，LIBERO-X、VLA-Adapter、OpenPI、RynnValue、Robometer、FACTR 为同级条目；读取接口为 `core.upstream.upstream_lock`。
- 删除未使用的补丁 `vla_adapter_liberox_registry.patch`。

### 文档

- 重写 README（英文精简版）与 README_CN，`docs/` 改为 install、architecture、studio、training、methods、models、tasks、rewards、evaluation、cluster、data。
- 新增 `docs/agent.md`：给 AI agent 的代码地图、验证命令、常见改动的落点和必须保持的约定。
- 新增技能 `skills/deploy`：部署、升级和环境核对的流程，以及只读检查 `python skills/deploy/scripts/preflight.py --role <角色>`；Claude Code 通过 `.claude/skills/deploy` 发现它。
- 版本号统一由测试核对：`VERSION`、`pyproject.toml`、后端、前端和文档标题必须一致。

### 升级说明

- **数据不需要迁移**：`dataset-root/`、`models/`、`policy-registry/`、`evaluator/`、`third_party/` 的位置不变；已导出的策略、RynnValue 原始输出和数据集清单照常使用。
- **派生奖励会重新物化**：奖励缓存按实现文件的名称和内容建立指纹，文件移动后首次使用会在 CPU 上重新计算，不需要重跑模型评价。
- **替换旧的可编辑安装**：在 vla-liberox、pi05 中卸载 `vla-rynn-iql` 后执行 `pip install --no-deps -e .`；在 robometer-reward 中只卸载 `vla-adapter-robometer`。
- **自定义 YAML 和脚本**：按下表更新路径与命令。

| 0.8.0 | 0.8.5 |
| --- | --- |
| `liberox-vla-adapter-terminal/backend/app/` | `studio/`（`core/config.py` → `studio/config.py`） |
| `liberox-vla-adapter-terminal/frontend/` | `web/` |
| `liberox-vla-adapter-terminal/scripts/run_ui.py` | `python -m studio` |
| `.../scripts/test_model.py`、`test_spacemouse.py`、`test_factr.py`、`setup_factr.py`、`migrate_run_layout.py` | `python -m studio check-model`、`check-spacemouse`、`check-factr`、`setup-factr`、`migrate-run-layout` |
| `.../scripts/simulation_core.py`、`trajectory_utils.py`、`eval_pickplace_direct.py`；后端 `simulators/`、`evaluation/` | `sim/` |
| `liberox-vla-adapter-terminal/requirements-*.txt`、`udev/`、`patches/` | `envs/layers/*.txt`、`envs/udev/`、`third_party/patches/` |
| `configs/config.yaml`、`configs/ui_config.yaml` | `configs/studio/studio.yaml`、`configs/studio/ui.yaml`（`offline_rl_root` → `workspace_root`） |
| `configs/spacemouse_test_config.yaml`、`configs/factr_test_config.yaml` | `configs/devices/spacemouse.yaml`、`configs/devices/factr.yaml` |
| `vla-adapter-rynn-iql/src/vla_rynn_iql/` | `core/`、`rewards/`、`train/`、`cluster/`、`sim/` |
| `vla-adapter-rynn-iql/configs/models/`、`configs/training/` | `configs/models/`、`configs/methods/` |
| `vla-adapter-rynn-iql/configs/reward.yaml`、`runtime.yaml`、`inference.yaml` | `configs/rewards/reward.yaml`、`configs/runtime.yaml`、`configs/inference.yaml` |
| `vla-adapter-rynn-iql/configs/terminal_pipeline.yaml` | `configs/pipeline.yaml` |
| `.../configs/server_pipeline.yaml`、`server_evaluation.yaml`、`hpc/configs/pbs_pipeline.yaml` | `configs/cluster/pipeline.yaml`、`evaluation.yaml`、`pbs.yaml` |
| `.../configs/dependency-lock.yaml`、`configs/factr_official.lock.json`、Robometer 的依赖锁 | `third_party/lock.yaml` |
| `.../scripts/train_terminal.py`、`train.py`、`prepare_dataset.py` | `python -m train pipeline`、`train`、`prepare-dataset` |
| `.../scripts/export_checkpoint.py`、`export_policy.py`、`setup_pi05.py`、`prepare_pi05.py` | `python -m train export-checkpoint`、`export-policy`、`setup-pi05`、`prepare-pi05` |
| `.../scripts/materialize_rewards.py`、`annotate_rewards.py`、`verify_reward_environment.py` | `python -m rewards materialize`、`rynnvalue-annotate`、`rynnvalue-verify` |
| `vla-adapter-robometer/`（`scripts/evaluate_trajectories.py`、`verify_environment.py`） | `rewards/robometer/`（`python -m rewards robometer-evaluate`、`robometer-verify`），配置 `configs/rewards/robometer.yaml` |
| `.../scripts/train_server.py`、`train_queue.py`、`hpc/train_pbs.py` | `python -m cluster train`、`queue`、`pbs` |
| `.../scripts/evaluate_server.py`、`serve_evaluation.py`、`evaluate_remote.py` | `python -m cluster evaluate`、`serve`、`evaluate-remote` |
| `.../scripts/nightly_transfer_*.py`、`seed_transfer_payload.py` | `python -m cluster transfer-*` |
| `docs/RUN_COMMANDS.md`、`DEPENDENCIES.md`、`Model.md`、`Task.md`、`EVALUATION.md`、`hpc/README.md` | `docs/studio.md`、`install.md`、`models.md`、`tasks.md`、`evaluation.md`、`cluster.md` |

### 验证

- 后端与训练 1855 项通过、49 项跳过；前端 205 项通过，类型检查通过。
- 新旧平台的接口逐项对照：路由一致，响应仅构建信息中的前端目录和 W&B 项目名不同。
- 按角色安装只做了 `--dry-run` 核对；本版未在评测 PC 和服务器上部署。

## 0.8.0

以下汇总 v0.7.2 之后至 0.8.0 的 main 更新。BC/IQL、π₀.₅ 接入、统一队列和 Final Reward 已在此前提供，本版重点是完善配置、数据链路和运行可靠性。

### 新增功能与任务

- 支持将完成的分支设为主线，再从指定位置或末帧继续推理、接管；保留原始前缀、来源及接管边界。
- 任务改为 YAML 注册，统一管理任务家族、难度、提示词、BDDL 和初始状态，并为运行保存任务快照。
- 注册 VLA-Adapter Pro LIBERO-X 基础模型，支持独立配置其观测处理与执行步数。
- 补齐两个组合任务的官方 L2–L4 变体，五类已注册任务均覆盖 L1–L4，共 21 个场景；提示词遵循各变体的实际目标。
- 仿真记录和数据集支持按模型家族、具体模型筛选，同一筛选范围用于检索、批量评价和打包。
- 模型库增加训练摘要和有效 YAML 查看、下载，复制的训练模型也保留来源信息。

### 代码与框架优化

- 统一模型家族适配器，由模型契约定义输入输出、组件和训练接口；相机上下／左右翻转与默认执行步数归属具体 variant，隔离不同模型配置。
- 分离模型预测长度、动作监督长度和采集执行区间：监督长度按模型配置，策略段按实际推理边界切分，人工段沿用录制时的执行步数，不再统一按 8 步处理。
- 统一使用完整任务提示词，删除自定义策略指令及相关界面、配置和采样分组逻辑。
- 第三方源码集中到 `third_party/`，使用文档集中到 `docs/`，开发记录归入忽略的 `logs/`；清理过时报告并完善安装指南。
- 补强上游源码版本、本地加载补丁和预处理依赖校验，避免部署环境悄然偏离配置。

### 性能与稳定性

- 回放图像一次解压后复用帧缓存，使用 mmap 读取和 NPZ 文件头检查，减少重复解码与整包读取。
- 帧缓存增加损坏恢复、逐键文件锁及原子发布；权重摘要增加缓存，避免重复计算大模型文件哈希。
- TensorFlow 预处理限定使用 CPU，减少与策略训练、推理争抢显存。
- 预览和批量测试复用仿真环境；seed 变化时先软复位再恢复状态，复位失败后重建环境。
- 批量测试默认取消实时等待，保持仿真控制频率不变；目录扫描移出控制锁，减少阻塞。

### 正确性与漏洞修复

- VLA-Adapter 后训练与推理共用 JPEG、缩放和中心裁剪处理，并通过逐像素对照测试；翻转配置在训练与仿真中一致应用，LIBERO-X 变体设为仅上下翻转。
- 执行步数上限依据模型原生输出长度，默认值按 variant 管理；训练子模型默认值不超过本次监督长度，避免执行未被本次后训练直接监督的位置。
- 修复 SpaceMouse 接管瞬间的夹爪扰动、工具坐标轴映射和水平姿态下的方向问题。
- 恢复数据集缺失关键帧及奖励的全局继承；完整性检查分别报告全局、数据集专属和实际可用奖励，缺少所需输入时明确报错。
- 统一连续 `done` 成功判定，分类、统计和批量测试使用系统配置；数据集冻结当时的规则，保留历史训练与奖励的可复现性。
- 相机遮挡消融记录不再进入普通训练，覆盖旧记录扫描、冻结数据集检查和独立训练入口。
- 修复模型训练来源误关联：基础模型没有本机训练记录，子模型只跟踪生成自身的唯一训练记录。
- 模型验收脚本自动清理本次创建的记录、数据集、任务和训练模型；测试缓存隔离到临时目录，避免污染真实数据。

### UI 与评测展示

- 轨迹信息、评价数据来源和测试标签整合为紧凑区域，精简关键帧标注说明，为视频和进度条腾出空间。
- 修复仿真页面左侧会话面板无法滚动，模型筛选保持同一行；模型训练记录使用独立卡片，按需展开格式化 YAML。
- 批量测试展示各初始状态的成功率及其平均值；撤销缺乏覆盖率保证的聚类区间，并注明原 Wilson 区间依赖回合独立假设。

### 更新与验证说明

- 旧数据的切块或奖励边界不匹配时需重新准备、生成奖励；人工关键帧保留，仍兼容的模型评价优先复用，不对旧数组插值。
- 增补模型配置隔离、预处理一致性、切块、缓存并发、成功判定、数据继承和任务选择回归测试；21 个场景的 210 个初始状态已完成真实 MuJoCo 加载、恢复和步进检查，不代表策略成功率验收。
- 发布回归：平台 886 项、训练包 467 项、评价包 8 项、前端 190 项通过，另有 23 项跳过；前端类型检查和生产构建通过。
