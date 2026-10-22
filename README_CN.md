# LIBERO-X 仿真与干预控制台

版本：**v0.8.0** · [本版更新汇总](#4-v080-更新汇总)

一个用于机器人仿真、人工接管、数据标注、模型训练和测试的平台。支持 VLA-Adapter、π₀.₅，提供 BC / IQL 训练，以及 RynnValue、Robometer 和人工关键帧评价。

## 1. 项目结构

```text
vla-liberox-workspace/
├── configs/                         # 平台、仿真和控制器配置
│   └── tasks/                       # 任务家族和场景注册
├── tasks/                           # 自备 BDDL 和初始状态
├── liberox-vla-adapter-terminal/
│   ├── frontend/                    # 网页界面
│   ├── backend/                     # 接口、任务队列和数据管理
│   └── scripts/                     # 平台启动、控制器安装与诊断
├── vla-adapter-rynn-iql/
│   ├── configs/models/              # 模型注册与默认配置
│   ├── configs/training/            # BC、IQL 训练配置
│   ├── configs/reward.yaml          # RynnValue 与奖励默认配置
│   └── src/                         # 模型加载、数据处理和训练
├── vla-adapter-robometer/            # Robometer 评价
├── models/                          # 基础模型权重
├── policy-registry/                 # 训练后导出的模型
├── dataset-root/                    # 平台数据库及各项目数据
│   └── projects/libero_x_vla/
│       ├── runs/                    # 运行记录、视频和标注
│       ├── datasets/                # 打包的训练数据集
│       └── training/                # 网页创建的训练结果
├── dataset-exports/                 # 导出的数据集
├── docs/                            # 使用指南、模型接口与启动指令
├── logs/                            # 本地开发记录、调研及测试报告（不提交）
└── third_party/                     # 上游源码及其隔离运行环境（不提交）
    ├── VLA-Adapter/
    ├── LIBERO-X/
    ├── OpenPI/
    ├── RynnValue/
    ├── Robometer/                   # 按需安装
    ├── FACTR_Teleop/
    └── factr-runtime/
```

[快速启动](docs/RUN_COMMANDS.md) · [模型注册](docs/Model.md) · [第三方依赖与旧目录迁移](docs/DEPENDENCIES.md)

网页操作由后端执行，仿真、评价、训练和测试共用一个工作队列。模型权重、采集数据和训练结果不随 Git 上传；`models/` 会保留目录占位。

## 2. 环境配置

以下以 Ubuntu、NVIDIA GPU 和已安装的 Conda 为例。所有命令都在项目根目录执行；需要能访问 GitHub、Hugging Face，π₀.₅ 权重下载还需要访问 Google Cloud Storage。

### 2.1 下载项目

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential pkg-config \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev libhidapi-dev
git lfs install

git clone https://github.com/EclipseaHime017/vla-liberox-workspace.git
cd vla-liberox-workspace

mkdir -p third_party
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git third_party/VLA-Adapter
git -C third_party/VLA-Adapter checkout 23fa0c9c159e2aa04341cdd3e924f44061311060
git clone https://github.com/meituan/LIBERO-X.git third_party/LIBERO-X
git -C third_party/LIBERO-X checkout f528726421c7211d8eb05fe48e9e5e2535ccc813
```

### 2.2 平台与 VLA-Adapter

`vla-liberox` 负责网页后端、仿真、数据处理及 VLA-Adapter 训练。

```bash
conda create -n vla-liberox python=3.10.16 pip -y
conda activate vla-liberox
conda install -c conda-forge 'nodejs>=22.12,<23' -y

python -m pip install -e ./third_party/VLA-Adapter
python -m pip install packaging ninja
python -m pip install -e ./third_party/LIBERO-X --no-deps
python -m pip install -e ./third_party/LIBERO-X/packages/openpi-client
python -m pip install -r liberox-vla-adapter-terminal/requirements-sim.txt
python -m pip install -r liberox-vla-adapter-terminal/requirements-ui.txt
python -m pip install -r vla-adapter-rynn-iql/requirements-train.txt
python -m pip install -e ./vla-adapter-rynn-iql

python -m pip install --upgrade \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 setuptools==69.5.1

git -C third_party/VLA-Adapter apply ../../liberox-vla-adapter-terminal/patches/vla_adapter_hf_local_autoclass.patch

npm --prefix liberox-vla-adapter-terminal/frontend ci
npm --prefix liberox-vla-adapter-terminal/frontend run build
```

这里使用支持 RTX 50 系列的 CUDA 12.8 PyTorch 包，需要匹配的 NVIDIA 驱动。不要再安装 `third_party/LIBERO-X/requirements.txt`，它会替换当前环境的依赖。无需安装 FlashAttention。

上述补丁包含本地 OpenVLA 类注册和 Hub 缓存识别；源码提交、补丁哈希及 TensorFlow 版本统一锁定，环境核验见 [依赖说明](docs/DEPENDENCIES.md#环境一致性检查)。

默认模型为 `VLA-Adapter/LIBERO-Object-Pro`。第一次加载时自动下载到 `models/VLA-Adapter-LIBERO-Object-Pro/`，以后直接复用。模型配置位于 `vla-adapter-rynn-iql/configs/models/vla_adapter.yaml`。

首次运行 LIBERO 若询问是否自定义数据集路径，输入 `N` 即可。

### 2.3 π₀.₅

π₀.₅ 使用独立的 `pi05` 环境。下面的安装脚本会下载 OpenPI 到 `third_party/OpenPI/` 并安装所需依赖。

```bash
conda create -n pi05 python=3.11 pip -y
conda run -n pi05 python -m pip install uv
conda run --no-capture-output -n pi05 \
  python vla-adapter-rynn-iql/scripts/setup_pi05.py
conda run --no-capture-output -n pi05 \
  python vla-adapter-rynn-iql/scripts/prepare_pi05.py
```

完成后，官方 π₀.₅-LIBERO 权重位于 `models/pi05_libero_torch/`，可在平台选择。首次转换需要下载约 12 GiB 原始权重，另生成约 7 GiB 部署权重，并占用较多内存，请预留磁盘与内存空间。

如需 π₀.₅-LIBERO-X，先在 Hugging Face 获得 [Adam-YAN/pi05-liberox-base](https://huggingface.co/Adam-YAN/pi05-liberox-base) 的访问权限，再执行：

```bash
conda run --no-capture-output -n vla-liberox hf auth login
conda run --no-capture-output -n pi05 \
  python vla-adapter-rynn-iql/scripts/prepare_pi05.py --base-model pi05-liberox-base
```

该模型保存到 `models/pi05_liberox_torch/`。两种 π₀.₅ 的配置都在 `vla-adapter-rynn-iql/configs/models/pi05.yaml`。tokenizer 等配套文件按需下载到 `models/.cache/openpi/`，无需手动创建。

### 2.4 RynnValue

需要 RynnValue 轨迹评价时安装；仅使用 BC 训练不需要它。

```bash
git clone https://github.com/alibaba-damo-academy/RynnValue.git third_party/RynnValue
git -C third_party/RynnValue checkout 10e0d333f5f3811d0d130587e50f1faf48da49e5

conda create -n rynnvalue-reward python=3.10 pip -y
conda run -n rynnvalue-reward python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n rynnvalue-reward python -m pip install \
  -r vla-adapter-rynn-iql/requirements-reward.txt
conda run -n rynnvalue-reward \
  python vla-adapter-rynn-iql/scripts/verify_reward_environment.py
```

首次评价自动下载 `Alibaba-DAMO-Academy/RynnValue-4B`。评价默认配置在 `vla-adapter-rynn-iql/configs/reward.yaml`；平台使用的环境名称在 `configs/ui_config.yaml`。不需要执行 `pip install -e ./third_party/RynnValue`。

### 2.5 Robometer

需要 Robometer 进度和成功概率评价时安装。

```bash
git clone https://github.com/robometer/robometer.git third_party/Robometer
git -C third_party/Robometer checkout 352d160389daa964788de1ec933d1925f3a6de4f

conda create -n robometer-reward python=3.10 pip -y
conda run -n robometer-reward python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchao==0.13.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n robometer-reward python -m pip install \
  xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
conda run -n robometer-reward python -m pip install \
  -c vla-adapter-robometer/constraints-robometer.txt -e './third_party/Robometer[robometer]'
conda run -n robometer-reward python -m pip install -e ./vla-adapter-robometer
conda run -n robometer-reward \
  python vla-adapter-robometer/scripts/verify_environment.py
```

首次评价自动下载 `aliangdw/Robometer-4B-LIBERO`。默认配置在 `vla-adapter-robometer/configs/robometer_evaluation.yaml`。请按上述组合安装，不要单独升级 torchao。

### 2.6 人工控制器（可选）

SpaceMouse：

```bash
conda run -n vla-liberox python -m pip install \
  -r liberox-vla-adapter-terminal/requirements-spacemouse.txt
sudo cp liberox-vla-adapter-terminal/udev/70-3dconnexion-spacemouse-wireless.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新插拔设备。自带权限规则适用于对应型号的 SpaceMouse Wireless，其他型号需要按实际 USB ID 调整。

FACTR Franka：

```bash
conda run -n vla-liberox python -m pip install \
  -r liberox-vla-adapter-terminal/requirements-factr.txt
conda run --no-capture-output -n vla-liberox \
  python liberox-vla-adapter-terminal/scripts/setup_factr.py
```

通过 `configs/factr_test_config.yaml` 设置设备。接入后在控制台选择 FACTR 并校准；开关重力补偿时支撑好机械臂，周围留出活动空间。

## 3. 使用

### 3.1 启动平台

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

浏览器打开 **http://127.0.0.1:8000**。前端源码有变化时，启动程序会自动重新构建。

平台地址和数据存放位置通过 `configs/ui_config.yaml` 修改；仿真默认设置在 `configs/config.yaml`。更新配置后重启平台。

### 3.2 仿真与人工接管

在「控制台」点击「创建仿真」，依次选择任务、难度、提示词和模型，然后开始。通过页面设置运行步数、随机种子和初始状态。

已有记录可以按模型家族和具体模型筛选，播放录像、拖动到指定位置，再选择重新推理或人工接管。人工接管前选择 SpaceMouse 或 FACTR，完成校准后启动；SpaceMouse 可以切换世界坐标／工具坐标，并通过滑杆调整灵敏度。接管结果会保存为新记录，原记录不变。

完成的分支可点击「设为主线」后继续操作。续跑可调整总步数，也可以从最后一帧继续；总步数包含已经保存的前缀。策略统一使用所选任务的完整提示词。

通过 `configs/tasks/registry.yaml` 注册任务场景，配置方法见 [任务配置](docs/Task.md)。

当前提供放碗、开抽屉、叠碗及两个长程组合任务，均支持 LEVEL1–4。提示词随官方变体切换，例如入抽屉任务的 LEVEL4 目标是 cream cheese，而不是黄碗。

### 3.3 查看记录与标记关键帧

「运行记录」用于检索历史仿真和接管记录。「数据集」中的轨迹详情可以查看录像、动作、评价曲线和奖励。

在详情中进入「切片」，拖动视频到目标位置，添加 positive 或 negative 标记后保存。通过切片页面选择标注归属：保存为全局标注，或保存到指定数据集；选择会保持，便于连续标注多条轨迹。

### 3.4 评价与打包数据集

在「数据集 → 轨迹评价」选择任务和记录，勾选 RynnValue、Robometer 后点击「评价所选」或「批量评价」。需要重算模型结果时勾选「覆盖已有评价」。

在「打包训练数据集」中选择记录、填写名称，预览后点击「冻结数据集」。模型筛选同时用于检索、批量评价和打包抽样；已打包列表显示包含所选模型记录的数据集。新数据集默认复用轨迹已有的全局评价。

数据集缺少关键帧时会自动继承全局标注，无需逐条点击继承；已有数据集标注（包括已保存的空标注）保持不变。所需奖励或关键帧在全局也不存在时，会列出缺失轨迹并停止相关评价或训练。「验证完整性」在后台检查原始文件，并在任务监视器中列出全局、数据集专属和实际可用的奖励情况。

通过已打包数据集右侧的「配置」可以：

- 选择保留或截断成功后的动作，点击保存即可更新训练取样，不需要重新跑模型评价。
- 配置 RynnValue、Robometer 或 Final Reward，并重新生成对应结果。
- 调整 Stage 与 RynnValue 的奖励组合；只改奖励公式时复用已有模型评价和关键帧。

同一数据集重新评价会更新对应类型的结果，不会删除其他类型。数据集专属评价只对该数据集生效；需要同步修改全局结果时，在配置中开启相应选项。

点击「成员」查看数据集内轨迹及其评价，点击「导出数据集」打包数据和评价结果。迁移到其他设备时复制完整导出目录。

### 3.5 训练模型

在「训练」页面选择数据集、基础模型和训练方法，然后创建训练任务。

- **BC**：直接学习所选数据集中的动作，不需要奖励评价。
- **IQL**：使用数据集奖励训练；先在数据集配置中准备所需评价和 Final Reward。

通过「模型训练配置」选择训练哪些模型组件；通过训练配置和高级参数面板调整训练步数、batch size、梯度累积及其他训练设置。

动作监督长度使用模型 YAML 中的 `io.replay_horizon`，不再固定为 8。数据按采集时的执行区间切分：策略段使用实际预测调用边界，人工段使用该次仿真的执行步数。旧记录缺少执行步数时，使用来源模型的默认值；无法识别来源时明确报错。旧奖励与新切片不一致时，需要在数据集配置中重新生成，关键帧无须重新标记。

旧数据先在 CPU 上重新准备。BC 不需要评价；Sparse、Stage 和不使用 RynnValue 的 Final Reward 只需 CPU 重算。需要 RynnValue 时，关闭强制模型评价：完整评价边界和推理配置不变就复用原输出，仅不兼容或缺失的轨迹需要重新跑模型。历史评价和训练结果保留，不把旧数组插值到新边界。比较新旧训练结果时，应固定推理执行步数、场景和成功规则。

训练进度、日志和任务队列显示在页面下方。训练完成后，导出模型会出现在模型列表中，可直接用于仿真和测试。

需要 W&B 时，先登录，再在训练配置中开启：

```bash
conda run --no-capture-output -n vla-liberox wandb login
```

### 3.6 模型管理与测试

「模型」页面查看基础模型和训练结果。基础模型不显示本机训练记录；训练模型只关联生成它的那一次训练，可查看方法、数据集、步数和种子，并打开或下载当次训练的有效 YAML。「测试」页面选择模型、任务和测试次数，提交后查看成功率与各回合结果。

新增基础权重时，将完整模型文件放到 `models/`，在 `vla-adapter-rynn-iql/configs/models/` 对应家族 YAML 中添加模型，再重启平台。已有家族可以复用加载代码；全新模型结构需要新增适配器。

新模型接入后，保持平台运行，可用以下命令检查仿真、短训练及训练结果加载：

```bash
python liberox-vla-adapter-terminal/scripts/test_model.py --model <模型ID>
```

单模型测试结束后自动清理本次生成的记录、数据集、训练模型和后台任务。

### 3.7 不开网页训练

编辑 `vla-adapter-rynn-iql/configs/terminal_pipeline.yaml`，选择数据、模型和训练方法，然后执行：

```bash
conda activate vla-liberox
python vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml
```

确认终端显示的执行计划后输入 `y` 开始。无人值守运行时加 `--yes`。多卡服务器训练使用独立的 `server` 分支，启动命令见 [快速启动指令](docs/RUN_COMMANDS.md)。

### 3.8 Critic 诊断工具

进入「工具 → IQL Critic 诊断」，选择已有数据集及已生成的奖励、任务和模型，设置 rollout 步数、critic 训练步数及 batch size 后提交。任务进入统一队列，先用所选模型采集独立轨迹，再仅训练 Q/V，不更新或导出 actor。结果中可查看 Matplotlib 曲线，下载逐 chunk 的 Q/V、优势、CSV/NPZ、critic checkpoint 和配置。

训练与验证曲线可对照数据集已有奖励的有限轨迹回报；新 rollout 不生成奖励，只展示 critic 估计。IQL 的 V 是数据动作上的 expectile，不是所选策略的真实回报，Q/V 也不是理想真值。工具产物只保存在其任务目录，删除工具任务即可一并清理，不加入普通运行记录、数据集或模型库。

## 4. v0.8.0 更新汇总

以下汇总 v0.7.2 发布提交之后至本版的 main 更新。BC/IQL、π₀.₅ 接入、统一队列和 Final Reward 已在此前提供，本版重点是完善配置、数据链路和运行可靠性。

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
- 本次发布范围为 main；不包含 dev 自动标注、FQE 或 server 多卡功能。server 的新录制数据兼容同步仍单独处理。
