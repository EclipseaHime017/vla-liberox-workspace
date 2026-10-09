# LIBERO-X 仿真与干预控制台

版本：**v0.7.2**

一个用于机器人仿真、人工接管、数据标注、模型训练和测试的平台。支持 VLA-Adapter、π₀.₅，提供 BC / IQL 训练，以及 RynnValue、Robometer 和人工关键帧评价。

## 1. 项目结构

```text
vla-liberox-workspace/
├── configs/                         # 平台、仿真和控制器配置
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
├── VLA-Adapter/                     # 以下为安装时下载的上游代码
├── LIBERO-X/
├── OpenPI/
├── RynnValue/
└── Robometer/
```

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

git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
git -C VLA-Adapter checkout 23fa0c9c159e2aa04341cdd3e924f44061311060
git clone https://github.com/meituan/LIBERO-X.git
git -C LIBERO-X checkout f528726421c7211d8eb05fe48e9e5e2535ccc813
```

### 2.2 平台与 VLA-Adapter

`vla-liberox` 负责网页后端、仿真、数据处理及 VLA-Adapter 训练。

```bash
conda create -n vla-liberox python=3.10.16 pip -y
conda activate vla-liberox
conda install -c conda-forge 'nodejs>=22.12,<23' -y

python -m pip install -e ./VLA-Adapter
python -m pip install packaging ninja
python -m pip install -e ./LIBERO-X --no-deps
python -m pip install -e ./LIBERO-X/packages/openpi-client
python -m pip install -r liberox-vla-adapter-terminal/requirements-sim.txt
python -m pip install -r liberox-vla-adapter-terminal/requirements-ui.txt
python -m pip install -r vla-adapter-rynn-iql/requirements-train.txt
python -m pip install -e ./vla-adapter-rynn-iql

python -m pip install --upgrade \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 setuptools==69.5.1

git -C VLA-Adapter apply ../liberox-vla-adapter-terminal/patches/vla_adapter_hf_local_autoclass.patch

npm --prefix liberox-vla-adapter-terminal/frontend ci
npm --prefix liberox-vla-adapter-terminal/frontend run build
```

这里使用支持 RTX 50 系列的 CUDA 12.8 PyTorch 包，需要匹配的 NVIDIA 驱动。不要再安装 `LIBERO-X/requirements.txt`，它会替换当前环境的依赖。无需安装 FlashAttention。

默认模型为 `VLA-Adapter/LIBERO-Object-Pro`。第一次加载时自动下载到 `models/VLA-Adapter-LIBERO-Object-Pro/`，以后直接复用。模型配置位于 `vla-adapter-rynn-iql/configs/models/vla_adapter.yaml`。

首次运行 LIBERO 若询问是否自定义数据集路径，输入 `N` 即可。

### 2.3 π₀.₅

π₀.₅ 使用独立的 `pi05` 环境。下面的安装脚本会下载 OpenPI 并安装所需依赖。

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
git clone https://github.com/alibaba-damo-academy/RynnValue.git
git -C RynnValue checkout 10e0d333f5f3811d0d130587e50f1faf48da49e5

conda create -n rynnvalue-reward python=3.10 pip -y
conda run -n rynnvalue-reward python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n rynnvalue-reward python -m pip install \
  -r vla-adapter-rynn-iql/requirements-reward.txt
conda run -n rynnvalue-reward \
  python vla-adapter-rynn-iql/scripts/verify_reward_environment.py
```

首次评价自动下载 `Alibaba-DAMO-Academy/RynnValue-4B`。评价默认配置在 `vla-adapter-rynn-iql/configs/reward.yaml`；平台使用的环境名称在 `configs/ui_config.yaml`。不需要执行 `pip install -e ./RynnValue`。

### 2.5 Robometer

需要 Robometer 进度和成功概率评价时安装。

```bash
git clone https://github.com/robometer/robometer.git Robometer
git -C Robometer checkout 352d160389daa964788de1ec933d1925f3a6de4f

conda create -n robometer-reward python=3.10 pip -y
conda run -n robometer-reward python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchao==0.13.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n robometer-reward python -m pip install \
  xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
conda run -n robometer-reward python -m pip install \
  -c vla-adapter-robometer/constraints-robometer.txt -e './Robometer[robometer]'
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

已有记录可以播放录像、拖动到指定位置，再选择重新推理或人工接管。人工接管前选择 SpaceMouse 或 FACTR，完成校准后启动；SpaceMouse 可以切换世界坐标／工具坐标，并通过滑杆调整灵敏度。接管结果会保存为新记录，原记录不变。

当前提供放碗、开抽屉、叠碗及两个长程组合任务。前三类提供 LEVEL1–4 的可用场景，两个长程任务使用 LEVEL1。

### 3.3 查看记录与标记关键帧

「运行记录」用于检索历史仿真和接管记录。「数据集」中的轨迹详情可以查看录像、动作、评价曲线和奖励。

在详情中进入「切片」，拖动视频到目标位置，添加 positive 或 negative 标记后保存。通过切片页面选择标注归属：保存为全局标注，或保存到指定数据集；选择会保持，便于连续标注多条轨迹。

### 3.4 评价与打包数据集

在「数据集 → 轨迹评价」选择任务和记录，勾选 RynnValue、Robometer 后点击「评价所选」或「批量评价」。需要重算模型结果时勾选「覆盖已有评价」。

在「打包训练数据集」中选择记录、填写名称，预览后点击「冻结数据集」。新数据集默认复用轨迹已有的全局评价。

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

训练进度、日志和任务队列显示在页面下方。训练完成后，导出模型会出现在模型列表中，可直接用于仿真和测试。

需要 W&B 时，先登录，再在训练配置中开启：

```bash
conda run --no-capture-output -n vla-liberox wandb login
```

### 3.6 模型管理与测试

「模型」页面查看基础模型和训练结果。「测试」页面选择模型、任务和测试次数，提交后查看成功率与各回合录像。

新增基础权重时，将完整模型文件放到 `models/`，在 `vla-adapter-rynn-iql/configs/models/` 对应家族 YAML 中添加模型，再重启平台。已有家族可以复用加载代码；全新模型结构需要新增适配器。

新模型接入后，保持平台运行，可用以下命令检查仿真、短训练及训练结果加载：

```bash
python liberox-vla-adapter-terminal/scripts/test_model.py --model <模型ID>
```

### 3.7 不开网页训练

编辑 `vla-adapter-rynn-iql/configs/terminal_pipeline.yaml`，选择数据、模型和训练方法，然后执行：

```bash
conda activate vla-liberox
python vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml
```

确认终端显示的执行计划后输入 `y` 开始。无人值守运行时加 `--yes`。多卡服务器训练使用独立的 `server` 分支，启动命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
