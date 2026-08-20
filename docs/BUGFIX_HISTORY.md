# Bug 修复记录

本文记录 LIBERO-X × VLA-Adapter 最小测试框架在安装和运行过程中的问题、原因及修复结果。

> 说明：2026-08-14 之前条目中的长命令用于记录当时的旧版 CLI；当前版本请编辑 `configs/config.yaml`，并使用文末的简化命令运行。其余路径保留历史发生时的名称。

## 2026-08-13：3.1 环境验证无法运行

### 问题一：`--env-only` 意外加载模型依赖

运行命令：

```bash
python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py \
  --vla-root ./VLA-Adapter \
  --liberox-root ./LIBERO-X \
  --env-only
```

原始错误：

```text
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

调用链会经过：

```text
run_libero_eval.py
  -> robot_utils.py
  -> prismatic RLDS dataset
  -> dlimp
  -> tensorflow_datasets
  -> tensorflow_metadata
  -> protobuf
```

根因：

- `load_runtime()` 在程序启动时同时导入仿真模块和 VLA 模型模块。
- 即使指定了 `--env-only`，仍会加载 RLDS、TensorFlow 和 protobuf。
- 这与 README 中“3.1 不加载模型权重，只验证仿真环境”的设计不符。

修复：

- 将运行时加载拆分为 `load_runtime()` 和 `load_policy_runtime()`。
- `load_runtime()` 只导入 LIBERO 仿真、PyTorch和图像写入模块。
- 仅在非 `--env-only` 模式下调用 `load_policy_runtime()`。
- 将 `quat2axisangle()` 放入评测脚本，避免为了构造本体状态而导入 VLA 的 `libero_utils.py`。

修改文件：

- `scripts/eval_pickplace_direct.py`

### 问题二：MuJoCo 3.x 与 robosuite 1.4.1 不兼容

修复问题一后，环境创建阶段出现：

```text
AttributeError: 'MjData' object has no attribute 'qM'. Did you mean: 'M'?
```

根因：

- `robosuite==1.4.1` 仍使用 MuJoCo 2.x 的 `MjData.qM` 接口。
- `robosuite` 的依赖只声明 `mujoco>=2.3.0`，因此 pip 自动安装了最新的 `mujoco==3.11.0`。
- MuJoCo 3.x 已改变相关接口，导致控制器初始化失败。

修复：

```bash
pip install "mujoco==2.3.7"
```

同时在以下文件中固定该版本：

- `requirements-sim.txt`
- `README_CN.md`

### 验证结果

修复后重新执行 README 3.1 命令，程序退出码为 0，输出包含：

```text
agentview_image: (256, 256, 3)
robot0_eye_in_hand_image: (256, 256, 3)
proprio: (8,)
dummy_action: (7,)
Environment-only smoke test passed
```

EGL 模式也已通过验证：

```bash
MUJOCO_GL=egl python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py \
  --vla-root ./VLA-Adapter \
  --liberox-root ./LIBERO-X \
  --env-only
```

## 2026-08-13：环境安装兼容性修复

### NumPy 被 OpenCV 升级到 2.x

现象：

```text
tensorflow 2.15.0 requires numpy<2.0.0,>=1.23.5
```

根因：最新版 `opencv-python` 将 NumPy 升级为 2.x，与 TensorFlow 2.15 不兼容。

修复：

```bash
pip install "numpy==1.26.4" "opencv-python==4.11.0.86"
```

### robosuite 导入时 Numba 缓存失败

现象：

```text
RuntimeError: cannot cache function 'mat2quat': no locator available
```

根因：robosuite 1.4.1 默认启用 Numba 文件缓存，但当前 wheel 安装位置无法生成有效缓存定位器。

修复：在 `vla-liberox` 环境的 `robosuite/macros_private.py` 中关闭缓存：

```python
CACHE_NUMBA = False
```

同时将 Numba 固定为兼容版本：

```bash
pip install "numba==0.59.1"
```

### LIBERO-X 缺少 `openpi-client`

LIBERO-X 使用 `--no-deps` 安装是为了避免覆盖 PyTorch，但这也会跳过其本地 `openpi-client` 包。

修复：

```bash
pip install -e ./LIBERO-X/packages/openpi-client
```

## 当前已知限制

### FlashAttention 尚未安装

安装 `flash-attn==2.5.5` 时出现：

```text
flash_attn was requested, but nvcc was not found
OSError: CUDA_HOME environment variable is not set
```

当前主机未发现 CUDA Toolkit 的 `nvcc`。本评测源码没有启用 FlashAttention，RTX 5090 上使用 PyTorch 2.7.0/CUDA 12.8 已在不安装 `flash-attn` 的情况下完成 300 步 rollout；因此它不是 3.1 或 3.2 的阻塞依赖。

### TensorFlow Addons 与 Tyro 的 `typeguard` 约束冲突

最初安装最新版 Tyro 时，`pip check` 报告：

```text
tensorflow-addons 0.23.0 requires typeguard<3.0.0,>=2.7
```

新版 Tyro 需要 Typeguard 4.x，而 TensorFlow Addons 0.23 要求 Typeguard 2.x。已固定为仍提供 `tyro.cli()` 且不依赖 Typeguard 4.x 的组合：

```bash
pip install "tyro==0.8.5" "typeguard==2.13.3"
```

修复后 `pip check` 不再报告该冲突。

### 非致命警告

以下信息不影响 3.1 通过：

- Gym 已停止维护的提示。
- 未安装 `OpenGL_accelerate` 的提示。
- Matplotlib 配置目录不可写时自动使用 `/tmp` 缓存的提示。

## 2026-08-13：3.2 模型 rollout 启动问题

### TensorFlow Metadata 与 protobuf 不兼容

现象：

```text
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

根因：pip 安装了过新的 `tensorflow-metadata==1.21.0`，其生成的 protobuf 代码与 TensorFlow 2.15 使用的 protobuf 版本不兼容。

修复组合：

```bash
pip install \
  "tensorflow-metadata==1.15.0" \
  "wandb==0.16.3" \
  "setuptools==69.5.1"
```

说明：`tensorflow-metadata==1.15.0` 需要 protobuf 3.20.x，因此同时将未锁定的新版 wandb 降为兼容版本；setuptools 69.5.1 用于保留 wandb 0.16.3 所需的 `pkg_resources`。

### 缺少 `msgpack_numpy`

现象：

```text
ModuleNotFoundError: No module named 'msgpack_numpy'
```

根因：VLA-Adapter 的 `robot_utils.py` 使用了该模块，但项目依赖中没有声明。

修复：

```bash
pip install "msgpack-numpy==0.4.8"
```

### Hugging Face checkpoint 被误判为必须执行远程代码

模型配置下载成功后出现：

```text
ValueError: Loading VLA-Adapter/LIBERO-Object-Pro requires you to execute
the configuration file in that repo ... set trust_remote_code=True
```

原因：checkpoint 的 `config.json` 通过 `auto_map` 指向模型仓库中的自定义 Python 实现；同时，VLA-Adapter 只在本地 checkpoint 分支注册本地 `OpenVLA` 类。Hub checkpoint 跳过注册后，Transformers 无法识别 `model_type=openvla`，而加载函数又明确使用 `trust_remote_code=False`，因此必然抛出该错误。

修复：无论 checkpoint 来自本地还是 Hub，都注册当前固定 VLA-Adapter 提交中的四个本地 AutoClass；只有配置文件同步仍限制在本地目录分支。模型与 processor 继续保持 `trust_remote_code=False`，不会下载或执行 Hub Python 文件。

```text
AutoConfig                 -> local OpenVLAConfig
AutoImageProcessor         -> local PrismaticImageProcessor
AutoProcessor              -> local PrismaticProcessor
AutoModelForVision2Seq     -> local OpenVLAForActionPrediction
```

修复文件：

- `VLA-Adapter/experiments/robot/openvla_utils.py`
- `patches/vla_adapter_hf_local_autoclass.patch`

离线使用缓存 `config.json` 验证后，`AutoConfig` 成功解析为本地 `prismatic.extern.hf.configuration_prismatic.OpenVLAConfig`。

### RTX 5090 / `sm_120` 与 PyTorch 2.2 不兼容

现象：

```text
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120
is not compatible with the current PyTorch installation
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

根因：项目固定的 `torch==2.2.0+cu121` 只包含到 `sm_90` 的 CUDA kernel，无法在 Blackwell `sm_120` 上运行。

修复：

```bash
pip install --upgrade \
  "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" \
  --index-url https://download.pytorch.org/whl/cu128
pip install "numpy==1.26.4" "setuptools==69.5.1"
```

GPU 实算验证：

```text
torch 2.7.0+cu128
device NVIDIA GeForce RTX 5090 Laptop GPU
capability (12, 0)
bfloat16 CUDA matmul passed
```

### 单次 rollout 验证结果

README 3.2 命令已完成一次完整回合。策略没有完成任务，但推理和仿真链路无异常，符合 README 中“完整 rollout 即接口链路跑通”的验收标准：

```text
steps: 300
policy_queries: 38
error: null
resolved_stats_key: libero_object_no_noops
video_frames: 300
```

生成文件：

- `runs/liberox_pickplace/episode_000_failure.mp4`
- `runs/liberox_pickplace/results.jsonl`
- `runs/liberox_pickplace/summary.json`

说明：RTX 5090 所需的 PyTorch 2.7.0 是对项目 `pyproject.toml` 中 `torch==2.2.0` 的有意硬件兼容覆盖，因此 `pip check` 仍会报告 torch、torchvision、torchaudio 三条项目元数据版本不一致；GPU 实算、模型加载和完整 rollout 均已通过。

## 2026-08-14：评测参数迁移到 YAML

原先 3.1、3.2 和 3.3 都需要在命令行重复传入仓库路径、checkpoint、任务、随机种子和输出目录。长命令容易遗漏参数，也不利于保留每组实验的完整配置。

调整后：

- 新增根目录 `config.yaml`，集中保存原有评测参数；
- 脚本强制读取与 README 同目录的 `config.yaml`，不再接受配置地址或其他运行参数；
- `vla_root`、`liberox_root` 和 `output` 相对配置文件目录解析，不再依赖启动目录；
- Hugging Face checkpoint ID 保持原字符串，显式本地路径才按配置文件目录解析；
- `cuda_visible_devices` 和 `mujoco_gl` 在导入 PyTorch、MuJoCo 前应用；
- 使用 PyYAML SafeLoader，并拒绝重复键、未知键、缺失键、错误类型和越界数值；
- `requirements-sim.txt` 显式固定 `PyYAML==6.0.3`。

默认配置对应 3.3 的 10 回合评测，运行命令简化为：

```bash
python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py
```

## 2026-08-14：轨迹记录、状态回溯与人工接管

新增内容：

- `scripts/trajectory_utils.py` 定义版本化轨迹格式；
- 评测过程中逐状态保存末端 XYZ、axis-angle、四元数、双指夹爪 qpos 和完整 MuJoCo state；
- 逐动作保存 VLA 原始动作、送入 LIBERO 的实际动作、reward、done 与 `action_source`；
- 双相机观测单独保存为压缩 NPZ，同时生成可读 CSV、元数据 JSON 和 6DoF/夹爪曲线 PNG；
- `scripts/intervene_pickplace.py` 可从 `resume_step` 或 `resume_time_seconds` 恢复历史状态；
- 干预后支持重新查询策略，以及 stdin、JSONL、UDP 三种人工/外部控制方式；
- 新分支保留原轨迹前缀，并把接管后的数据合并为新的完整示范；
- `intervention_config.yaml` 集中保存回溯位置、控制模式和输出参数，命令行不接受其他配置地址。

验证结果：

- 核心轨迹、双相机文件、CSV、JSON 和 PNG 完成无 pickle 往返测试；
- LIBERO-X 环境执行动作后恢复保存的 MuJoCo state，最大 state 误差为 `1.11e-16`，末端位置误差为 `0`；
- 干预输出使用带微秒时间戳的新目录，不覆盖之前的演示数据。

### 固定总长度与 VLA action 对比

- 删除 `max_steps_after_resume` 和成功后提前终止选项；干预步数自动取 `source_action_count - resume_step`，使合并轨迹与源轨迹具有相同总步数；
- 轨迹 schema 升级到 v2，同时保持读取 v1 文件的兼容性；
- 新增 `inference_query_step`、`inference_chunk_offset`、`inference_action`，保存每次 VLA 查询返回的完整 action chunk；
- 新增 `trajectory_*_inference.csv`，用可读表格记录 query step、chunk index、目标时间和 7 维 VLA action；
- `raw_action` 继续保存真正被选择并执行的 VLA action，`env_action` 保存夹爪归一化/符号恢复后的环境动作；
- 每次评测新增 7 张独立的 `trajectory_*_action_{dx,dy,dz,drx,dry,drz,gripper}.png`；
- 每次干预新增 7 张 `trajectory_action_comparison_{dx,dy,dz,drx,dry,drz,gripper}.png`：原始动作保留 frame 0 到结尾，分支动作仅从回溯帧开始叠加；
- action 图坐标轴明确标注控制帧 `[frame]`、仿真时间 `[s]` 和归一化控制量 `[-]`；
- 按最终需求取消 action 跳变断线和相关 YAML 参数，所有相邻帧保持连续连线。
- 干预结果只生成 7 张原始/分支 action 对比图，不再额外生成二次推理自身的 7 张 action 图和单独轨迹图；人工控制使用的 `latest_frame.png` 在结束后自动删除。
- 将 `intervention_config.yaml` 的 `policy_open_loop_steps` 统一更名为 `open_loop_steps`，与 `config.yaml` 使用相同名称；评测和干预 summary 也统一记录该字段。
- 将 VLA 观测分辨率与回放视频解耦：模型继续使用 `256×256` 的 agentview/腕部图像，视频新增 `video_camera`、`video_width`、`video_height`、`video_fps` 配置，默认使用更完整的 `frontview 640×480`；干预视频的历史前缀按保存的 MuJoCo state 用相同相机重新渲染。

### VLA 双相机拼接录像

- 新增虚拟录像模式 `video_camera: vla_views`，水平拼接 VLA 实际使用的两路观测：左侧 `agentview`，右侧 `robot0_eye_in_hand`；
- 默认输出尺寸改为 `1024×512`，并要求宽度等于高度的两倍，避免两幅正方形画面被拉伸；
- 拼接与放大只发生在录像路径，不改变策略的 `256×256` 原始观测和后续 `224×224` 模型输入；
- 原始评测视频、干预视频、回溯历史前缀和人工控制临时画面统一使用同一个拼接函数；
- 保留 `frontview`、`birdview`、`sideview` 和 `galleryview` 单相机模式，便于按需切换。

### 同推理视角的高清主相机录像

- 原始评测和 intervention 均改为同步输出两份视频：VLA 双视角拼接视频，以及单独的 `agentview` 高清视频；
- 高清视频直接调用 MuJoCo 对 `agentview` 相机重新渲染，默认 `1024×1024`。在相同分辨率下已验证其像素与策略主相机观测完全一致，只有输出分辨率不同；
- 不再将低角度 `frontview` 作为默认高清录像，避免机械臂遮挡桌面目标，同时不修改 checkpoint 所依赖的策略相机位姿；
- 新增 `main_view_video_width`、`main_view_video_height`，要求保持正方形以维持推理视角的宽高比；
- 两份视频改为同步流式编码，避免 300 帧双高清视频同时驻留内存；
- intervention 输出文件为 `intervention.mp4` 和 `intervention_agentview_hd.mp4`，summary 分别记录 `video` 与 `main_view_video`。

### 非 headless 实时窗口

- `config.yaml` 新增统一的 `headless` 布尔开关，原始评测与 intervention 共用；
- `headless: false` 时使用 LIBERO `ControlEnv`，同时启用窗口 renderer 和离屏 renderer，因此实时观察、双视角拼接录像与高清主视角录像可以并行工作；
- 实时窗口固定使用与推理主图一致的 `agentview`，尺寸沿用 `main_view_video_width`、`main_view_video_height`，默认 `1024×1024`；
- 原始评测在初始状态和每个控制步后刷新窗口，intervention 从恢复点开始刷新新分支，不快速播放历史前缀；
- Linux 下缺少 `DISPLAY` 和 `WAYLAND_DISPLAY` 时提前报错，避免 OpenCV/Qt 在创建窗口时无提示退出；
- 已分别验证 `OffScreenRenderEnv` headless 路径和 `ControlEnv agentview 1024×1024` 实际窗口路径。

### 拼接视频保持策略观测原分辨率

- `vla_views` 从 `1024×512` 显示放大改为 `512×256` 原像素拼接；
- 左右两幅图分别直接取 `256×256` 的 `agentview_image` 与 `robot0_eye_in_hand_image`，只执行与 VLA-Adapter 训练预处理相同的 180° 旋转，不做插值缩放或高清重渲染；
- 单独的 `*_agentview_hd.mp4` 继续使用 MuJoCo `1024×1024` 高清重渲染，两类视频职责明确分离；
- `512×256` 拼接帧与进入 `prepare_observation` 前的两路策略相机数据一致，模型内部后续的 `224×224` resize/crop 不变。

### 标准 episode 视频改为 VLA 外部视角

- 修复旧 `episode_000_failure.mp4` 仍指向低位 `frontview`，容易被误认为新视角未生效的问题；
- 标准文件 `episode_000_{success|failure}.mp4` 现在固定保存 `1024×1024` 的 `agentview` 高清重渲染，与 VLA 外部相机具有相同位姿和高度；
- 原始 VLA 双相机拼接改为显式后缀 `episode_000_{success|failure}_vla_views.mp4`，保持 `512×256` 且不做缩放；
- `results.jsonl` 中 `video`/`main_view_video` 指向标准高清文件，`vla_views_video` 指向拼接文件，并分别记录两类视频的相机和分辨率；
- 同时修复 `PairedVideoWriter` 调用多传运行参数的问题，改为具名参数以防再次错位；录像初始化也纳入 episode 的 `try/finally`，即使初始化失败仍会主动关闭 MuJoCo 环境，避免随后出现误导性的 EGL 析构报错。

### 使用 MuJoCo 原生 Viewer 实时观察

- `headless: false` 不再使用 robosuite 的 `OpenCVRenderer` 展示固定相机帧；
- 改为通过 `mujoco.viewer.launch_passive()` 直接共享 LIBERO 仿真正在使用的原生 `MjModel/MjData`，可在窗口中自由旋转、平移、缩放并使用 MuJoCo 可视化面板；
- Viewer 只同步状态、不自行执行物理步，VLA 推理和 `env.step()` 仍是仿真状态的唯一推进方；
- VLA 的 `agentview`/腕部离屏输入、双相机拼接录像和高清 `agentview` 录像继续使用原有固定摄像头，不受 Viewer 自由相机影响；
- 关闭 Viewer 后 rollout 会继续；退出时先等待 Viewer 释放共享资源，再关闭 LIBERO 环境，避免 EGL/GLFW 析构报错。

### 20 Hz 墙钟节拍、视频方向与 Viewer 视觉网格

- `config.yaml` 新增显式 `control_hz: 20` 和 `realtime_control: true`，LIBERO 每个 action 对应 50 ms 仿真时间，墙钟限速保证 `env.step()` 不会快于 20 Hz；耗时操作导致落后时不突发追帧；
- 视频仍每个控制步保存一帧，`video_fps` 从 10 改为并强制等于 20，使 300 步轨迹从错误的 30 秒播放时长恢复为与仿真一致的 15 秒；intervention 删除重复的 `video_fps`，统一继承 `config.yaml`；
- 保留 VLA 双相机拼接的官方 180° 训练预处理，避免改变 checkpoint 输入；供人查看的高清 `agentview` 改为只做 OpenGL 上下翻转，不再与 MuJoCo 自然方向左右相反；
- 原生 Viewer 初始固定到 `agentview`，需要自由观察时可在 Camera 面板选择 `Free`；
- Viewer 默认隐藏 robosuite collision geom group 0、显示 visual geom group 1，修复机械臂碰撞体与视觉网格叠加造成的绿色、黄色和斑点渲染问题。
- 将相机观测改为 VLA 查询时按需采集，并在 rollout 后根据已保存 state 重建完整观测与两份视频；移除控制环中的 `1024×1024` 渲染/编码后，无模型稳态实测步间隔为约 `50.1 ms`（约 `19.97 Hz`）；
- 首个 OSC action 原本会承担约 2.8 秒的一次性控制器初始化开销。验证发现“在正式环境中隐藏执行再回滚”会污染控制器内部目标，因此改为在独立临时环境中预热、销毁后再创建正式环境；实测正式环境的无模型步间隔为约 `50.07 ms`（约 `19.97 Hz`），且正式 episode 初始状态不受预热 action 影响；
- `results.jsonl`、轨迹 metadata 与 intervention summary 新增实际控制频率和周期最小/平均/最大值；直接评测的 `summary.json` 同时汇总平均实测频率、最大周期和发生 deadline miss 的回合数。

### 代码审查与冗余清理

- 删除未被任何入口调用的旧 `replay_frame_from_observation()`，统一复用 `trajectory_utils.quaternion_to_axis_angle()`，不再在评测脚本中维护第二份四元数转换实现；
- 删除 `validate_observation()` 与 Viewer 刷新函数未使用的参数，以及配置加载后重复执行的 `trials`/`open_loop_steps` 范围检查；
- intervention 不再解压、复制随后必定被 state 重渲染覆盖的原双相机观测数组。源轨迹存在观测 companion 文件时，仍会为新分支完整重建并保存观测，输出语义不变；
- 人工控制的 `latest_frame.png` 改为恢复点生成一次、每个新 state 更新一次，移除相邻循环对同一帧的重复编码；
- 将旧 `results.jsonl` 的清理移动到模型加载和仿真预热成功之后，避免初始化失败时提前破坏上一轮结果；评测出现真正的 episode 异常时写入 `episode_errors` 并返回非零退出码，普通任务失败仍视为一次成功完成的评测流程。
- 从仿真核心依赖 `requirements-sim.txt` 移除只被 LeRobot Parquet 检查器使用的 `pyarrow`；README 在数据检查步骤单独安装，现成 checkpoint 评测不再承担无关依赖。

## 2026-08-14：仿真、回溯与人工接管 Web UI

- 新增 `scripts/simulation_core.py`，原始评测、CLI intervention 和 Web UI 统一使用同一 action chunk、限速、停止边界和轨迹记录循环；
- 新增 `liberox_ui/` FastAPI 后端，会话状态机覆盖 `LOADING/RUNNING/STOPPING/POSTPROCESSING/COMPLETED/ERROR`，同一时刻只允许一个活动仿真；
- 新增可复用的 `VLAAdapterPolicyProvider` 和固定任务 `TaskCatalog`，模型延迟加载并跨会话复用，bootstrap 为后续模型/任务切换保留能力字段；
- 新增独立只读 MuJoCo 预览线程，只消费最新 state 并渲染 `agentview 512×512 / 10 fps`，不把浏览器渲染放入 20 Hz 控制线程；
- 新增完成后逐帧状态查看、恢复误差 `1e-9` 校验、一级策略/人工分支和多并列分支，分支保持源轨迹实际总长度；
- 新增浏览器键盘/按钮 OSC_POSE 控制、双增益、100 ms 心跳和 250 ms deadman；WebSocket 断开会停止人工分支并自动保存；
- 新增 `ui_config.yaml`、唯一 session 结果目录、历史 trajectory 递归索引、安全 artifact 白名单下载和原子 manifest/summary/results 发布；
- 轨迹、观测、图表和分支比较图先写入隐藏 staging 目录，再逐文件原子发布；两份视频同样以 pending 文件完成编码后改名；
- 新增 React + TypeScript 控制台，生产构建由 FastAPI 静态托管；锁定 npm lockfile，升级到已修复审计问题的 Vite `7.3.6` / Vitest `3.2.7`，`npm audit` 为 0；
- 新增 `scripts/run_ui.py`，固定监听 `127.0.0.1:8000`，UI 模式强制离屏，不改变两个现有 CLI 对 `headless` 的处理；
- 浏览器验收中修复终态 WebSocket 每 100 ms 重发同一会话、进而触发历史帧状态重复请求的问题；终态现在只推送一次，逐帧状态仅在会话/帧号实际变化时加载；
- 新增 Python 9 项与前端 4 项自动化测试。真实 LIBERO 短回归完成 5 个动作、6 个状态，实测 `19.98 Hz`、最短周期 `50.04 ms`、state 恢复误差 `0`，并成功渲染 `512×512` 预览帧。
- 浏览器 GPU 端到端验收完成：原始会话 `816422d9bdb0` 执行 1 次 VLA 查询、1 个动作并保存 18 个 artifact；策略分支 `e2229527a482` 从 frame 0 直接恢复，误差 `1.11e-16`，重新查询后生成 7 张比较图；人工分支 `36113bea2328` 同样恢复并保存 `action_source=human`，三者均为 `COMPLETED`、`error=null`。
- 修复 Web UI 主视角容器固定 `16:9` 与方形 `agentview` 不匹配的问题，容器现在使用 `ui_config.yaml` 的预览宽高比，完整显示画面；
- 修复创建分支时初始 `current_step/action_count/state_count` 暂时归零造成进度条先倒退再前进的问题，分支从提交的 `resume_step` 开始显示；
- 完成会话的预览区改为直接播放已保存的主视角 MP4，视频播放位置、回溯滑块和轨迹帧数值双向同步，避免跨 FastAPI 工作线程复用 EGL 上下文造成的花屏；
- artifact 响应由强制下载改为浏览器内联打开，并使用支持 Range 请求的 `FileResponse`，MP4 可直接 seek，前端结果链接在新标签页打开。
- Web UI 回溯分支关闭 `save_trajectory_bundle` 的独立绘图，只保留 7 张原始轨迹与二次推理/人工接管叠加的 action 对比图；原始会话的轨迹图和 7 张 action 图保持不变。
- Web UI 主视角在桌面端按宽高各 50% 等比例缩小并居中，窄屏仍使用全宽；修复活动分支时间轴值停在 `resume_step`、但最大值持续增长造成滑块相对位置倒退的问题，时间轴现在实时跟随 `current_step` 并在结束时停到最终帧；会话切换时重建上方进度条，避免从原轨迹 100% 向分支回溯点播放反向 CSS 动画。

## 2026-08-15：SpaceMouse 第一阶段诊断与无 VLA 控制

- 新增延迟加载 HID 依赖的 `SpaceMouseInput`，独立线程持续读取 3Dconnexion SpaceMouse，仿真线程只获取最新快照；
- 实机识别新版 USB 线连接 SpaceMouse Wireless 为 `256f:c63a` / `SpaceMouseWirelessNew`（产品字符串为 `SpaceMouse Wireless BT`），并据此收窄 udev 规则；
- 新增可审查的 `udev/70-3dconnexion-spacemouse-wireless.rules`；实测未安装规则时设备节点为 `root:root 0600`，HIDAPI 可以枚举但无法打开；
- 兼容 PyPI `pyspacemouse==2.0.0` 尚无 `AxisConvention` 参数的实际 API：读取 legacy 六轴后在本项目内显式变换为 ROS 右手 Z-up，并支持严格轴重排、符号、死区、增益和可选 EMA；
- 左键锁存夹爪打开、右键锁存夹爪闭合；设备断连、读取异常或输入超过 250 ms 未更新时将六维运动归零；
- 新增固定 `spacemouse_test_config.yaml` 和 `scripts/test_spacemouse.py`，先进行纯设备覆盖测试，再可切换到不加载 VLA 的 20 Hz MuJoCo 人工控制；
- 新增 HID 事件、输入样本、控制周期、`env.step`、Viewer 同步及 deadline miss 的 CSV/JSON 诊断，仿真结束后保存轨迹、动作图和离线 agentview 视频；
- 调整共享手动控制循环的采样顺序，在 20 Hz 控制边界等待完成后再读取最新动作，避免额外引入接近一个控制周期的输入延迟；策略推理的 action chunk 路径保持不变。

## 2026-08-15：SpaceMouse 接入 Web UI 人工分支

- 在保留浏览器键盘接管的基础上，为人工分支增加 `manual_source=browser|spacemouse`，旧人工轨迹仍按原格式兼容读取；
- SpaceMouse 仅在选中接管分支后打开并进行静止校准，UI 启动和普通 VLA 仿真不会占用 HID 设备；
- 位移、旋转增益在创建分支前可调，分支运行中通过 WebSocket 原子更新同一个 `SpaceMouseInput`，无需重连或重新校准；
- UI 新增独立的 HID 样本年龄条，以 `stale_timeout_ms=250` 为安全阈值，实时显示 `ready/stale/disconnected/error` 状态；超时或断连继续执行六维归零；
- SpaceMouse 分支额外原子保存 `spacemouse_samples.csv` 与 `spacemouse_device_summary.json`，记录逐控制步 raw/calibrated/command、按钮、最终夹爪动作、实际增益、延迟分位数和 HID 周期；
- 分支 metadata、session、summary 和历史索引统一保存 `manual_source`、双增益及 SpaceMouse deadman 信息，7 张原始/人工分支 action 对比图保持不变。

## 2026-08-15：SpaceMouse 全局服务、稳定校准与 UI 会话删除

- SpaceMouse 改为 UI 全局常驻服务，公开 `DISCONNECTED / UNCALIBRATED / CALIBRATING / READY / ARMED / ERROR` 状态；设备连接后由用户手动校准一次，后续分支复用 HID 句柄与校准结果；
- 静止校准允许帽盖轻微移动：超过阈值会重置连续静止窗口并提示松开，不再直接使分支 `ERROR`；总等待上限为 30 秒；
- 人工分支移除浏览器键盘/Jog/夹爪动作接口，只保留 SpaceMouse；取得有效实时预览首帧后进入 `READY` 并显示 3 秒倒计时，倒计时结束才 arm 控制器；
- 预览改为固定线程持有的单一只读 MuJoCo 环境；创建分支时从父视频提取 `resume_preview.jpg`，实时首帧出现前不再黑屏；
- 创建分支时物理复制父 `trajectory.npz/json` 为子目录的 `source_trajectory.npz/json`，删除父会话不会破坏已有子分支；准备阶段零新动作失败时跳过视频、observations 和曲线重建；
- 顶部增加控制器状态/延迟 pill：接管中 `<50 ms` 为绿、`50–249 ms` 为黄、`≥250 ms`/断连/错误为红；
- 新增受路径、清单 ID、符号链接和活动状态保护的 UI 会话永久删除接口与垃圾桶确认对话框；外部 CLI、intervention 和 legacy 数据保持只读且不可删除。

## 2026-08-15：UI 结果 JSON 去重

- 新 UI 会话的 JSON 输出收敛为 `session.json` 与 `summary.json`：前者只保留安全删除、崩溃恢复和历史识别所需字段，后者只保留任务上下文、最终结果和关键性能指标；
- 停止为 UI 会话生成与 summary 重复的 `results.jsonl`，轨迹 metadata 只嵌入 `trajectory.npz`，不再额外输出 `trajectory.json`；
- 分支只复制包含内嵌 metadata 的 `source_trajectory.npz`，不再生成重复的 `source_trajectory.json`；
- SpaceMouse 的设备名、增益、校准 ID/偏置、样本年龄和 HID 时序摘要合并进 `summary.json`，完整逐步输入仍保留在 `spacemouse_samples.csv`，不再生成独立的 `spacemouse_device_summary.json`；
- 历史扫描同时兼容旧版扁平清单/summary 与新版精简格式；已有实验目录保持原样，不自动迁移或删除。

## 2026-08-15：三任务目录与仿真草稿

- 固定任务目录扩展为三个 LEVEL1 Franka 任务：黑碗放到平炉、打开木柜顶层抽屉、蓝碗叠到绿碗；启动时统一校验 BDDL、init 文件和 BDDL 提示词；
- 三个任务继续共用 `LIBERO-Object-Pro`，不增加 `experimental_ood` 字段、提示或特殊执行路径，任务成功与成功率继续使用各自 LIBERO 环境的原始判定；
- 新增纯内存 `SimulationDraft` 与 `/api/draft` 接口，将“创建仿真”和“开始仿真”分离；草稿可切换任务、修改总步数和 action chunk 执行步数，预览就绪前不能开始；
- 草稿加载第一个 benchmark init state 并复用单一任务感知预览线程，revision 防止旧任务异步帧覆盖新选择；取消、刷新或替换草稿不会创建结果目录；
- 原始会话固化 `task_id`，分支继承父任务；控制环境、策略提示词、预览、历史帧渲染、视频重建和结果 metadata 均按会话任务解析，不再回退到全局固定场景；
- 已知旧轨迹按 `LEVEL + task_name` 映射到任务目录并可继续创建一级分支，未知任务保持只读；历史帧环境在任务变化时安全关闭并重建；
- 修复策略模型跨会话复用时 `open_loop_steps` 停留在首次加载值的问题，每次会话开始都会更新查询截断配置。
- 修复离线时已缓存 Hub checkpoint 被辅助 action/proprio projector 误判为本地目录的问题；先检查 Hugging Face 本地缓存，再决定是否走 Hub 文件解析，避免主模型加载完成后才因目录断言失败。

## 2026-08-15：Local Data Studio 架构与数据目录整理

- 将扁平 `liberox_ui/` 拆为 `backend/app/api、domain、services、workers、simulators、policies、recording、evaluation、storage、core`，HTTP/WebSocket 只依赖 `RunService`；
- 会话和草稿成为不导入 FastAPI、MuJoCo 或 SQLite 的领域对象；LIBERO-X 模拟器、VLA provider、任务目录、轨迹 recorder 和 evaluator 均使用独立适配器；
- 保持三任务一致的 LIBERO `done` 成功判定，没有引入 OOD 特殊分支；
- 新增 `dataset-root/catalog.sqlite3` 目录索引，新 run 按项目、任务和日期分组，run 根目录只放 `run.json/config.yaml/summary.json`，高体积 artifact 收入 `episodes/episode_000/`；
- SQLite 只作为可重建索引，run 文件仍是事实来源；旧 `runs/` 通过 `legacy_scan_roots` 只读展示，不迁移、不覆盖；
- FastAPI 路由拆分为 runs、draft、controller、dataset 和 WebSocket，并新增 `/api/runs` 与 `/api/datasets/summary`；
- React 拆为 app/pages/features/components/api/styles，新增运行记录、数据集和设置页面；采集页常驻以避免页面切换中断 SpaceMouse；
- UI 改为浅灰背景、半透明白色面板和系统字体的简洁样式，删除原深蓝渐变主题；
- 新增架构、数据目录、SQLite repository 测试和英文入口说明；旧扁平后端目录已删除。

## 2026-08-18：主视角录像播放抽动

- 定位到视频与回溯时间轴的双向同步反馈：视频 `timeupdate` 更新 `selectedStep` 后，通用 effect 又把离散帧时间写回 `currentTime`；每秒会话轮询替换 `selected` 对象时还会重复触发 seek，使浏览器短暂进入 `seeking/waiting`；
- 删除播放过程中由 `selectedStep` 自动反向 seek 视频的 effect。正常播放现在只执行“视频 → 时间轴/轨迹数值”；仅用户拖动时间轴、视频首次加载或切换会话时执行“时间轴 → 视频”；
- seek 回调依赖从整个会话对象收窄为稳定的 `action_count` 与 `video_fps`，会话状态轮询不再干预播放器。

## 2026-08-18：拉取新代码后仍显示旧版 UI

- 根因是 `frontend/dist` 属于被 Git 忽略的本机构建产物：`git pull` 更新 React/CSS 源码时不会更新或删除另一台电脑已有的旧 `dist`，FastAPI 因而继续托管旧版三行监视器页面；
- 新增前端源码指纹，覆盖 `src/`、入口 HTML、Vite/TypeScript 配置和 npm 清单。`run_ui.py` 启动时发现指纹缺失或变化会先显示完整构建目录与命令，再自动执行一次 `npm run build`；未变化时不重复构建，并显示当前指纹已是最新版本；
- 新电脑缺少 `node_modules` 或 npm 时不再静默加载旧页面，而是停止启动并给出 `npm ci` 的明确处理命令；
- 前端入口 HTML 增加 `Cache-Control: no-store`，避免构建已更新但浏览器仍复用旧入口；Vite 的哈希资源文件继续正常使用文件名版本隔离；
- 新增自动构建、无变化跳过、源码变化重建、依赖缺失提示和 HTML 缓存头测试。
- 新增 `/api/build-info`，返回实际服务目录、源码/构建指纹与 bundle 文件名；顶部常驻显示 12 位 UI 构建指纹，启动日志同时打印项目绝对路径，能够区分旧端口进程、旧目录和旧静态资源；
- SpaceMouse 的 `DISCONNECTED` 状态不再隐藏探测错误：缺少 `pyspacemouse`、HID 权限问题、枚举异常和未发现配置中的精确 VID/PID 会给出不同提示，完整错误仍通过 `/api/controller` 返回。

## 2026-08-19：RynnValue + IQL 独立后训练与 policy overlay

- 新增独立 `vla-adapter-rynn-iql/` 项目，按 YAML 完成采集数据只读导入、固定版本 RynnValue-4B 时间价值标注、PyTorch Pixel-IQL 后训练和 LIBERO-X 推理；不引入 Robometer，也不修改 VLA-Adapter 上游源码；
- 数据导入严格校验 N+1 状态/双视角对齐、20 Hz 时间网格、Franka 8D proprio、7D OSC action 和策略动作回放；原始轨迹完整入库，分支仅增加回溯点后的新后缀，父子轨迹按 root 分组切分；
- RynnValue 按官方固定前缀采样方式预测每个 action-chunk 边界的剩余秒数，保存 entropy 与文本诊断，并用 `Φ=-v` 和 `γ^L` 构造 PBRS chunk reward；环境 `done` 仍是唯一成功依据；
- IQL 实现 double-Q、expectile V、Polyak target、实际 chunk 长度折扣和 advantage-weighted masked L1，只训练连续 action head 与 proprio projector；checkpoint 保存优化器、目标网络、采样器与 RNG，可精确断点续跑；
- 训练结果以带组件哈希和兼容性哈希的 `policy.yaml` 发布到 `policy-registry/`。UI 草稿新增基础模型/overlay 选择，会话和分支固化策略 ID，同一 Object-Pro backbone 上热切换两个组件；非法 overlay 明确拒绝且不回退。

## 2026-08-20：RynnValue 官方源码不可 editable-install

- 根因是固定 commit 的官方顶层 `pyproject.toml` 设置 `tool.uv.package = false`，且没有限制 setuptools 的 flat-layout package discovery；`pip install -e ./RynnValue` 会同时发现 `assets/robometer/rynn_value/rynn_infer` 并拒绝构建，license 信息只是警告而不是失败原因；
- 删除错误的 editable-install 步骤，新增严格 YAML 路径 `paths.rynnvalue_root`，奖励进程直接从固定的官方 checkout 导入 `rynn_value`，无需修改上游仓库或持久设置 `PYTHONPATH`；
- 补齐官方模型源码实际需要的 `einops` 依赖；验证脚本现在输出真实 import 路径，并在执行官方 Python 代码前先检查 Git commit 与工作树洁净状态；
- 在 `rynnvalue-reward` 实机环境验证通过：源码 commit 为 `10e0d333f5f3811d0d130587e50f1faf48da49e5`，模型 revision 为 `3f73b5d2b5e53b21f248c8791004dde6a8cf2b92`。

## 2026-08-20：成功后锁存 `done=True` 导致数据准备失败

- 根因是固定总时长的接管轨迹会在首次成功后继续记录，环境 `done` 因而在余下每一步保持 `True`；旧导入器错误地只允许最后一步出现一次 `True`，将合法的终止状态锁存误判为“终止后仍有动作”；
- 数据准备现在以第一次 `done=True` 为唯一 terminal，保留成功动作及其 next observation，只在 replay、RynnValue 标注和 IQL chunk 中逻辑排除后续锁存尾段，不修改源轨迹；
- manifest 同时保存原始 `recorded_action_count`、有效 `action_count`、`terminal_step` 与排除的 `trailing_action_count`，数据哈希也覆盖这些终止语义；
- 后续考虑到短暂进入成功区域不一定代表稳定完成，新增严格 YAML 参数 `data.success_consecutive_steps`，默认连续 5 步（20 Hz 下 250 ms）才确认成功；一次 `False` 会重置连续计数，第 5 个确认 action 才成为 terminal，未达到阈值的零散 `True` 按失败处理；
- PBRS sparse reward 与 replay bootstrap 改为只使用去抖后的有效 terminal，不再直接读取可能波动的原始 `done`；manifest 同时保留源成功判定、去抖训练判定、连续区间、原始 True 数和终止后统计；回归测试覆盖锁存成功、`True → False → True` 重新计数、未确认脉冲、terminal chunk 长度和源 NPZ 字节不变。

## 2026-08-20：RynnValue value head 与 Qwen backbone 精度不一致

- 根因是官方 `LinearValueHead` 和 `BroValueHead` 构造函数显式默认 `torch.float32`；原标注器虽然用 BF16 参数加载 checkpoint，最后却只执行 `.to(device)`，无法保证自定义 value head 与 BF16 Qwen backbone 使用相同 dtype；
- 标注器改为显式使用固定本地源码的 `RynnValueLangConfig/Processor/Model`，不允许错误回退到普通 Qwen language-model head；加载后按官方推理程序执行整模 `.to(device=..., dtype=torch.bfloat16)`；
- 标注前校验 `2560 × 8 = 20480` 的 value-head 输入契约，并扫描全部浮点参数；任何残留 FP32/FP16 参数都立即报告具体名称。第一版配置固定 `reward.dtype: bfloat16`，value 解码与 entropy softmax 仍按官方实现使用 FP32 保持数值稳定。
