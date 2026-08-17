# LIBERO-X × VLA-Adapter Terminal

这是一个面向 Franka/LIBERO-X 的本机仿真、VLA 评测、轨迹回溯、SpaceMouse 接管与数据管理终端。当前 UI 已验证三个 LEVEL1 任务；下文保留黑碗任务作为 CLI 配置示例。

默认任务：

- 场景：`LEVEL1`
- BDDL：`EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove.bddl`
- 指令：`place the black bowl on the flat stove`
- 机器人：Franka/Panda 单臂
- 策略：`VLA-Adapter/LIBERO-Object-Pro`
- 输入：第三视角 RGB、腕部 RGB、8 维本体状态、语言指令
- 输出：8 步动作块；每步为 7 维末端增量动作

> 这是仿真评测框架。不要把仿真动作直接发送给真机。真机需要独立的坐标标定、限位、碰撞检测、急停和低速验证层。

## 1. 推荐架构

```text
LIBERO-X BDDL + 固定初始状态
              │
              ▼
   OffScreenRenderEnv (Franka)
              │
      RGB × 2 + proprio(8) + text
              │
              ▼
       VLA-Adapter policy
              │
       action chunk: 8 × 7
              │
              ▼
      gripper 反归一化/符号恢复
              │
              ▼
        env.step(action)
              │
              ▼
   success rate + JSON + rollout MP4
```

本模板采用单进程直连方式，便于第一次排错。LIBERO-X 官方 `eval_template.py` 使用 WebSocket 客户端/服务端；等单任务成功后再切回服务模式更稳妥。

## 2. 环境准备

建议：Ubuntu 20.04/22.04、NVIDIA GPU、CUDA 可用、至少 16 GB 显存更从容。以下命令假设两个上游仓库和本模板位于同一工作目录。

```bash
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
git clone https://github.com/meituan/LIBERO-X.git

# 本模板已按以下上游版本核对；固定版本可避免后续接口变化。
git -C VLA-Adapter checkout 23fa0c9c159e2aa04341cdd3e924f44061311060
git -C LIBERO-X checkout f528726421c7211d8eb05fe48e9e5e2535ccc813

conda create -n vla-liberox python=3.10.16 -y
conda activate vla-liberox

cd VLA-Adapter
pip install -e .
pip install packaging ninja

# 使用 LIBERO-X 自带的 libero 分支，但不要安装其 torch==1.11 依赖，
# 否则会覆盖 VLA-Adapter 所需的 torch==2.2.0。
pip install -e ../LIBERO-X --no-deps
pip install -e ../LIBERO-X/packages/openpi-client
pip install -r ../liberox-vla-adapter-terminal/requirements-sim.txt

# Web UI 后端依赖；只使用 CLI 时可跳过。
pip install -r ../liberox-vla-adapter-terminal/requirements-ui.txt

# 可选：3Dconnexion SpaceMouse 独立诊断和人工控制。
sudo apt-get install -y libhidapi-dev
pip install -r ../liberox-vla-adapter-terminal/requirements-spacemouse.txt

# 构建由 FastAPI 托管的前端静态资源。之后日常启动不再需要 npm。
cd ../liberox-vla-adapter-terminal/frontend
npm ci
npm run build
cd ../../VLA-Adapter

# 让 Hub checkpoint 使用当前固定提交中的本地 OpenVLA 实现，
# 保持 trust_remote_code=False，不执行模型仓库中的 Python 文件。
git apply ../liberox-vla-adapter-terminal/patches/vla_adapter_hf_local_autoclass.patch
```

`flash-attn` 不是当前评测链路的必需项，源码中的 FlashAttention 开关也未启用。只有在系统存在匹配的 CUDA Toolkit 和 `nvcc` 时再按需安装。

RTX 50 系列（Blackwell、`sm_120`）不能使用项目原始的 PyTorch 2.2/CUDA 12.1 wheel。本机 RTX 5090 Laptop 使用以下组合完成了 rollout：

```bash
pip install --upgrade \
  "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" \
  --index-url https://download.pytorch.org/whl/cu128

# 防止间接依赖覆盖 TensorFlow 2.15 和旧版 wandb 的兼容版本。
pip install "numpy==1.26.4" "setuptools==69.5.1"
```

无桌面服务器需要 EGL 运行库：

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
export MUJOCO_GL=egl
```

评测脚本默认还会从 `config.yaml` 设置 `MUJOCO_GL=egl`；上面的 `export` 适用于其他 MuJoCo 程序，或在配置中将 `mujoco_gl` 设为 `null` 时使用。

先检查关键版本：

```bash
python - <<'PY'
import torch, robosuite, mujoco
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("robosuite", robosuite.__version__)
print("mujoco", mujoco.__version__)
PY
```

普通 CUDA GPU 可沿用项目固定的 `torch==2.2.x`；RTX 50 系列应为 `torch==2.7.0+cu128` 或其他明确支持 Blackwell 的 CUDA 12.8+ wheel。`robosuite` 应为 `1.4.x`，MuJoCo 应为 `2.3.7`。不要在同一环境中再次运行 `pip install -r LIBERO-X/requirements.txt`。

## 3. 分阶段跑通

评测参数集中在本模板根目录的 `config.yaml`。配置文件内部的相对路径均以该 YAML 所在目录为基准；下文的相对脚本命令则假设当前目录为 `vla-liberox-workspace`。先进入工作区：

```bash
cd vla-liberox-workspace
```

默认配置对应单回合完整评测，并包含：

```yaml
vla_root: ../VLA-Adapter
liberox_root: ../LIBERO-X
output: ../runs/liberox_pickplace_l1
checkpoint: VLA-Adapter/LIBERO-Object-Pro
use_pro_version: null
stats_key: libero_object
level: LEVEL1
task_name: EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove
trials: 1
max_steps: 300
seed: 0
control_hz: 20
realtime_control: true
env_resolution: 256
video_camera: vla_views
video_width: 512
video_height: 256
main_view_video_width: 1024
main_view_video_height: 1024
video_fps: 20
open_loop_steps: 8
headless: false
env_only: false
no_video: false
save_trajectory: true
save_observation_images: true
trajectory_plot: true
cuda_visible_devices: "0"
mujoco_gl: egl
```

`use_pro_version: null` 表示根据 checkpoint 名称中的 `Pro` 自动选择动作头。`cuda_visible_devices` 或 `mujoco_gl` 设为 `null` 时保留当前 shell 环境。Hub checkpoint 保持 `组织/仓库` 格式；本地 checkpoint 使用以 `./`、`../`、`~` 或 `/` 开头的路径。

`control_hz: 20` 显式把 LIBERO 控制周期设为 `50 ms`；`realtime_control: true` 在每次 `env.step()` 前使用单调时钟限速，保证控制步不会快于 20 Hz，也不会在落后后突发“追帧”。正式 rollout 前会在一个随后销毁的临时环境中完成 robosuite 控制器的一次性初始化，再新建正式环境，因而不会用隐藏 action 污染正式初始状态。模型查询或系统调度如果超过 50 ms，有效频率仍只会低于 20 Hz；这是软实时仿真，不是具有硬实时保证的机器人控制器。每回合的实际 `measured_control_hz` 与控制周期最小值、平均值、最大值会写入 `results.jsonl`，汇总值和超期回合数写入 `summary.json`。每个控制步记录一帧，因此 `video_fps` 强制等于 `control_hz`：300 步对应 15 秒仿真时间和 15 秒视频，不再以 10 fps 播放成 30 秒。

`env_resolution` 控制送给 VLA 和写入观测轨迹的 `agentview`/腕部相机，保持 `256×256` 以维持模型输入契约。默认录像模式 `vla_views` 会把这两路原始策略观测直接水平拼接：左侧是 `agentview`，右侧是 `robot0_eye_in_hand`，不做高清重渲染或缩放，最终保存为 `episode_000_{success|failure}_vla_views.mp4`（`512×256`）。按照 VLA-Adapter 官方训练契约，两路输入必须旋转 180°，因此它可能与 MuJoCo Viewer 的自然显示方向不同，不能为了观感而改动；模型内部仍会按原路径将每路图像处理为 `224×224`。

每次评测的标准文件 `episode_000_{success|failure}.mp4` 固定为 `1024×1024` 高清主视角。它使用同一个 `agentview` 相机位姿，但只校正 OpenGL framebuffer 的上下方向，不应用 VLA 训练所需的额外水平翻转，因此是供人查看的自然方向。`main_view_video_width` 和 `main_view_video_height` 只控制录像，必须保持相等，不会改变模型观测。`results.jsonl` 中的 `video` 和 `main_view_video` 都指向该标准文件，`vla_views_video` 指向保持模型输入方向的双相机拼接文件。

`headless: false` 会把当前 LIBERO 环境的同一个底层 `mujoco.MjModel/MjData` 交给 `mujoco.viewer.launch_passive()`，打开 MuJoCo 原生交互窗口。Viewer 初始选择与标准视频相同的 `agentview` 固定相机；可在右侧 Camera 面板切换为 `Free` 后旋转、平移和缩放。代码默认关闭 robosuite 的 collision geom group 0、保留 visual geom group 1，避免碰撞简化体与机械臂视觉网格重叠产生绿色/黄色块和闪烁；仍可在 Viewer 面板中手动重新打开。原始评测和 intervention 都支持该模式，VLA 离屏输入和录像不受 Viewer 自由相机操作影响。改回 `headless: true` 即不创建窗口。非 headless 模式需要本地图形桌面、X11 转发或可用的 Wayland/XWayland 会话。

原生 Viewer 以 passive 模式连接，只查看当前仿真，不自行推进物理状态；脚本在恢复状态和每个控制步后调用 `sync()`。当前评测是单进程的，因此 VLA 正在生成下一组 action 时画面可能短暂停留在上一状态，推理完成后会继续更新；这不表示仿真停止。关闭 Viewer 只会关闭观察窗口，rollout 会继续；需要提前结束推理时仍在运行终端按 `Ctrl+C`。

为避免相机渲染破坏 50 ms 控制预算，实时 `env.step()` 不再无条件生成两路图像：只在发起 VLA 查询时按需采集策略输入。完整观测序列和两份视频在 rollout 结束、Viewer 关闭后，依据已记录的 MuJoCo state 逐帧重建。因此动作结束后终端还会出现 `Post-processing ... recorded states`，这是在生成结果，不是再次执行策略；高清编码耗时不会改变刚才的仿真节拍。

也可以把 `video_camera` 改为 `frontview`、`birdview`、`sideview` 或 `galleryview`，此时 `video_width`/`video_height` 是该单相机的渲染分辨率。`birdview` 覆盖范围最大，`frontview` 更适合观察机械臂整体运动；宽高有效范围均为 `64..2048`。

脚本强制读取 `liberox-vla-adapter-terminal/config.yaml`，不接受其他配置地址或运行参数。需要调整实验时直接编辑该文件，运行命令只有一行：

```bash
python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py
```

### 3.1 只验证场景、初始状态和观察量

在 `config.yaml` 中设置：

```yaml
env_only: true
headless: false  # 同时验证实时窗口；服务器上设为 true
```

然后运行：

```bash
python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py
```

通过标准：输出包含：

- `agentview_image: (256, 256, 3)`
- `robot0_eye_in_hand_image: (256, 256, 3)`
- `video_frame[vla_views]: (256, 512, 3)`（左 `agentview`，右腕部相机，无缩放）
- `proprio: (8,)`
- `dummy_action: (7,)`

这一步不下载模型权重。

### 3.2 单次模型 rollout

在 `config.yaml` 中设置：

```yaml
env_only: false
trials: 1
max_steps: 300
output: ../runs/liberox_pickplace
```

```bash
python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py
```

首次运行会下载 VLA、action head 和 proprio projector。通过标准：

1. 模型成功加载，打印 LIBERO 常量：action chunk 8、action dim 7、proprio dim 8；
2. 每次模型查询返回形状为 `(N, 7)` 的动作；
3. 生成与 VLA 外部 `agentview` 同机位的高清 `episode_000_{success|failure}.mp4`，以及原像素双输入拼接的 `episode_000_{success|failure}_vla_views.mp4`；
4. 生成 `results.jsonl` 和 `summary.json`；
5. 即使首次任务失败，只要能完整 rollout，也说明接口链路已经跑通。

本机 RTX 5090 Laptop 实测完成一次完整回合：300 步、38 次模型查询、`error: null`，每份输出视频都包含 300 帧。策略本次未完成任务，因此文件名中包含 `failure`；这里的 `failure` 表示任务结果，不表示 rollout 链路异常。正常完成但任务失败仍返回退出码 0；如果任一回合出现模型、环境或后处理异常，`summary.json` 的 `episode_errors` 会大于 0，进程返回非零退出码。

### 3.3 扩大评测

需要扩大到 10 回合时，把 `config.yaml` 改为：

```yaml
env_only: false
trials: 10
output: ../runs/liberox_pickplace_l1_config
```

```bash
python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py
```

报告至少保留：成功数、总回合数、成功率、checkpoint、BDDL 文件、随机种子、每回合视频和轨迹文件。

### 3.4 保存末端 6DoF、夹爪与可回溯状态

默认配置中的以下选项已经启用：

```yaml
save_trajectory: true
save_observation_images: true
trajectory_plot: true
```

每个 episode 会新增：

- `trajectory_000.npz`：轻量核心轨迹，包含每一步的完整 MuJoCo state、末端位置、旋转、双指夹爪位置、实际执行的 VLA raw action、环境动作、reward、done、动作来源，以及每次模型查询返回的完整 action chunk；
- `trajectory_000_observations.npz`：主相机和腕部相机图像，用于构造新的示范数据；
- `trajectory_000.csv`：可直接查看的末端 6DoF、夹爪、VLA raw action 和实际环境动作表格；
- `trajectory_000_inference.csv`：逐 query、逐 chunk index 保存 VLA 返回的全部候选动作，包括没有被执行的预测动作；
- `trajectory_000.json`：任务、checkpoint、随机种子、成功状态和数据格式说明；
- `trajectory_000_plot.png`：一张同时包含 XYZ 位置、axis-angle 三轴旋转和左右夹爪位置的曲线图。
- `trajectory_000_action_dx.png`、`..._dy.png`、`..._dz.png`、`..._drx.png`、`..._dry.png`、`..._drz.png`、`..._gripper.png`：7 张彼此独立的 VLA action 曲线；粗蓝线是实际执行的 raw action，淡橙线是每次推理提出的完整 action chunk。

Action 图的下横轴单位是控制帧 `[frame]`，上横轴单位是仿真时间 `[s]`；纵轴是 VLA 输出的归一化控制量 `[-]`，不是米或弧度。所有相邻 action 都按照时间顺序连续连线。

末端 6DoF 表示为：

```text
[eef_x, eef_y, eef_z, axis_angle_x, axis_angle_y, axis_angle_z]
```

文件中还会额外保留原始四元数 `[qx, qy, qz, qw]`。轨迹采用 `N` 个动作对应 `N+1` 个状态的格式：`state[i] --action[i]--> state[i+1]`。默认 LIBERO 控制频率是 20 Hz，因此第 150 个 state 对应 7.5 秒。

### 3.5 回溯、重新推理与人工接管

先至少重新运行一次 3.2 或 3.3，生成上述 `trajectory_*.npz`。然后编辑固定的 `intervention_config.yaml`：

```yaml
source_trajectory: ../runs/liberox_pickplace_l1_config/trajectory_000.npz

# 两者只设置一个；另一个必须为 null。
resume_step: 150
resume_time_seconds: null

control_mode: policy
open_loop_steps: 1
output_root: ../runs/liberox_interventions
```

干预脚本不再单独设置运行长度。假设源轨迹共有 `N` 个动作，从 `resume_step=K` 回溯后会自动执行 `N-K` 个新动作，因此合并后的轨迹仍然严格包含 `N` 个动作，视频帧数和源轨迹保持一致。例如 300 步源轨迹从 step 50 回溯，会重新生成后 250 步，最终仍是 300 步。

两个 YAML 都使用 `open_loop_steps` 表示“每次 VLA 预测 8 个 action 后，实际连续执行其中多少个”。字段名称和有效范围 `1..8` 完全一致，但两个文件的值彼此独立：`config.yaml` 控制原始评测，`intervention_config.yaml` 控制回溯后的分支推理。

干预结束后会根据完整 state 序列流式生成两份视频，播放帧率统一继承 `config.yaml` 的 `video_fps`，并与 `control_hz` 相等：

- `intervention.mp4`：沿用 `config.yaml` 的 `video_camera`、`video_width` 和 `video_height`；默认是未经缩放的 `512×256` VLA 双视角拼接，左主视角、右腕部视角；
- `intervention_agentview_hd.mp4`：沿用 `main_view_video_width` 和 `main_view_video_height`，使用与推理主图相同、位置更高的 `agentview` 相机，默认 `1024×1024`。

两份视频的原轨迹前缀和新分支都依据保存的 MuJoCo state 在控制结束后重新渲染，因此帧数、回溯点和相机位姿严格对齐。视频采用流式编码，不会把两组高清帧同时堆积在内存中，也不会在人工/VLA 控制期间阻塞 20 Hz 循环。

intervention 同样读取 `config.yaml` 的 `headless`：为 `false` 时打开 MuJoCo 原生 Viewer，并从 `resume_step` 恢复点开始同步当前新分支，不会把回溯点之前的历史前缀快速播放一遍；历史前缀仍会正常写入两份最终视频。

运行命令同样不接受配置地址：

```bash
python liberox-vla-adapter-terminal/scripts/intervene_pickplace.py
```

脚本会恢复指定 state，而不是从头近似执行旧动作。恢复后清空旧 action chunk，并按以下模式产生一条新分支：

- `policy`：从指定点重新查询 VLA。`open_loop_steps` 与 `config.yaml` 中的字段同名；设为 `1` 时每一步都重新规划；
- `manual_stdin`：人或外部程序通过标准输入逐条发送动作；
- `manual_jsonl`：从 `manual_action_file` 读取可复现的人工动作；
- `manual_udp`：外部手柄/控制器程序通过 UDP 实时发送动作，默认监听 `127.0.0.1:5555`。

人工动作直接使用 LIBERO 的 7 维 OSC_POSE 环境格式：

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

每个值必须在 `[-1, 1]`，夹爪 `-1` 表示张开、`+1` 表示闭合。`manual_stdin` 接受空格分隔、JSON 数组，或带重复次数的对象：

```text
0.1 0 0 0 0 0 -1
{"action": [0, 0, -0.1, 0, 0, 0, 1], "repeat": 5}
stop
```

`manual_jsonl` 的每一行使用相同 JSON 格式。`manual_udp` 接收同样的 JSON 数据报，并在每个动作执行后回复当前 `step`、`time_seconds`、`eef_6d`、`gripper_qpos` 和成功状态。人工模式可在运行期间持续刷新临时的 `latest_frame.png`，方便外部控制器界面显示当前视角；干预结束后该文件会自动删除，不作为最终结果保留。policy 模式不会创建这个文件。

输出目录自动带时间戳，不覆盖以前的干预。新轨迹会把原轨迹 `[0, resume_step)` 的前缀和接管后的新动作合并保存；CSV/NPZ 中通过 `action_source=policy`、`policy_requery` 与 `human` 区分各段，轨迹图用竖直虚线标出干预点。这就是失败轨迹回溯、人工接管并生成新示范数据的完整闭环。

每次干预只生成一组 7 张动作对比图：

- `trajectory_action_comparison_{dx,dy,dz,drx,dry,drz,gripper}.png`：每个 action 维度一张比较图。第一次推理曲线完整保留在 frame `0..N-1`；二次推理/人工接管曲线只从 `resume_step` 开始叠加到相同的总终点。例如回溯到第 50 帧时，原始曲线从第 0 帧开始，分支曲线从第 50 帧开始。

干预输出不再生成二次推理自身的 `trajectory_action_*.png`、单独的轨迹曲线图或最终时刻截图；轨迹 NPZ、CSV、JSON、双相机观测、两份视频和 summary 仍正常保存。

该 NPZ 是无 pickle 的中间数据格式；用于 `finetune.py` 前仍需按 5.2 节转换为 RLDS。旧 schema v1 轨迹仍可回溯和比较实际执行的 raw action；若需要保留原始评测每次查询返回的完整 action chunk，需要用当前脚本重新运行一次评测。

### 3.6 SpaceMouse 设备诊断与无 VLA 人工控制

第一阶段 SpaceMouse 支持使用 `PySpaceMouse 2.0 + HIDAPI` 直接读取设备，不使用 3Dconnexion 驱动，也不与 `spacenavd` 混用。实机枚举确认：当前通过 USB 数据线连接的是新版 SpaceMouse Wireless；产品字符串虽然显示 `SpaceMouse Wireless BT`，USB 身份实际是：

接口依据：[PySpaceMouse](https://github.com/JakubAndrysek/pyspacemouse)、[HIDAPI](https://github.com/libusb/hidapi) 与 [3Dconnexion Software Developer Program](https://3dconnexion.com/us/software-developer-program/)。

```text
VID=256f PID=c63a（PySpaceMouse 名称：SpaceMouseWirelessNew）
```

Ubuntu 默认可能禁止普通用户读取对应的 `hidraw` 节点。创建只匹配这一型号的 udev 规则，不要对全部 HID 设备设置 `0666`：

```bash
sudo install -m 0644 \
  liberox-vla-adapter-terminal/udev/70-3dconnexion-spacemouse-wireless.rules \
  /etc/udev/rules.d/70-3dconnexion-spacemouse-wireless.rules
sudo udevadm control --reload-rules
```

随后拔下并重新插入 SpaceMouse（只 reload 规则不会修改已经存在的 hidraw 节点）。测试脚本固定读取模板根目录的 `spacemouse_test_config.yaml`，不接受配置地址：

```bash
cd vla-liberox-workspace
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/test_spacemouse.py
```

首次诊断时设置 `mode: device`，只验证 HID，不导入 MuJoCo、LIBERO 或 VLA。静止校准期间不要触摸帽盖；设备在完全静止时不发送新报告属于正常情况，此时使用已初始化的零状态，随后的覆盖测试仍会验证真实 HID 报告。倒计时后依次让六个轴向正负两个方向运动并按下左右键。终端显示 raw 输入和最终 OSC_POSE command，运行结束后检查 `summary.json` 的 `functional_check_complete` / `acceptance_passed` 与 `device_summary.json` 的轴/按钮覆盖率。未完成覆盖或性能验收时脚本以状态码 `2` 结束，运行错误使用状态码 `1`。

确认设备读取正确后，将 `spacemouse_test_config.yaml` 改为：

```yaml
mode: simulation
```

再次运行相同命令即可在当前 `config.yaml` 的 LEVEL、任务、seed 和第一个 benchmark init state 中控制机械臂。该模式不加载 VLA：

- SpaceMouse 独立线程以约 1 ms 间隔非阻塞读取 HID，20 Hz 控制环在每个控制边界只取最新快照；
- PySpaceMouse 2.0.0 固定输出 legacy 轴；本项目再显式映射为 ROS 右手 Z-up normalized OSC_POSE `[X,Y,Z,Rx,Ry,Rz,gripper]`，默认映射是 `[legacy_y,-legacy_x,legacy_z,legacy_roll,legacy_pitch,-legacy_yaw]`；当前位移/旋转增益分别为 `0.25 / 0.08`；
- 左键将夹爪锁存为打开 `-1`，右键锁存为闭合 `+1`；进入 simulation 即启用六轴，不要求按住按钮；
- 设备断连、读取异常或超过 `250 ms` 没有新 HID 报告时，六维运动立即归零，夹爪保持最后状态；
- `Ctrl+C`、关闭 MuJoCo Viewer、任务成功或达到 `max_steps` 都会安全结束并保存已有轨迹；
- 视频根据 state 在控制结束后离线渲染，不消耗实时 50 ms 控制预算。

每次运行都会在 `../runs/spacemouse_tests` 下创建唯一目录，主要文件如下：

- `device_summary.json`：设备身份、校准、HID 事件间隔、样本年龄和六轴/按钮覆盖率；
- `spacemouse_samples.csv`：每次控制采样的 raw、校准后、最终 command、按钮和样本年龄；
- `control_timing.csv`：控制周期、`env.step`、Viewer 同步、样本年龄和 deadline miss；
- `spacemouse_trajectory.{npz,csv,json}` 与动作图；
- `spacemouse_agentview.mp4`：结束后生成的 agentview 回放；
- `summary.json`：成功状态、停止原因、实测频率、P50/P95/P99 和推测瓶颈。

`smoothing_alpha: 1.0` 默认不滤波，用于真实测量设备输入。若实测确认有抖动，再降低该值；轴方向不符合操作习惯时，只修改严格校验的 `axis_order` 和 `axis_signs`。如果脚本能枚举设备但无法打开，先检查 udev 规则和是否有其他 HID/spacenav 进程占用设备。

独立测试和 Web UI 复用同一个 `SpaceMouseInput`、轴映射、静止校准、按钮锁存和 250 ms stale deadman 实现，因此通过本节实机验收后无需维护第二套设备控制代码。

### 3.7 仿真与干预 Web UI

Web UI 将原始仿真、实时查看、结束后的逐帧回溯、VLA 重新推理、人工接管和结果下载整合到同一个本机页面。后端固定监听 `127.0.0.1:8000`，不开放局域网，也不包含认证功能。

从 `vla-liberox-workspace` 启动：

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

然后浏览器打开 `http://127.0.0.1:8000`。

UI 仍读取 `config.yaml` 中的 checkpoint、seed、相机和 20 Hz 控制设置。任务目录默认包含 `config.yaml` 的黑碗任务，并由 `ui_config.yaml` 追加两个 LEVEL1 Franka 任务：

- `place the black bowl on the flat stove`；
- `open the top drawer of the wooden cabinet`；
- `stack the blue bowl on the green bowl`。

三个任务都使用 `VLA-Adapter/LIBERO-Object-Pro`，不附加 `experimental_ood` 标签，也不修改环境成功条件：成功仍完全由对应 BDDL/LIBERO 环境判定，`success` 和成功率沿用原有统计语义。

新建原始仿真采用明确的两阶段流程：先点击“创建仿真”，在内存草稿中切换任务、调整以下参数并检查第一个 init state 的静态预览；只有预览就绪后，“开始仿真”才会创建唯一结果目录并加载策略：

- `max_steps`：本次总控制步数；
- `open_loop_steps`：每次 VLA 预测 8 个 action 后实际执行的数量，有效范围 `1..8`。

UI 专属参数固定从 `ui_config.yaml` 读取，启动命令不接受配置地址：

```yaml
host: 127.0.0.1
port: 8000
dataset_root: ../dataset-root
project_id: libero_x_vla
legacy_scan_roots: [../runs]
preview_width: 512
preview_height: 512
preview_fps: 10
jpeg_quality: 85
manual_translation_gain: 0.25
manual_rotation_gain: 0.08
additional_tasks:
  - level: LEVEL1
    task_name: EXTENSION_KITCHEN_SCENE1_open_the_top_drawer_of_the_wooden_cabinet
  - level: LEVEL1
    task_name: EXTENSION_KITCHEN_SCENE25_stack_the_blue_bowl_on_the_green_bowl
```

草稿不写入 `runs/`、不加载 VLA、也不推进物理仿真；切换任务只重建预览，修改步数不会重复渲染。点击“取消草稿”或刷新页面会丢弃草稿。活动仿真期间不能创建或修改草稿。模型权重仍跨会话复用，但每次开始都会更新本会话的 `open_loop_steps`，不会沿用第一次加载模型时的旧值。

同一时刻只允许一个活动会话。状态依次为：

```text
LOADING → READY（人工接管倒计时）→ RUNNING → STOPPING → POSTPROCESSING → COMPLETED / ERROR
```

“停止”在下一个控制边界生效；若 GPU 正在同步推理，会先显示 `STOPPING`，等待这次 CUDA 调用自然返回，不会强行中断模型。停止和异常都尽量保存已经产生的部分轨迹及可诊断的 `run.json`。

实时预览不打开 MuJoCo 原生 Viewer。控制线程只写入最新 MuJoCo state，一个由固定线程长期持有的只读 MuJoCo 环境跨会话复用，并以 `agentview 512×512 / 10 fps` 渲染 JPEG；过期预览帧直接丢弃，因此不占用 20 Hz 控制循环。分支创建后先从父视频提取回溯帧作为 `resume_preview.jpg`，实时预览首帧产生前持续显示它，避免准备阶段黑屏。现有两个 CLI 仍遵循 `config.yaml` 的 `headless`，行为不变。

预览容器严格使用 `preview_width:preview_height` 的宽高比，默认与方形 `agentview` 完全匹配，不再用 `16:9` 裁剪或拉伸；桌面页面中的显示宽高按原容器的 50% 等比例缩小并居中，窄屏设备恢复为 100% 宽度。会话完成后，预览区自动切换为已保存的 `agentview` 视频播放器；拖动回溯时间轴会 seek 到对应视频时刻，播放或拖动视频也会同步时间轴和该帧的轨迹数值，不再为每次拖动跨线程调用 EGL 重渲染。

原始会话结束且后处理完成后，时间轴范围为 `0..state_count-1`。拖动时间轴会把所选 `sim_state` 直接恢复到只读环境，不会从第 0 帧重新执行。页面同时显示仿真时间、EEF XYZ `[m]`、axis-angle `[rad]`、双指夹爪 qpos、VLA raw action、实际环境 action 和成功状态。

点击“从此帧重新推理”会验证 MuJoCo state 最大恢复误差不超过 `1e-9`，创建空 action queue，从所选帧重新查询 VLA，并执行到源轨迹的实际结束帧。人工接管只支持 SpaceMouse，使用相同的精确恢复与源轨迹总长度规则。每个原始轨迹可产生多个并列分支；分支本身只读，不能再创建子分支。

UI 启动后只探测控制器，不占用动作输出。连接设备后顶部状态显示 `UNCALIBRATED`（已连接·待校准），点击“校准”并松开帽盖连续静止 2 秒即可。校准期间若任一轴超过 `neutral_max_abs`，静止进度会重置并提示松开帽盖，不再使分支进入 `ERROR`；30 秒内始终无法稳定才报告可重试的校准失败。一次成功校准会由全局控制器服务持续复用，设备拔插或 UI 服务重启后必须重新校准。

点击 SpaceMouse 接管后，后端依次显示“读取轨迹、加载环境、恢复状态、准备预览”，取得有效首帧后进入 `READY`，显示清晰的 `3、2、1` 倒计时；倒计时结束前控制器保持 disarmed，机械臂不会运动。开始接管后左键张开夹爪，右键闭合；位移和旋转增益可在创建分支前调整，也可在运行时通过 `0.05..1.0` 的滑杆实时更新。顶部控制器 pill 显示输入样本年龄：绿色 `<50 ms`，黄色 `50–249 ms`，红色 `≥250 ms`、断连或读取错误；红色状态下六维运动自动归零。接管会话的 WebSocket 断开会安全停止分支并保存已有轨迹。

`ui_config.yaml` 的 `legacy_scan_roots` 会在启动时递归索引已有 `trajectory_*.npz`：

- 现有 CLI 原始轨迹如果任务和 LEVEL 能映射到上述任务目录，可在 UI 中查看并创建一级分支；
- 现有 intervention 轨迹识别为分支，只读展示；
- 任务目录之外的 legacy 轨迹保留在历史列表，但不能用未知环境恢复或创建分支；旧目录不会被修改或迁移。

新会话按 `dataset-root/projects/libero_x_vla/runs/<task_name>/<YYYY-MM-DD>/<时间>__<session_id>/` 分组，不覆盖历史结果。`catalog.sqlite3` 只保存可重建的检索和成功率索引；run 目录仍是事实来源。`run.json` 是生命周期和安全删除所需的极简清单，`config.yaml` 固化任务、模型和控制参数，`summary.json` 只记录用户关心的结果与关键时序；轨迹、视频、图表和 SpaceMouse 采样统一放在 `episodes/episode_000/`。不再重复生成 `results.jsonl`、`trajectory.json`、`source_trajectory.json` 或 `spacemouse_device_summary.json`。完整回放 metadata 已内嵌在 `trajectory.npz`，逐步可读数据保留在 `trajectory.csv`。

创建分支时会立即把父轨迹控制数据物理复制为子目录中的 `source_trajectory.npz`，但不复制父 observations、视频或图表；因此父会话被删除后，子分支仍能独立恢复状态和绘制对比。原始会话生成轨迹图和 7 张 action 图；回溯分支不生成只包含二次推理的单独图表，只生成 7 张“完整原始轨迹 + 从回溯帧开始的二次推理/人工接管”action 对比图。若准备阶段尚未执行新动作就失败，仅保存精简轨迹、清单和 summary，不再重建整段视频、observations 或对比图。已有历史目录不会自动删除或迁移。

会话列表只为 `dataset_root` 项目运行树内、带匹配 `run.json` 标记的 UI 会话显示垃圾桶图标。删除需要再次提交会话 ID，并永久移除整个 run 目录，同时更新 SQLite 索引；活动仿真、控制器校准/接管期间禁止删除，CLI、intervention 和 legacy 数据不提供删除入口。删除原始会话不会级联删除已有分支。

结果文件链接使用浏览器内联打开：MP4、PNG、JSON、CSV 等由浏览器直接预览，并在新标签页显示；浏览器无法原生显示的 NPZ 等二进制格式仍会按浏览器自身规则保存。视频接口支持 HTTP Range，因此播放器可直接跳转到时间轴指定位置。

后端遵循 `API → RunService → SimulationWorker → Simulator / Policy / Recorder / Evaluator` 的单向依赖，React 不直接接触 MuJoCo，模拟器不写数据库，Recorder 不回调 UI。前端拆分为采集、运行记录、数据集与设置页面；采集页始终挂载，浏览数据时不会使活动 SpaceMouse WebSocket 意外断开。界面使用浅色半透明面板，不再使用深蓝渐变背景。详细边界见 `docs/ARCHITECTURE.md`，目录规则见 `docs/DATA_LAYOUT.md`。

当前 bootstrap 返回 `model_switching=false`、`task_switching=true`；任务控件使用已验证的目录，模型仍只读。以后加入新的 provider 后即可启用模型控件，不需要重写会话状态机。

主要接口：

```text
GET  /api/bootstrap
GET  /api/controller
POST /api/controller/calibrate
GET  /api/draft
POST /api/draft
PATCH /api/draft
DELETE /api/draft
GET  /api/draft/preview.jpg
POST /api/draft/start
GET  /api/sessions
GET  /api/runs
GET  /api/datasets/summary
DELETE /api/sessions/{id}
POST /api/sessions/{id}/stop
POST /api/sessions/{id}/branches
GET  /api/sessions/{id}/frames/{step}
GET  /api/sessions/{id}/frames/{step}/state
GET  /api/sessions/{id}/stream.mjpeg
GET  /api/sessions/{id}/artifacts/{name}
WS   /ws/sessions/{id}
```

开发与回归测试：

```bash
cd liberox-vla-adapter-terminal
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

cd frontend
npm test
npm run build
```

这里禁用 pytest 外部插件自动加载，是为了避免系统 ROS `launch_testing` 插件把 Python 3.12 的包注入 Python 3.10 conda 环境；不影响本项目自身测试。

## 4. 为什么这样适配

| 接口 | LIBERO-X | VLA-Adapter | 本模板处理 |
|---|---|---|---|
| 主相机 | `agentview_image` | `full_image` | 用 VLA-Adapter 官方函数转换 |
| 腕部相机 | `robot0_eye_in_hand_image` | `wrist_image` | 同上 |
| 状态 | EEF 位置 + 四元数 + 双指夹爪 | 8 维 POS_EULER 状态 | 四元数转 axis-angle，拼接双指位置 |
| 动作 | 7 维 EEF delta | 7 维连续动作 | 保持前 6 维；恢复夹爪符号 |
| 动作块 | 客户端默认最多执行 25 步 | checkpoint 生成 8 步 | 每次只执行 8 步再重规划 |
| 图像方向 | 官方 openpi 模板默认只上下翻转 | VLA-Adapter LIBERO 训练预处理旋转 180° | 沿用 VLA-Adapter `prepare_observation` |
| 归一化 | 评测环境不提供训练统计 | checkpoint 内 `norm_stats` | `stats_key: libero_object` |

`stats_key` 必须和 checkpoint 的训练数据匹配。用 `LIBERO-Object-Pro` 时是 `libero_object`（加载器会自动尝试 `_no_noops` 后缀）。这不是 LIBERO-X 的场景名。

## 5. 使用 LIBERO-X 训练数据微调

这一步不是跑预训练基线所必需的。官方 LIBERO-X 训练集是 LeRobot v2.1：2,520 条轨迹、889,277 帧、Parquet 数据，图像为 256×256，状态 8 维、动作 7 维、频率 10 Hz。VLA-Adapter 的 `finetune.py` 默认只接收 TFDS/RLDS，所以需要格式桥接。

### 5.1 下载并检查 LeRobot 数据

```bash
huggingface-cli download meituan/LIBERO-X \
  --repo-type dataset \
  --local-dir ./data/liberox_lerobot

pip install pyarrow  # 仅数据 schema 检查需要，仿真评测不依赖
python liberox-vla-adapter-terminal/scripts/inspect_lerobot_schema.py \
  ./data/liberox_lerobot
```

完整数据约 102 GB。先确认磁盘空间；若只做现成任务的仿真评测，不需要下载它。

### 5.2 转为 RLDS 时的目标 schema

每条 RLDS episode 应包含 `steps`，每个 step 至少含：

```text
observation.image          uint8[256,256,3]
observation.wrist_image    uint8[256,256,3]
observation.state          float32[8]
action                     float32[7]
language_instruction       string
is_first/is_last/is_terminal
```

转换规则：

1. 以 `episode_index` 分组，并按 `frame_index` 排序；
2. 通过 `task_index` 从 `meta/tasks.jsonl` 取语言指令；
3. 保留 `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`；
4. 若要与标准 LIBERO 配方一致，过滤前 6 维接近零且夹爪不变的 no-op 帧；
5. 将数据集命名为 `liberox_pickplace_no_noops`；
6. 输出目录必须形如 `data/liberox/liberox_pickplace_no_noops/1.0.0/*.tfrecord`；
7. 在 VLA-Adapter 的 `configs.py`、`transforms.py` 和 `mixtures.py` 注册新名称；本模板的 `patches/vla_adapter_liberox_registry.patch` 提供最小补丁。

应用注册补丁：

```bash
cd VLA-Adapter
git apply ../liberox-vla-adapter-terminal/patches/vla_adapter_liberox_registry.patch
```

先用 1 条 episode 构建小型 TFDS 数据集并初始化 `RLDSDataset`；通过后再转换全部数据。数据统计会由 `finetune.py` 写入 checkpoint，评测自训练模型时在 `config.yaml` 中改为：

```yaml
checkpoint: /path/to/your/merged/checkpoint
stats_key: liberox_pickplace_no_noops
use_pro_version: true
```

### 5.3 微调命令骨架

```bash
cd VLA-Adapter
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir ../data/liberox \
  --dataset_name liberox_pickplace_no_noops \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --max_steps 20005 \
  --num_steps_before_decay 20000 \
  --save_freq 5000
```

`20k` 只是小规模验证起点，不是论文复现配置。先确认 loss、动作统计和 1 个任务 rollout 正常，再根据数据子集大小与显存调整。

## 6. 常见故障

| 现象 | 最可能原因 | 处理 |
|---|---|---|
| `eglQueryString` / 无法创建 offscreen context | EGL 库或 `MUJOCO_GL` 缺失 | 安装 Mesa/EGL 包并设置 `MUJOCO_GL=egl` |
| `torch` 被降到 1.11 | 安装了 LIBERO-X 全量 requirements | 重建环境；LIBERO-X 用 `pip install -e ... --no-deps` |
| checkpoint 能加载，但动作全零或乱跳 | 图像方向、prompt、`norm_stats` 不匹配 | 保持本模板预处理，并核对 `stats_key` |
| `proprio` 不是 8 维 | 状态拼接错或环境包导入错 | 检查是否导入 LIBERO-X 的 `libero` 包 |
| action 不是 7 维 | 常量误识别为 ALOHA/双臂 | 启动脚本名需含 `libero`，日志应显示 action dim 7 |
| 成功率低但 rollout 正常 | 标准 LIBERO checkpoint 对 LIBERO-X 有分布偏移 | 这是预期的鲁棒性差距；再做 LIBERO-X 微调 |
| 结果不稳定 | 初始状态、seed、依赖版本不一致 | 固定 `.init`、seed、commit 和环境版本 |

## 7. 建议的验收顺序

1. `env_only: true` 通过；
2. 单次 rollout 无异常并生成视频；
3. 10 次固定初始状态得到基线成功率；
4. 将同一任务复制到 LEVEL2/3/4，保持 checkpoint 和随机种子不变；
5. 最后才做 LeRobot→RLDS 转换和微调；
6. 对比微调前后各 Level 成功率，避免只报告训练分布上的结果。

## 8. 对应上游

- VLA-Adapter：https://github.com/OpenHelix-Team/VLA-Adapter
- LIBERO-X：https://github.com/meituan/LIBERO-X
- LIBERO-X 数据：https://huggingface.co/datasets/meituan/LIBERO-X
- VLA-Adapter Object-Pro：https://huggingface.co/VLA-Adapter/LIBERO-Object-Pro
