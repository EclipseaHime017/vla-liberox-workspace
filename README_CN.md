# LIBERO-X × VLA-Adapter Terminal

当前版本：**v0.6.0**

本版主要内容：

1. 融合 Stage 与 RynnValue 奖励，通过 α、κ 配置相加或 chunk-based 相乘的 Final Reward。
2. 统一数据集评价配置，支持复用已有输出重算 Final Reward 或串行执行 All，保护原始评价与历史训练。
3. 详情分别显示 Original Final Reward 三组件和融合后的 Final Reward，训练不再单独选择奖励类型。

这是一个面向 Franka/LIBERO-X 的本机仿真、VLA 评测、轨迹回溯、SpaceMouse / FACTR 人工接管与数据管理终端。UI 按「任务 → 难度 → 提示词」提供三个任务族的 LEVEL1–4 官方变体，暂不开放 LEVEL5；下文保留 LEVEL1 黑碗任务作为 CLI 配置示例。实体控制器能力仍需在连接对应硬件后验收。

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

# 安装前端依赖并完成首次构建。之后 run_ui.py 会在源码变化时自动重建。
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

评测脚本默认还会从 `configs/config.yaml` 设置 `MUJOCO_GL=egl`；上面的 `export` 适用于其他 MuJoCo 程序，或在配置中将 `mujoco_gl` 设为 `null` 时使用。

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

评测参数集中在仓库根目录的 `configs/config.yaml`。配置文件内部的相对路径均以 `configs/` 为基准；下文的相对脚本命令则假设当前目录为 `vla-liberox-workspace`。先进入工作区：

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
disabled_policy_cameras: []
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

`env_resolution` 控制送给 VLA 和写入观测轨迹的 `agentview`/腕部相机，保持 `256×256` 以维持模型输入契约。`disabled_policy_cameras` 默认为空；可填写 `[agentview]` 或 `[robot0_eye_in_hand]` 做单路输入消融，但不能同时关闭两路。禁用只在进入 VLA 前将该槽替换成同尺寸黑帧，录像与 observation 仍保留原图。默认录像模式 `vla_views` 会把这两路原始策略观测直接水平拼接：左侧是 `agentview`，右侧是 `robot0_eye_in_hand`，不做高清重渲染或缩放，最终保存为 `episode_000_{success|failure}_vla_views.mp4`（`512×256`）。按照 VLA-Adapter 官方训练契约，两路输入必须旋转 180°，因此它可能与 MuJoCo Viewer 的自然显示方向不同，不能为了观感而改动；模型内部仍会按原路径将每路图像处理为 `224×224`。

每次评测的标准文件 `episode_000_{success|failure}.mp4` 固定为 `1024×1024` 高清主视角。它使用同一个 `agentview` 相机位姿，但只校正 OpenGL framebuffer 的上下方向，不应用 VLA 训练所需的额外水平翻转，因此是供人查看的自然方向。`main_view_video_width` 和 `main_view_video_height` 只控制录像，必须保持相等，不会改变模型观测。`results.jsonl` 中的 `video` 和 `main_view_video` 都指向该标准文件，`vla_views_video` 指向保持模型输入方向的双相机拼接文件。

`headless: false` 会把当前 LIBERO 环境的同一个底层 `mujoco.MjModel/MjData` 交给 `mujoco.viewer.launch_passive()`，打开 MuJoCo 原生交互窗口。Viewer 初始选择与标准视频相同的 `agentview` 固定相机；可在右侧 Camera 面板切换为 `Free` 后旋转、平移和缩放。代码默认关闭 robosuite 的 collision geom group 0、保留 visual geom group 1，避免碰撞简化体与机械臂视觉网格重叠产生绿色/黄色块和闪烁；仍可在 Viewer 面板中手动重新打开。原始评测和 intervention 都支持该模式，VLA 离屏输入和录像不受 Viewer 自由相机操作影响。改回 `headless: true` 即不创建窗口。非 headless 模式需要本地图形桌面、X11 转发或可用的 Wayland/XWayland 会话。

原生 Viewer 以 passive 模式连接，只查看当前仿真，不自行推进物理状态；脚本在恢复状态和每个控制步后调用 `sync()`。当前评测是单进程的，因此 VLA 正在生成下一组 action 时画面可能短暂停留在上一状态，推理完成后会继续更新；这不表示仿真停止。关闭 Viewer 只会关闭观察窗口，rollout 会继续；需要提前结束推理时仍在运行终端按 `Ctrl+C`。

为避免相机渲染破坏 50 ms 控制预算，实时 `env.step()` 不再无条件生成两路图像：只在发起 VLA 查询时按需采集策略输入。完整观测序列和两份视频在 rollout 结束、Viewer 关闭后，依据已记录的 MuJoCo state 逐帧重建。因此动作结束后终端还会出现 `Post-processing ... recorded states`，这是在生成结果，不是再次执行策略；高清编码耗时不会改变刚才的仿真节拍。

也可以把 `video_camera` 改为 `frontview`、`birdview`、`sideview` 或 `galleryview`，此时 `video_width`/`video_height` 是该单相机的渲染分辨率。`birdview` 覆盖范围最大，`frontview` 更适合观察机械臂整体运动；宽高有效范围均为 `64..2048`。

脚本强制读取 `configs/config.yaml`，不接受其他配置地址或运行参数。需要调整实验时直接编辑该文件，运行命令只有一行：

```bash
python liberox-vla-adapter-terminal/scripts/eval_pickplace_direct.py
```

### 3.1 只验证场景、初始状态和观察量

在 `configs/config.yaml` 中设置：

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

在 `configs/config.yaml` 中设置：

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

需要扩大到 10 回合时，把 `configs/config.yaml` 改为：

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

先至少重新运行一次 3.2 或 3.3，生成上述 `trajectory_*.npz`。然后编辑固定的 `configs/intervention_config.yaml`：

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

两个 YAML 都使用 `open_loop_steps` 表示“每次 VLA 预测 8 个 action 后，实际连续执行其中多少个”。字段名称和有效范围 `1..8` 完全一致，但两个文件的值彼此独立：`configs/config.yaml` 控制原始评测，`configs/intervention_config.yaml` 控制回溯后的分支推理。

干预结束后会根据完整 state 序列流式生成两份视频，播放帧率统一继承 `configs/config.yaml` 的 `video_fps`，并与 `control_hz` 相等：

- `intervention.mp4`：沿用 `configs/config.yaml` 的 `video_camera`、`video_width` 和 `video_height`；默认是未经缩放的 `512×256` VLA 双视角拼接，左主视角、右腕部视角；
- `intervention_agentview_hd.mp4`：沿用 `main_view_video_width` 和 `main_view_video_height`，使用与推理主图相同、位置更高的 `agentview` 相机，默认 `1024×1024`。

两份视频的原轨迹前缀和新分支都依据保存的 MuJoCo state 在控制结束后重新渲染，因此帧数、回溯点和相机位姿严格对齐。视频采用流式编码，不会把两组高清帧同时堆积在内存中，也不会在人工/VLA 控制期间阻塞 20 Hz 循环。

intervention 同样读取 `configs/config.yaml` 的 `headless`：为 `false` 时打开 MuJoCo 原生 Viewer，并从 `resume_step` 恢复点开始同步当前新分支，不会把回溯点之前的历史前缀快速播放一遍；历史前缀仍会正常写入两份最终视频。

运行命令同样不接受配置地址：

```bash
python liberox-vla-adapter-terminal/scripts/intervene_pickplace.py
```

脚本会恢复指定 state，而不是从头近似执行旧动作。恢复后清空旧 action chunk，并按以下模式产生一条新分支：

- `policy`：从指定点重新查询 VLA。`open_loop_steps` 与 `configs/config.yaml` 中的字段同名；设为 `1` 时每一步都重新规划；
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

随后拔下并重新插入 SpaceMouse（只 reload 规则不会修改已经存在的 hidraw 节点）。测试脚本固定读取 `configs/spacemouse_test_config.yaml`，不接受配置地址：

```bash
cd vla-liberox-workspace
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/test_spacemouse.py
```

首次诊断时设置 `mode: device`，只验证 HID，不导入 MuJoCo、LIBERO 或 VLA。静止校准期间不要触摸帽盖；设备在完全静止时不发送新报告属于正常情况，此时使用已初始化的零状态，随后的覆盖测试仍会验证真实 HID 报告。倒计时后依次让六个轴向正负两个方向运动并按下左右键。终端显示 raw 输入和最终 OSC_POSE command，运行结束后检查 `summary.json` 的 `functional_check_complete` / `acceptance_passed` 与 `device_summary.json` 的轴/按钮覆盖率。未完成覆盖或性能验收时脚本以状态码 `2` 结束，运行错误使用状态码 `1`。

确认设备读取正确后，将 `configs/spacemouse_test_config.yaml` 改为：

```yaml
mode: simulation
```

再次运行相同命令即可在当前 `configs/config.yaml` 的 LEVEL、任务、seed 和第一个 benchmark init state 中控制机械臂。该模式不加载 VLA：

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

#### 3.6.1 FACTR Franka 校准、手动重力补偿与无 VLA 测试

GUI 与独立测试共用固定版本的官方 FACTR 类、驱动及参数，不再保留自写重力公式、重复 URDF 或另一套被动 GUI 采样器。未修改的源码在 `third_party/FACTR_Teleop/`，commit 为 `7a07ab3629af03a91c0198df7c44a0082dca77ab`，版本与执行文件哈希由 `configs/factr_official.lock.json` 校验。下载目录与隔离环境不提交 Git。

从 workspace 根目录安装、检查（这两条命令不打开设备）：

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/setup_factr.py
python liberox-vla-adapter-terminal/scripts/setup_factr.py --check
```

系统 Python 需先具备 ROS 2、Pinocchio、NumPy、PyYAML、pyzmq，本机为 ROS 2 Jazzy。其他机器参考[官方安装说明](https://github.com/JasonJZLiu/FACTR_Teleop#installation)并 source ROS 环境。脚本只在隔离环境安装固定 SDK/pyserial，不把系统 Python 3.12 的 ROS 库装入 Python 3.10 的 VLA 环境；通过本地父子进程通信。检查失败不会回退自写后端。

固定配置为 `configs/factr_test_config.yaml`，GUI/CLI 共用设备参数，按顶层 USB VID/PID 与可选序列号发现串口；`runtime` 中的路径相对 YAML：

```yaml
vendor_id: 0x0403
product_id: 0x6014
serial_number: null
runtime:
  upstream_root: ../third_party/FACTR_Teleop
  runtime_python: ../third_party/factr-runtime/bin/python
  calibration_file: ../runs/factr_calibration/calibration.json
```

发现过程只读枚举 USB 串口元数据，不打开串口、不启动官方工作进程、不自动接管或上力。必须恰好匹配一个设备；相同 VID/PID 下有多个设备时，将 `serial_number` 设为目标设备的 USB 序列号。无匹配或匹配不唯一时，选择 FACTR 后可在设置区看到具体诊断。工作进程启动时重新解析当前串口，不再配置固定 `/dev/ttyUSB*` 或 `/dev/serial/by-id/...` 路径。

USB 连接检测独立于官方运行环境检查：即使隔离环境缺失，也能显示实体设备已连接及运行环境诊断；校准和控制仍需先完成上述 FACTR 隔离环境安装。VLA 环境中的串口枚举依赖 `requirements-ui.txt` 的 `pyserial==3.5`；更新部署时需同步安装该文件。GUI 设备重连后需重新校准，重力补偿不会自动恢复。CLI 可加载身份匹配的有效校准；没有 USB 序列号时不复用持久校准。旧路径指纹的校准文件需要重新采集一次。

旧 `standalone` 段改名为 `runtime`；删除旧 `device_path`、被动采样 `poll_hz`、多阶段校准和测试录制参数。部署时同步 USB 选择器配置。SpaceMouse 是环境安装的 `pyspacemouse==2.0.0`，无需复制到 `third_party`。

**GUI 流程：**选择 FACTR → 整臂置于[官方 Figure 1](https://github.com/JasonJZLiu/FACTR_Teleop/blob/7a07ab3629af03a91c0198df7c44a0082dca77ab/src/factr_teleop/README.md#initialization-settings)参考构型并松开触发器 → 点击一次“校准控制器” → 点击“开启重力补偿” → 回溯接管。

原“校准控制器”按钮会先检查 USB latency：已经是 1 ms 时直接校准，不提权；否则在本机桌面弹出 **polkit 系统授权窗口**，密码仅由系统认证代理收集，不经过网页或 API。授权成功后安装按 VID/PID 匹配的永久规则，并立即应用到当前唯一匹配设备，无需重插，然后继续校准；不会自动上力。拒绝、取消或超时都不启动校准，没有额外的“修复”按钮。此流程需要本机桌面会话、`pkexec` 和 polkit 认证代理；SSH/headless 或远程浏览器请使用下方终端备用命令，不在网页输入 sudo 密码。

校准直接调用官方 `_get_dynamixel_offsets`（预热十包，按 π/2 候选选偏置），同时捕获松开触发器零点，使用官方 0.8 rad 行程；不再分别采集开/闭端点。这仍需要近似参考构型，**不是将任意姿态当成 Panda 物理零位**，也不逐电机标定。只保存主机文件，不写舵机 Homing Offset；旧后端校准需重新采集。GUI 启动需显式校准，不自动上力。

当前仅支持官方 **Figure 1** 作为校准参考，七轴角度固定为 `[0, -0.7854, 0, -2.356, 0, 1.57, 0]` rad。`configs/factr_test_config.yaml` 的 `reference_joint_positions` 必须保持该值；不能用折叠或任意摆放姿态代替。校准不修改电机 Homing Offset、控制参数或 `third_party/FACTR_Teleop/` 源码。

开启补偿后，运动、倒计时、正常完成/停止仿真和回溯之间持续保持，可在接管中手动关闭。补偿开启时锁定控制器选择与重新校准，避免隐藏仍上力的设备。**仿真异常、设备故障、后端关闭时先停止实体输出，再做数据后处理或等待仿真线程。**重连不自动恢复。关闭浏览器不等于关闭后端；离开前请手动关闭补偿或退出后端。通信损坏导致无法确认 OFF 时显示“撤力未确认”，操作者需支撑主臂并检查实体电源，不能假定已经撤力。

控制参数来自上游 `grav_comp_demo.yaml`：目标 500 Hz、增益 0.85，以及官方摩擦补偿、零空间调节、关节限位屏障。无额外 2 rad/s 速度阈值、渐入/软电流限制或 120 秒时限；保留设备电流上限、编码器不连续检测、七轴 RAM watchdog 与故障撤力。触发器电机始终 torque OFF。这些不是第二套动力学公式，也不能保证断电后悬停。关闭补偿前请支撑主臂。

**硬件准备：**核对结构、型号、ID 1–8、方向、4 Mbps 与官方一致，current mode、Return Delay Time=0，官方 USB latency=1 ms。不会自动修改端口权限、舵机模式、EEPROM 或杀占用进程；USB latency 仅经上述系统授权或下方终端命令修改。

**终端备用：永久设置 USB latency（每台 PC 执行一次，CLI 流程不变）：**

```bash
python liberox-vla-adapter-terminal/scripts/setup_factr.py --install-usb-rule
```

命令只根据 YAML 的 USB 选择器安装 udev 规则并重新加载规则，安装时会请求 sudo；不安装运行环境、不打开串口、不校准或上力，也不自动触发设备事件。`serial_number: null` 时规则对**所有相同 VID/PID 的设备**生效，因此其他 `0403:6014` FTDI 转接器也会设置为 1 ms；只有在 YAML 明确填写序列号时才限定单台设备，不自动绑定当前设备序列号。规则处理 `add|bind` 事件，无需固定 `ttyUSB0`，重插或重启后自动应用。终端命令默认不应用当前连接，安装后先退出控制程序、支撑实体主臂，再重新插拔 USB；GUI 授权则同时应用当前唯一匹配设备，无需重插。更改 USB 选择器后需更新规则。

临时应急才使用下面的手动写入，重插后可能失效。关闭所有控制程序，确认真实设备名后检查/设置：

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# ttyUSB0 必须换成本设备真实名称
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

1 ms 为 USB 参数，不代表实测达到 500 Hz，状态中的 `cycle_ms` 用于检查。GUI、CLI、上游 demo、厂商工具不能同时占用串口。

独立测试固定读取同一 YAML：`mode: device` 不加载 MuJoCo/VLA，`simulation` 为无 VLA 仿真。

```bash
conda run --no-capture-output -n vla-liberox python liberox-vla-adapter-terminal/scripts/test_factr.py
```

| 输入 | 操作 |
| --- | --- |
| `c` | 松开触发器，一次官方整臂校准 |
| `s` | 开始设备测试或无 VLA 仿真 |
| `g` | 启用补偿，不再输入 ENABLE |
| `d` | 直接关闭补偿，无二次确认；操作前支撑主臂 |
| `i` | 查看版本、补偿、样本年龄和循环耗时 |
| `q` / Ctrl+C | 先关闭电机输出，再退出整个测试程序，无二次确认 |

`FACTR>` 表示等待输入，后台持续补偿。正常完成测试/关闭 Viewer 返回菜单不撤力；Ctrl+C、错误或退出停止输出。独立测试不保存轨迹、CSV、视频或运行目录，仅持久化校准。

仿真控制：校准后的主臂七关节角 → Panda 绝对关节位置目标 → MuJoCo 关节位置控制器（仍计算物理动力学，不直接覆盖 qpos）。七关节按 1:1 目标跟随，不再使用相对末端偏移，也不使用位移/旋转增益。夹爪保持源指令，触发器先对齐再切换。

接管流程：先开启补偿 → 回溯后点击 FACTR 接管 → **仿真保持静止，实体主臂缓慢靠近仿真姿态** → 对齐后保持姿态并倒计时 3 秒 → 释放对齐保持，开始关节跟随。请给实体主臂留出运动空间。关闭补偿或停止仿真可取消准备。对齐目标必须在官方关节范围内，不做任意置零或角度夹断。

对齐使用官方关节位置 PD 增益，但改为非阻塞逐周期执行，并保留官方重力、摩擦及关节限位项；不是直接调用官方阻塞式 `set_leader_joint_pos`。只在准备阶段将参考目标以最多 0.15 rad/s 移动，限制跟踪误差积累为 0.12 rad；误差 ≤0.08 rad、速度 ≤0.15 rad/s 持续 0.5 秒才进入倒计时，60 秒未完成则报错。官方文档对该 PD 要求至少 200 Hz；项目在启动和运行对齐时按最近 50 个周期的平均频率检查，不能把它解释成每个周期绝不超过 5 ms。单次超过 5 ms 时只暂停该周期的对齐 PD 和参考推进，保持普通补偿，不退出；持续窗口低于 200 Hz 或硬件看门狗超时仍停止。这些不是人工遥操作的速度或补偿时长限制。实体对齐效果仍需实机确认。

退出诊断将“运行故障”和“电机 OFF 校验”分开：非零退出码不代表撤力失败。子进程显式传回关闭寄存器读回结果，只有关闭校验失败或没有收到确认才显示关闭未确认；重复的 RuntimeError 前缀不再层层叠加。

如果校准时立即出现连接重置，先查看具体的子进程启动错误。父进程会优先读取退出消息，并从内存中的有界 stderr 尾部提取启动异常，不额外保存测试日志。未安装上述 udev 规则时，USB 拔插后 `latency_timer` 可能恢复为 16 ms；当前启动检查要求 1 ms。本机 GUI 校准可通过系统授权修复；CLI 则安装规则、关闭 FACTR 程序并支撑主臂，重插后读回确认，再重新校准。只看到 USB 枚举成功不代表子进程启动检查已经通过。

运行中 `communication failed: -3001` 表示 DYNAMIXEL 状态包接收超时（`-3002` 为坏包），不等同于整条 USB 设备被拔出。接入层允许一次完整同步重读，丢弃残留接收数据；成功后才发布新关节包，不复用旧反馈。与官方默认最多十次重试不同，这里总读取预算为 75 ms，首读已用掉半数预算时不再重试；连续失败、完整包缺失或超时仍停机。硬件 100 ms watchdog 不变，发送电流前也检查循环是否已超时。`GET /api/controller?controller_id=factr` 的 `serial_read` 提供失败/恢复累计次数和最近读取耗时。偶发恢复会提示 `sync read recovered`；若频繁发生，需要排查线缆、接口、供电和系统负载，不能仅靠不断增加重试掩盖。

**FACTR 与 SpaceMouse 使用相同的人工接管记录、结果展示和训练入口。** GUI 正常创建结果目录、复制父轨迹并保存完整前缀和新接管后缀、双视角视频、同步 observation 及末端轨迹；不再提供“仅控制、不记录”的临时会话。FACTR 保持七关节跟随，每个 20 Hz 控制步结束后，用实际仿真末端的世界坐标位移及相对旋转反算归一化七维 OSC 动作标签，夹爪仍为 `-1` 打开、`+1` 闭合，`action_source=human`。不新增关节示教日志，现有 `sim_state` 仍用于准确恢复物理状态。末端增量标签是实际运动的重标注，不声称与关节控制的驱动力学严格等价，也不另设训练数据类别。

`raw_action` 保留未裁剪的转换结果，`env_action` 限于现有 `[-1, 1]` 范围供训练读取；超限次数、比例和轴计数保存在 controller 诊断中，终端会提示，不静默丢弃超限信息。视频与 observation 从真实记录的仿真状态重建，不使用反算 action 重放。正常结束只 disarm 仿真，保留补偿和校准；设备/仿真错误或后端退出仍 disable。成功后继续记录至停止或步数上限，故障则保存已完成部分。独立 `test_factr.py` 仍为不保存数据的设备测试。训练/reward/server 不变。

前端改动后需重新构建并重启后端：

```bash
cd liberox-vla-adapter-terminal/frontend
npm ci
npm run build
```

### 3.7 仿真与干预 Web UI

Web UI 将原始仿真、实时查看、结束后的逐帧回溯、VLA 重新推理、人工接管和结果下载整合到同一个本机页面。后端固定监听 `127.0.0.1:8000`，不开放局域网，也不包含认证功能。

从 `vla-liberox-workspace` 启动：

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

然后浏览器打开 `http://127.0.0.1:8000`。

`frontend/dist` 是被 Git 忽略的本机构建产物。`run_ui.py` 会记录前端源码指纹：执行 `git pull` 后若 React、CSS、Vite 配置或依赖清单发生变化，启动时会先在终端显示即将执行的完整目录和 `npm run build` 命令，再自动更新静态资源；没有变化时也会显示当前源码指纹和“已是最新版本”。新电脑仍需先在 `frontend/` 执行一次 `npm ci`；如果依赖缺失，启动脚本会直接给出该命令并停止，而不会静默使用旧页面。入口 HTML 使用 `Cache-Control: no-store`，重启后普通刷新即可取得新构建。

前端有三种明确的更新途径：

```bash
# 1. 日常自动更新：拉取代码后直接重启，源码变化时自动构建
git pull
python liberox-vla-adapter-terminal/scripts/run_ui.py

# 2. 手动重建：只重新生成 frontend/dist，不更新 npm 依赖
cd liberox-vla-adapter-terminal/frontend
npm run build

# 3. 新电脑或 package-lock.json 已变化：严格按锁文件重装并构建
npm ci
npm run build
```

`npm run build` 和 `npm test` 不会安装或升级依赖；只有 `npm ci` 会根据已提交的 `package-lock.json` 重建本机 `node_modules`。项目不使用隐式 `npm update`。

如果更新后页面外观仍像旧版本，不要继续用强制刷新猜测。新界面顶部会显示 `UI <12位指纹>`，并可直接检查当前占用 8000 端口的后端实际提供了哪个目录和 bundle：

```bash
curl -s http://127.0.0.1:8000/api/build-info | python -m json.tool
```

如果该接口返回 `404`，说明端口上仍是旧后端进程；如果 `current` 不是 `true`，则源码与 `dist` 不一致。启动终端也会打印项目绝对目录、相同的 UI 指纹和具体静态资源文件名，便于发现从旧仓库目录启动的进程。

`lsusb` 识别到正确的 `256f:c63a` 只证明 USB 层发现了设备，不等于运行 UI 的 Conda 环境能够通过 HIDAPI 读取它。以下接口会返回缺失依赖、HID 枚举错误和匹配到的 `hidraw` 节点；新版界面也会直接显示探测失败原因，而不再统一写成“未连接”：

```bash
curl -s http://127.0.0.1:8000/api/controller | python -m json.tool
```

确认 UI 使用的同一个环境安装了 `requirements-spacemouse.txt`，并按 3.6 节安装精确 udev 规则、重新加载规则后拔插设备。无需修改 VID/PID。

UI 仍读取 `configs/config.yaml` 中的 checkpoint、seed、相机和 20 Hz 控制设置。`configs/ui_config.yaml` 的 `task_families` 显式定义三个实际目的，不以颜色或提示词措辞拆成不同任务：

- `place the bowl on the stove`；
- `open the top drawer of the wooden cabinet`；
- `stack the bowls`。

随后选择难度和真实提示词，例如 `place the bowl on the stove → LEVEL1 → place the black bowl on the flat stove`。LEVEL4 同一任务内可再选青色碗或大浅灰碗；最终仍提交唯一原始 `task_id`，不改变轨迹身份、训练提示词或数据存储格式。

| 难度 | 资源与变化 | 当前选择数 |
|---|---|---:|
| LEVEL1 | 局部空间扰动；原有三个任务 | 3 |
| LEVEL2 | 扩展空间扰动；同名任务对应 LEVEL2 BDDL/init | 3 |
| LEVEL3 | 场景布局重构；按官方 T7/T70/T133 对应任务 | 3 |
| LEVEL4 | 物体视觉属性变体；放炉任务为青色碗/大浅灰碗两个版本，叠碗为青色碗，抽屉保留真实原任务提示词 | 4 |

等级含义见 [LIBERO-X 官方说明](https://meituan.github.io/LIBERO-X/)。LEVEL5 资源仍保留在 `LIBERO-X/libero/libero_x/LEVEL5/`，它是复用 LEVEL4 场景的语言改写评价，不是另一个物理难度场景；本阶段从默认选择和检索选项中移除，不删除上游文件或历史数据。

当前共 13 个场景，本地每个场景有 10 个 init states。仿真和测试配置使用三个关联选择框，切换上级会同步选择有效下级；活动仿真和分支仍锁定场景。数据检索也使用相同顺序，允许停在「全部任务／全部难度／全部提示词」；跨场景筛选在分页之前执行，计数与结果一致。创建数据集、轨迹评价和 ZIP 导出仍需选定具体提示词，训练可从检索结果中选择一个冻结数据集。缺失可选资源禁止创建仿真，但历史数据仍可检索。

所有任务继续使用所选基础策略或 overlay，不附加 `experimental_ood` 标签，不改变环境 done、连续成功阈值或成功率统计逻辑。扩展难度不意味着已测得策略成功率；本机已完成资源验证和短无模型场景烟测。seed 和初始状态索引的说明保留在文档，表单下方不再显示说明段落；数值范围校验不变。

新建原始仿真采用明确的两阶段流程：先点击“创建仿真”，在内存草稿中切换任务、调整以下参数并检查所选 init state 的静态预览；只有预览就绪后，“开始仿真”才会创建唯一结果目录并加载策略：

- `max_steps`：本次总控制步数；
- `open_loop_steps`：每次 VLA 预测 8 个 action 后实际执行的数量，有效范围 `1..8`；
- `seed`：本次运行上下文的随机种子，有效范围 `0..2147483647`，默认 0；策略会话设置 Python/NumPy/Torch RNG，环境 reset 也使用此值，但随后恢复明确选定的 benchmark state，因此改变 seed **不保证画面改变**，也不会继承训练的 seed=7。不同训练 seed 通过学到的权重影响结果，不通过 overlay 恢复 RNG。各类选择、划分、训练和测试 seed 分别由对应流程的配置控制；
- `init_state_index`：选择当前任务预先保存的 benchmark 初始状态，采用从 0 开始的索引。UI 根据任务实际状态数动态显示范围 `0..N-1`，切换任务时自动重置为 0，越界值会在创建会话前被后端拒绝；
- `VLA 摄像头输入`：可关闭 `agentview` 或 `robot0_eye_in_hand` 中的一个做视觉消融。关闭后对应的固定输入槽传入同尺寸黑帧，避免破坏 Object-Pro 的双图像结构；至少保留一个摄像头。四视角实时预览、轨迹 observation 和两路录像仍保存未经遮挡的原图。

UI 专属参数固定从 `configs/ui_config.yaml` 读取，启动命令不接受配置地址：

```yaml
host: 127.0.0.1
port: 8000
dataset_root: ../dataset-root
policy_registry: ../policy-registry
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
  # 完整 LEVEL2–4 目录及三个任务分组见 configs/ui_config.yaml。
task_families:
  - family_id: bowl_on_stove
    label: place the bowl on the stove
    task_names:
      - EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove
      - EXTENSION_KITCHEN_SCENE11_LEVEL3__T70_place_the_black_bowl_on_the_flat_stove
      - EXTENSION_KITCHEN_SCENE11_LEVEL4__T70__A1_place_the_cyan_bowl_on_the_stove
      - EXTENSION_KITCHEN_SCENE11_LEVEL4__T70__A2_place_the_large_light_grey_bowl_on_the_stove
```

草稿不写结果目录、不加载 VLA、也不推进物理仿真；切换具体场景、seed 或 `init_state_index` 会重新生成预览，修改步数或 VLA 摄像头输入不会重复渲染。目录只在启动时校验小型 BDDL/提示词/init，后续查询复用缓存；MuJoCo 按需加载。相同物理场景和 seed 复用预览环境，其他场景安全关闭后重建，旧异步预览不能覆盖新任务。点击“取消草稿”或刷新页面会丢弃草稿。活动仿真期间不能创建或修改草稿。模型权重跨会话复用，每次开始更新 `open_loop_steps`、seed 和视觉消融设置。分支继承父会话的完整任务 ID、seed、init 索引、策略和摄像头输入。

同一时刻只允许一个活动会话。状态依次为：

```text
LOADING → READY（人工接管倒计时）→ RUNNING → STOPPING → POSTPROCESSING → COMPLETED / ERROR
```

“停止”在下一个控制边界生效；若 GPU 正在同步推理，会先显示 `STOPPING`，等待这次 CUDA 调用自然返回，不会强行中断模型。停止和异常都尽量保存已经产生的部分轨迹及可诊断的 `run.json`。

实时预览不打开 MuJoCo 原生 Viewer。控制线程只写入最新 MuJoCo state，一个由固定线程长期持有的只读 MuJoCo 环境跨会话复用，并以 10 fps 生成 `2×2` 操作视图：主视角、腕部视角、`−45°` 斜视角和 `+45°` 斜视角，单格渲染分辨率为 `512×512`，桌面 UI 按 `512×512` 合并窗口显示。两个斜视角只存在于内存中的实时 MJPEG，不写入 trajectory、observation、视频或图表；VLA 推理仍严格只使用原有的 `agentview + robot0_eye_in_hand`。过期预览帧直接丢弃，因此不会阻塞 20 Hz 控制循环。分支创建后先从父视频提取回溯帧作为 `resume_preview.jpg`，实时四视角首帧产生前持续显示它，避免准备阶段黑屏。现有两个 CLI 仍遵循 `configs/config.yaml` 的 `headless`，行为不变。

预览容器严格使用 `preview_width:preview_height` 的宽高比，默认与方形 `agentview` 完全匹配，不再用 `16:9` 裁剪或拉伸；桌面页面中的显示宽高按原容器的 50% 等比例缩小并居中，窄屏设备恢复为 100% 宽度。会话完成后，预览区自动切换为已保存的 `agentview` 视频播放器；拖动回溯时间轴会 seek 到对应视频时刻，播放或拖动视频也会同步时间轴和该帧的轨迹数值，不再为每次拖动跨线程调用 EGL 重渲染。

原始会话结束且后处理完成后，时间轴范围为 `0..state_count-1`。拖动时间轴会把所选 `sim_state` 直接恢复到只读环境，不会从第 0 帧重新执行。页面同时显示仿真时间、EEF XYZ `[m]`、axis-angle `[rad]`、双指夹爪 qpos、VLA raw action、实际环境 action 和成功状态。

点击“从此帧重新推理”不会立即启动分支，而会先在左侧打开二次推理配置。源任务、`episode_000`、回溯帧和原轨迹结束步均锁定为灰色，只允许调整 `open_loop_steps`（每次预测后实际连续执行的步数，范围 `1..8`）；点击“开始二次推理”后才真正创建分支。后端随后验证 MuJoCo state 最大恢复误差不超过 `1e-9`，创建空 action queue，从所选帧重新查询 VLA，并执行到源轨迹的实际结束帧。人工接管可选择 SpaceMouse 或 FACTR，使用相同的精确恢复、源轨迹总长度及数据保存规则；FACTR 按 3.6.1 节一次校准后开启补偿。每个原始轨迹可产生多个并列分支；分支本身只读，不能再创建子分支。

UI 启动后只探测控制器，不占用动作输出。选择 SpaceMouse 并连接后，控制器设置区显示已连接·待校准（`UNCALIBRATED`）；顶部只显示是否有控制器连接，点击“校准”并松开帽盖连续静止 2 秒即可。校准期间若任一轴超过 `neutral_max_abs`，静止进度会重置并提示松开帽盖，不再使分支进入 `ERROR`；30 秒内始终无法稳定才报告可重试的校准失败。一次成功校准会由全局控制器服务持续复用，设备拔插或 UI 服务重启后必须重新校准。FACTR 按官方整臂近似参考构型一次校准，不做帽盖静止或三步端点校准。

点击 SpaceMouse 接管后，后端依次显示“读取轨迹、加载环境、恢复状态、准备预览”，取得有效首帧后进入 `READY`，显示清晰的 `3、2、1` 倒计时；倒计时结束前控制器保持 disarmed，机械臂不会运动。开始接管后左键张开夹爪，右键闭合；位移和旋转增益可在创建分支前调整，也可在运行时通过 `0.05..1.0` 的滑杆实时更新。顶部控制器 pill 显示输入样本年龄：绿色 `<50 ms`，黄色 `50–249 ms`，红色 `≥250 ms`、断连或读取错误；红色状态下六维运动自动归零。接管会话的 WebSocket 断开会安全停止分支并保存已有轨迹。

`configs/ui_config.yaml` 的 `legacy_scan_roots` 会在启动时递归索引已有 `trajectory_*.npz`：

- 现有 CLI 原始轨迹如果任务和 LEVEL 能映射到上述任务目录，可在 UI 中查看并创建一级分支；
- 现有 intervention 轨迹识别为分支，只读展示；
- 任务目录之外的 legacy 轨迹保留在历史列表，但不能用未知环境恢复或创建分支；旧目录不会被修改或迁移。

新会话按 `dataset-root/projects/libero_x_vla/runs/<task_name>/<YYYY-MM-DD>/<时间>__<session_id>/` 分组，不覆盖历史结果。`catalog.sqlite3` 只保存可重建的检索和成功率索引；run 目录仍是事实来源。`run.json` 是生命周期和安全删除所需的极简清单，`config.yaml` 固化任务、模型和控制参数，`summary.json` 只记录用户关心的结果与关键时序；轨迹、视频、图表和 SpaceMouse 采样统一放在 `episodes/episode_000/`。不再重复生成 `results.jsonl`、`trajectory.json`、`source_trajectory.json` 或 `spacemouse_device_summary.json`。完整回放 metadata 已内嵌在 `trajectory.npz`，逐步可读数据保留在 `trajectory.csv`。

采集主界面的会话侧栏按“任务 → 难度 → 提示词”检索，支持全部任务、全部难度和全部提示词，切换后只列出并预览对应任务记录；导出仍集中在“数据集”页面，避免把浏览与数据生成操作混在一起。数据集页进一步拆成“轨迹评价”和“打包训练数据集”：批量评价默认跳过已有 RynnValue sidecar，也可显式覆盖；选择一条或多条轨迹时可显式开启覆盖，未开启时只补充缺失结果。评价成功后 `rynnvalue_evaluation.json/npz` 与 `trajectory.npz` 位于同一 episode，后续冻结不同数据集时直接复用。轨迹表默认每页 5 条，可选 10/20/50，详情页显示结果视频、7维action、RynnValue的absolute/relative remaining time、由 `Φ(s)=-v(s)` 得到的observation potential与entropy估计，并在同一图中显示稀疏奖励、Shape Reward和Final Reward；点击后可拖动滑块或自动播放并查看chunk范围和实际长度`L`。完整head logits和Analysis仍保存在sidecar供审计，但不在详情UI展示。冻结数据集只固定成员；通过每行右侧“配置”更新当前评价结果，训练固定使用启动时的快照，具体操作见 §4.9。按任务导出的 offline RL ZIP 保留 `runs/<run_id>/episodes/episode_000/` 层级，包含 `runs.csv`、`export.json`、`DATA_FORMAT.md`、可用的 `run.json/config.yaml/summary.json`、逐步 `trajectory.csv`、推理 chunk CSV、可用的 RynnValue 评价 sidecar，以及 `agentview.mp4` 和同步的 VLA 双视角 `vla_views.mp4`。核心 `trajectory.npz`、observation NPZ、普通图表和原始 SpaceMouse 诊断默认排除；MP4 不在 ZIP 内重复压缩。UI 的运行监视器通过已有会话 WebSocket 显示模型加载、控制器预热、环境创建、状态恢复和预览阶段，并记录首次模型加载或缓存复用耗时。桌面端监视器位于方形视频/仿真窗口右侧并与视频卡片等高，内部可滚动查看全部历史；方形预览会根据视口高度自动缩小，使顶部控制器延迟、视频和回溯进度条尽量保持在同一屏。窄屏时监视器自动移动到窗口下方。监视器停留在底部时自动追随最新事件，向上滚动后不再抢回滚动位置。Uvicorn 的逐请求 access log 已关闭，终端仍保留应用警告、错误和关键里程碑。

创建分支时会立即把父轨迹控制数据物理复制为子目录中的 `source_trajectory.npz`，但不复制父 observations、视频或图表；因此父会话被删除后，子分支仍能独立恢复状态和绘制对比。原始会话生成轨迹图和 7 张 action 图；回溯分支不生成只包含二次推理的单独图表，只生成 7 张“完整原始轨迹 + 从回溯帧开始的二次推理/人工接管”action 对比图。若准备阶段尚未执行新动作就失败，仅保存精简轨迹、清单和 summary，不再重建整段视频、observations 或对比图。已有历史目录不会自动删除或迁移。

会话列表只为 `dataset_root` 项目运行树内、带匹配 `run.json` 标记的 UI 会话显示垃圾桶图标。删除需要再次提交会话 ID，并永久移除整个 run 目录，同时更新 SQLite 索引；活动仿真、控制器校准/接管期间禁止删除，CLI、intervention 和 legacy 数据不提供删除入口。删除原始会话不会级联删除已有分支。

结果文件链接使用浏览器内联打开：MP4、PNG、JSON、CSV 等由浏览器直接预览，并在新标签页显示；浏览器无法原生显示的 NPZ 等二进制格式仍会按浏览器自身规则保存。视频接口支持 HTTP Range，因此播放器可直接跳转到时间轴指定位置。

后端遵循 `API → RunService → SimulationWorker → Simulator / Policy / Recorder / Evaluator` 的单向依赖，React 不直接接触 MuJoCo，模拟器不写数据库，Recorder 不回调 UI。前端拆分为采集、运行记录、数据集与设置页面；采集页始终挂载，浏览数据时不会使活动 SpaceMouse WebSocket 意外断开。界面使用浅色半透明面板，不再使用深蓝渐变背景。

当前 bootstrap 返回 `model_switching=true`、`task_switching=true`。这里的“模型切换”仅指在创建草稿时选择基础 Object-Pro 或与其兼容的 IQL policy overlay，不会更换视觉/语言 backbone；会话开始后策略锁定，分支继承父会话策略。缺失、被篡改、维度不符或基础 checkpoint 不兼容的 overlay 会在仿真开始前报错，不会静默回退到基础模型。

左侧“模型”页面列出基础 checkpoint 和 `policy-registry` 中的 IQL overlay。基础模型只读；overlay 可修改显示名称、复制为独立 policy ID 或永久删除。详情页显示训练 step、数据/奖励及兼容性哈希、组件大小与 SHA-256，以及本机能够匹配到的训练作业记录。活动仿真、草稿或后台作业正在使用的模型不能删除或重命名。

主要接口：

```text
GET  /api/bootstrap
GET  /api/controller
GET  /api/controllers
POST /api/controller/calibrate
POST /api/controller/gravity?controller_id=factr  # JSON: {"enabled": true|false}
GET  /api/draft
POST /api/draft
PATCH /api/draft
DELETE /api/draft
GET  /api/draft/preview.jpg
POST /api/draft/start
GET  /api/sessions
GET  /api/runs
GET  /api/datasets/summary
GET  /api/datasets/runs?page=1&page_size=5
GET  /api/datasets/runs/{id}
POST /api/datasets/evaluations
GET  /api/models
GET  /api/models/{policy_id}
PATCH /api/models/{policy_id}
POST /api/models/{policy_id}/copy
DELETE /api/models/{policy_id}
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

## 4. RynnValue + IQL 独立后训练

### 4.1 适用范围与处理流程

离线后训练位于独立目录 `vla-adapter-rynn-iql/`，不修改 `liberox-vla-adapter-terminal/` 的采集数据，也不修改上游 `VLA-Adapter/` 源码。处理链被明确拆成：只读导入数据、冻结 RynnValue 轨迹评价、确定性奖励派生、Pixel-IQL 后训练和独立 LIBERO-X 推理。其中只有轨迹评价需要运行 4B VLM；奖励派生只使用已缓存的评价输出。默认训练 Object-Pro 的连续 action head 和 proprio projector、冻结 backbone；BC/IQL 均可通过独立模型配置选择冻结、LoRA 或全量 Backbone 微调，并分别控制两个动作组件是否训练（见 4.4.3）。下图展示默认 IQL 路径。

```text
采集数据 dataset-root（只读）
        │
        ▼
prepare_dataset
  ├── 校验轨迹、双视角图像、20 Hz 与成功状态
  ├── 原始轨迹与分支的完整物理轨迹均保留
  ├── 按 root trajectory 划分训练集与验证集
  └── 生成最大 8-step、按接管/动作来源截断的变长 transition 与 dataset_manifest.json
        │
        ▼
冻结的 RynnValue-4B 离线标注
  ├── 输入：agentview + BDDL 任务提示词
  └── 输出：absolute/relative remaining time、entropy、logits 和 Analysis
        │       （只按模型/输入/max_frames 缓存，不含训练奖励）
        ▼
确定性奖励派生
  ├── 环境 terminal → sparse reward
  ├── absolute remaining time → PBRS Shape Reward
  └── gamma + kappa + reward mode → Final Reward 二级缓存
        │
        ▼
ReplayDataset
  └── (s_t, action[t:t+L], R_t, s_{t+L}, mask, terminal)
        │
        ├──────────────────────────────┐
        ▼                              ▼
Pixel-IQL                         冻结的 VLA backbone
  ├── 双 Q critic                agentview + wrist + prompt
  ├── expectile value                    │
  ├── target Q                           ▼
  └── Advantage A             视觉/语言/action-query hidden
        │                              │
        └──── exp(beta * A) 权重 ──────┤
                                       ▼
                         advantage-weighted masked L1
                                       │
                                       ▼
                      action head + proprio projector
                                       │
                                       ▼
                              policy overlay
                               ├── 独立 LIBERO-X 推理
                               └── policy-registry → Web UI
```

四个主要处理环节分别负责：

1. **Prepare（数据准备）**：递归读取 `paths.dataset_sources` 中的已完成轨迹，按照 `data.task_ids` 筛选任务，校验 20 Hz、N+1 状态/图像、动作维度、成功状态和父子分支关系。原始rollout与接管/重新推理分支都从第0步开始保留完整物理轨迹；接管前自然rollout、接管后新后缀及成功后的采集尾段始终完整用于评价，默认也全部用于训练；数据集可选择只训练到连续成功确认动作，详见 §4.9。transition 不跨越 `policy`、`policy_requery`、`human` 来源边界；若接管发生在 8 步 chunk 中间，接管前最后一个 policy transition 以实际长度结束。固定 8 步只作为最大 horizon，实际长度写入 `chunk_length`。组成训练 replay 时，再按 `(root_run_id,start,end,action_source)` 去重父轨迹与 sibling 分支物理复制的相同前缀，既不丢失整轨迹评价，也不把相同自然 rollout 重复放大。训练集与验证集仍按 root trajectory 划分。该阶段不运行模型、不计算奖励，也不修改源动作，输出schema-v4 `outputs/work/dataset_manifest.json`，以 `replay_policy: full_recording_v1` 标明完整记录采样方式。
2. **Annotate（RynnValue 轨迹评价）**：读取 prepare 生成的 manifest，在每个 chunk 边界取第三人称 `agentview` 和任务提示词，使用冻结的 RynnValue-4B 生成 absolute/relative remaining time、entropy、logits 和 Analysis。该阶段不训练 RynnValue、不更新 VLA，也不计算 sparse/Shape/Final Reward。输出位于 `outputs/work/annotations/` 和全局 `annotation-cache/`；已有 schema-v4/v5 评价会复用其完整模型输出并迁移，不会再跑 RynnValue forward。
3. **Materialize Rewards（奖励派生）**：在 `vla-liberox` 环境中，按 α、κ 和融合形式组合环境稀疏奖励、人工 Stage 分数与缓存的 RynnValue 塑形奖励。结合 `gamma` 和 macro/cumulative 模式生成 Final Reward；参数、关键帧或公式变化时只用 NumPy 重算，不加载奖励模型。参数与输入要求见 §4.4.2。
4. **Train（IQL 后训练）**：`ReplayDataset` 将轨迹、双视角图像、proprio、action chunk、mask 和已派生 reward 组合成离线 transition。Pixel-IQL 每个 step 更新双 Q、expectile value 和 target Q，并把 advantage 转成行为克隆权重；actor 按模型配置更新对应组件，默认仅更新 continuous action head 与 proprio projector。训练 checkpoint 保留 Q/V、actor、optimizer 和随机状态以便恢复；部署 overlay 不包含 critic，Backbone 经过适配时另包含其完整推理权重。

RynnValue 不是执行动作的策略，也不会在这里被训练；它只离线读取轨迹并提供时间价值。执行策略始终是 `VLA-Adapter/LIBERO-Object-Pro` 及其 IQL overlay。本训练系统不使用 Robometer、在线 RL、奖励模型微调或真机控制；Robometer 仅作为后述独立诊断评价器，不进入 IQL reward。

这里的“一致”指算法与数据语义一致，而不是把论文的 π₀.₅ 模型原样复制进 VLA-Adapter：

- RynnValue 使用固定的官方源码 commit、模型 snapshot、processor、absolute/relative value head 和 causal prefix-uniform 推理；奖励只使用论文指定的 absolute temporal distance，relative、entropy、logits 和 Analysis 原文仅完整留档。
- IQL 使用 target 双 Q 的最小值拟合 expectile V，随后用更新后的 V 构造 Q 的 Bellman target，再对 online 双 Q 做 Polyak target 更新；策略权重来自更新后的 online `min(Q1,Q2)-V`。当前 Q/V 默认恢复为历史 AdamW 配置 `β=(0.9,0.999)、ε=1e-8、weight_decay=0.01`，仍可在 YAML 中切换为官方 Adam；策略 optimizer 使用论文的 AdamW `β=(0.9,0.95)、ε=1e-8、weight_decay=1e-10`。
- 论文实验使用 π₀.₅、`H=16`、224px ResNet-18 critic、绝对关节动作且不输入 proprio；本项目必须适配 Object-Pro 的 `H≤8`、7D normalized OSC_POSE、8D proprio 和 16 GB 单卡，因此 actor loss 改为 VLA-Adapter 原生 masked L1，critic 使用轻量双视角 encoder。它们是明确的框架适配，不应表述成论文逐层复现。
- 同理，当前 16 GB profile 的 critic micro-batch、actor 梯度累积、128px critic 输入，以及未启用论文的随机裁剪/策略 EMA，均属于资源与框架配置差异；论文的 batch 64、224px、2000-step LR warmup、EMA 0.99 不能被声称已经原样复现。若实验目标改成论文超参数复现，应另建高显存 profile，而不是静默改变现有 Object-Pro profile 的训练含义。
- 被接管或终点截断的 transition 按实际执行长度 `L` 使用 mask，并把后继状态取为 `s_{t+L}`；但每个 chunk 仍只是一条宏动作 transition，因此 reward 与 Bellman bootstrap 都只使用一次 `γ`，不使用 `γ^L`。padding 从不当作已执行动作。

### 4.2 复用 VLA 环境，只新建奖励环境

RynnValue 与 VLA 需要不同版本的 Transformers，因此 RynnValue 轨迹评价必须隔离；但数据准备、奖励派生、IQL 训练和仿真推理与现有 VLA/LIBERO 运行栈完全一致，可以直接复用已经验证的 `vla-liberox`。因此只需要新建 `rynnvalue-reward`，然后把轻量的训练包安装到现有环境，不再克隆第二个 VLA 环境：

```bash
cd ~/eclipseaws/vla-liberox-workspace

conda create -n rynnvalue-reward python=3.10 -y
conda run -n rynnvalue-reward pip install -r vla-adapter-rynn-iql/requirements-reward.txt
git clone https://github.com/alibaba-damo-academy/RynnValue.git RynnValue
git -C RynnValue checkout 10e0d333f5f3811d0d130587e50f1faf48da49e5
conda run -n rynnvalue-reward python vla-adapter-rynn-iql/scripts/verify_reward_environment.py \
  --checkout ./RynnValue

# 复用原有环境，不执行 conda create。
conda run -n vla-liberox pip install -r vla-adapter-rynn-iql/requirements-train.txt
conda run -n vla-liberox pip install -e ./vla-adapter-rynn-iql
```

不要执行 `pip install -e ./RynnValue`。固定 commit 的官方顶层 `pyproject.toml` 设置了 `tool.uv.package = false`，不是可由 setuptools editable-install 的发行包；新版 setuptools 会把 `assets`、`robometer`、`rynn_value` 和 `rynn_infer` 同时识别为顶层包并拒绝构建。这里仅安装 `requirements-reward.txt` 中的运行依赖，由配置项 `paths.rynnvalue_root` 将固定 checkout 加入标注进程的导入路径，既不修改上游仓库，也不依赖 shell 中持久设置 `PYTHONPATH`。

这里不会把 RynnValue 安装进 `vla-liberox`，也不会升级其中的 Transformers。`requirements-train.txt` 保持 `transformers==4.40.1`，并兼容本项目已经验证的 Pillow 12.x 与 Accelerate 1.x。如果现有环境还没有按第 2 章完成 VLA-Adapter、LIBERO-X 和 GPU 配置，应先完成第 2 章，而不是用本节重新创建它。

固定版本记录在 `vla-adapter-rynn-iql/configs/dependency-lock.yaml`。当前 RynnValue 源码 commit 为 `10e0d333f5f3811d0d130587e50f1faf48da49e5`，RynnValue-4B Hugging Face snapshot revision 为 `3f73b5d2b5e53b21f248c8791004dde6a8cf2b92`。轨迹评价器导入本地固定版本的官方模型类，使用 `trust_remote_code=False` 加载 snapshot，并把代码版本、实际 snapshot 与模型文件 SHA-256 写入缓存元数据。

### 4.3 YAML 配置

基础配置固定 RynnValue-4B snapshot revision、Franka 的 `8×7` action chunk、8维proprio、20 Hz数据、IQL超参数和兼容16 GB显存的 `1×32` profile；远程服务器配置覆盖为单任务 `8×4` profile。所有YAML内相对路径以该YAML所在目录为基准，重复键、未知键、维度错误和非20 Hz轨迹会立即拒绝。分阶段执行：

- `configs/training/iql.yaml` / `configs/training/bc.yaml`：同级的训练入口，各自设置方法参数和实验覆盖值；BC 不读取 IQL 或奖励配置。
- `configs/models/vla_adapter.yaml`：模型 checkpoint、输入输出约定、组件训练范围和 LoRA 配置；不包含 IQL/BC 参数。
- `configs/reward.yaml`：独立的 RynnValue 评价与 Final Reward 构造配置；训练入口按需引用，不将评价配置绑定到某种训练方法。Robometer 仅用于诊断，继续由 `vla-adapter-robometer/configs/robometer_evaluation.yaml` 独立配置。
- `configs/runtime.yaml`：公共运行默认值，包括数据源、batch、步数、actor 学习率、设备、日志和输出路径。
- `configs/inference.yaml`：基础策略/overlay 对比、LIBERO-X 任务、回合数、总步数、开环执行步数和评测输出。
- `configs/dependency-lock.yaml`：RynnValue Git commit 与模型 snapshot，不作为实验超参数修改。

UI 和 CLI 使用同一个配置加载器。优先级为「公共运行默认值 → 模型/奖励配置 → 训练入口的覆盖值 → 终端/UI 显式参数」；每个相对路径始终以声明它的文件为基准。有效配置保存为展开后的 schema 2 快照，恢复时不再引用实时配置。旧 schema 1 和旧 `vla.*` / `iql.*` 公共字段仍可读取，但只在加载入口转换一次。旧 `liberox_iql.yaml` / `liberox_bc.yaml` 命令路径改为 `training/iql.yaml` / `training/bc.yaml`；不再保留 `train.yaml` 中间层。

例如 `configs/training/iql.yaml` 按需引用模型和奖励配置；BC 使用同级的 `bc.yaml`，不引用 reward：

```yaml
extends: ../runtime.yaml
presets:
  model: ../models/vla_adapter.yaml
  reward: ../reward.yaml
training:
  method: iql
  train_steps: 10000
  micro_batch_size: 1
  gradient_accumulation_steps: 32
```

actor 学习率与 Q/V 学习率、actor LR warmup 与 critic warmup 是独立参数；数值相同不代表绑定。公共 batch 和步数来自 runtime，可在两个训练入口中分别覆盖。修改 runtime 会影响没有覆盖该字段的实验；只调整一种方法时，应修改对应入口。γ 和 cumulative 统一保存在有效配置的 `reward` 中，同时供奖励归约与 Bellman target 使用。

CLI 在训练入口的 `data.include_post_success` 配置布尔值（默认 `true`）；终端流水线使用 `overrides.data.include_post_success`。这一字段只选择 replay 样本，不改变 Prepare 的完整轨迹或奖励缓存，切换时可继续复用已有评价。

以下常用配置示例位于 `configs/training/`，路径相对该目录；runtime 中声明的路径仍相对 `configs/`：

```yaml
paths:
  dataset_sources:
    - ../../../dataset-root
  rynnvalue_root: ../../../RynnValue
  policy_registry: ../../../policy-registry

data:
  action_horizon: 8
  action_dim: 7
  proprio_dim: 8
  control_hz: 20.0
  success_consecutive_steps: 5
  allow_no_success: true

reward:
  model: Alibaba-DAMO-Academy/RynnValue-4B
  max_frames: 4
  annotation_batch_size: 1
  # 以上字段决定昂贵的 RynnValue 评价缓存。
  # 以下字段只决定快速的奖励派生缓存。
  source: rynnvalue  # sparse | rynnvalue | stage
  stage_exponent: 2.0
  gamma: 0.99
  shaping_weight: 0.1
  # false=chunk宏动作奖励；true=chunk内20Hz逐步累计奖励
  accumulate_primitive_steps: false

model:
  base_checkpoint: VLA-Adapter/LIBERO-Object-Pro
  stats_key: libero_object
  backbone: frozen

iql:
  expectile: 0.8
  beta: 3.0
  max_advantage_weight: 20.0
  target_tau: 0.005

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 32

logging:
  tensorboard: true
  wandb:
    enabled: false
    mode: online
    project: vla-adapter-rynn-iql
    entity: null
    run_name: null
    group: null
    tags: [liberox, rynnvalue, iql]
    log_interval_steps: 10
  flush_seconds: 5
  console_interval_steps: 10
```

`paths.dataset_sources` 中的每一项可以是当前 `dataset-root`，也可以是 UI 数据集页面导出的任务 ZIP。`reward.rynnvalue` 决定 Final Reward 是否包含 RynnValue 势函数项；设为 `false` 时仅使用环境 sparse reward。`reward.gamma` 同时用于 PBRS chunk 折扣与 IQL Bellman target，但不参与 RynnValue 模型评价缓存的身份计算。`reward.accumulate_primitive_steps` 是奖励语义开关：默认 `false` 表示每个 action chunk 是一个宏动作；设为 `true` 才累计其中每个 20 Hz primitive step，并使用实际长度折扣。修改这些奖励派生参数只会重建快速缓存，不会重新运行 RynnValue。导入器不会改写源文件；训练/验证按 root trajectory 分组，父轨迹和它的全部分支不会被拆到不同集合。

### 4.4 数据选择、轨迹评价、奖励派生、训练与评测

这些阶段使用两个 YAML：prepare、annotate、materialize rewards 和 train 共同读取 `configs/training/iql.yaml`；evaluate 单独读取 `configs/inference.yaml`。以下命令都从 `~/eclipseaws/vla-liberox-workspace` 执行。

#### 4.4.1 指定 prepare 的数据范围

prepare 不会根据日期目录手工选择文件。它会递归扫描 `paths.dataset_sources` 下的所有 `run.json`，所以 `任务/日期/run` 这类中间目录不会影响发现结果；真正的筛选条件来自每个 `run.json` 的内容：

```yaml
paths:
  # 可同时给出多个 dataset-root 目录或 UI 导出的任务 ZIP。
  dataset_sources:
    - ../../dataset-root
  # prepare manifest、ZIP 解包缓存和奖励缓存的共同工作目录。
  work_dir: ../outputs/work

data:
  project_id: libero_x_vla
  # 空列表表示导入该 project 下的全部任务。
  task_ids: []
  validation_fraction: 0.2
  split_seed: 7
  success_consecutive_steps: 5
```

只训练一个任务时，必须填写 `run.json` 中完整、大小写一致的 `task_id`，而不是提示词或磁盘目录名。例如：

```yaml
data:
  project_id: libero_x_vla
  task_ids:
    - LEVEL1::EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove
```

也可以同时指定多个任务：

```yaml
data:
  project_id: libero_x_vla
  task_ids:
    - LEVEL1::EXTENSION_KITCHEN_SCENE1_open_the_top_drawer_of_the_wooden_cabinet
    - LEVEL1::EXTENSION_KITCHEN_SCENE25_stack_the_blue_bowl_on_the_green_bowl
```

`task_ids: []` 才表示全部任务。prepare 只接收 `status=COMPLETED` 且没有 `error` 的会话，并验证 `project_id`、20 Hz 时间网格、`N` 个动作对应 `N+1` 个状态/图像、7 维 OSC action 范围和 policy action round-trip。原始轨迹和分支都完整导入；分支的 `[0,resume_step)` 是自然 policy rollout，之后是 human 或 policy-requery 后缀。prepare 在接管点建立硬边界，接管前最后一个 chunk 可短于 8 步；RynnValue 对整条分支从第 0 步开始评价。真正构造 `ReplayDataset` 时，相同 root 的父轨迹与多个分支所复制的完全相同前缀 transition 只保留一份。

运行 prepare：

```bash
conda run -n vla-liberox python \
  vla-adapter-rynn-iql/scripts/prepare_dataset.py \
  --config vla-adapter-rynn-iql/configs/training/iql.yaml
```

结果写入 `paths.work_dir/dataset_manifest.json`，其中最值得先检查的是：

- `episode_count`：筛选后包含的原始/分支轨迹数；
- `success_count`：经过连续成功阈值确认的成功轨迹数；
- `trajectory_chunk_count`：逐轨迹完整评价所包含的 chunk 总数，包含分支物理复制的自然 rollout 前缀；
- `chunk_count`：前缀去重后实际可供 IQL 采样的变长 transition 总数；每条最多 8 步，被接管、动作来源变化或轨迹终点截断时可以更短；
- 每条 episode 的 `task_id`、`kind`、`resume_step`、`split`、`action_count` 和 `terminal_step`。

同一个 `work_dir` 再次 prepare 会原子替换旧的 `dataset_manifest.json`，但不会修改 `dataset-root` 中的任何文件。训练/验证划分按 `root_run_id` 完成，因此父轨迹及其所有分支一定处于同一个 split。当前训练器只从 `split=train` 采样，validation 仅预留给独立评估，不会自动早停或选择 checkpoint。

`allow_no_success: true` 只允许“没有确认成功轨迹”的小数据继续做流程烟测，并会打印警告；它不会把失败数据伪装成成功。设为 `false` 时 prepare 会直接终止。`action_horizon/action_dim/proprio_dim/control_hz` 在当前实现中必须保持 `8/7/8/20`，它们是 VLA-Adapter 与 Franka 数据契约，不是用于筛选数据量的参数。

#### 4.4.2 annotate 与 reward materialize 的边界

新配置默认生成 **Final Reward**；UI 训练也可直接选择独立保存的 **RynnValue** 奖励，见 §4.9。Final Reward 相加形式中，α 控制 Stage 与 sparse 的混合比例，κ 控制 RynnValue 塑形强度；缩放前，α=0、κ=0 对应 Sparse，α=0、κ>0 对应原 RynnValue 奖励，α=1、κ=0 对应 Stage。κ=0 时不需要模型评价，执行 `prepare_dataset.py → materialize_rewards.py → train_iql.py` 即可；终端流水线据此选择阶段。

**Stage 标注：**进入数据集的单条数据详情，点击“切片 / 标记关键帧”，拖动主视角录像或逐帧定位，选择 Positive/Negative 后保存。关键帧是持久标注，p 和奖励数组不写入标注身份；已有旧格式标注无需重新保存。不裁剪源数据；success 按数据集连续 done 阈值自动添加。失败轨迹无关键事件时也需要显式保存空标注。Stage 公式结果不裁剪。

```yaml
reward:
  fusion_mode: additive  # additive / multiplicative
  final_normalization: initial_chunk_v1  # 以完整轨迹第一个 chunk 的原始融合值缩放
  alpha: 1.0
  shaping_weight: 0.0
  stage_exponent: 2.0  # p>=1；控制 Stage 阶段内插值
  gamma: 0.99
  accumulate_primitive_steps: false
```

记 Stage 分数为 S。成功 P 个 positive、N 个 negative 时每次增量为 `1/(P-N+1)`（成功本身额外算一次 positive）；失败为 `1/(P+1)`。分数从 −1 起，positive 加、negative 减。关键帧与插值公式不变，下限可以低于 −1；Final Reward 生成要求 S≤0，若旧标注产生正值则报错要求检查，不静默裁剪。保存关键帧只保存标注与 Stage 预览，**不会自动生成 Final Reward**。

记 B 为 sparse，F 为未乘 κ 的 RynnValue PBRS。macro 模式取 chunk 终点的 S，并使用：

```text
Original Final Reward = B + κF
相加：R_raw = (1−α)B + αS + κF
相乘：R_raw = (−S) × (B + κF)
Final Reward[i] = R_raw[i] / (−R_raw[0])，要求 R_raw[0] < 0
```

**相乘只使用 chunk-based macro 奖励**：取 chunk 终点 S，使用边界势函数 `F=γΦ_end−Φ_start`，Bellman 同样使用 γ。不使用 α，不为 `−S` 加正下限，不插值逐步势函数；数据集及训练页均隐藏 cumulative，YAML/API 遗留的 `true` 会在有效配置中规范化为 `false`。相加保留旧 cumulative 功能：On 时累积 primitive sparse/Stage 奖励，PBRS 与 Bellman 沿用 γ^L；Off 时沿用 γ。

新 Final Reward 在融合后按每条**完整记录的第一个 chunk** 缩放，首个值为 −1；接管分支不以去重后的后缀起点重新定标。若原始首值 ≥0，明确报错；负值不设置接近零的截断阈值，不平移、不裁剪。图表与训练使用同一缩放结果，Original Final Reward、Stage 和模型原始输出保持不变。该操作只保证起点为 −1，不保证后续值位于 `[-1,0]` 或单调；逐轨迹缩放也会改变轨迹间的奖励相对尺度。产物同时保存原始融合值、首值及缩放系数用于检查。

已有结果不会自动改写：在数据集配置中重新生成 **Final Reward** 即可应用新规则，无须重新标记或运行模型；需要更新全局详情时同时开启覆盖全局评价。历史快照缺少 `final_normalization` 时按旧的未缩放语义读取；新配置使用 `initial_chunk_v1`，`none` 仅用于保留旧语义。训练调整 γ/cumulative 后，从保存的原始信号重新融合并计算新的首值，只写训练自己的产物，不对旧 Final Reward 再次除以旧系数。

需要 Stage 时会检查**全部数据成员**，缺失、损坏或源轨迹哈希不匹配的标注都会列出并终止；公式无效单独报告，不删除关键帧。相加 α=0 不要求 Stage；κ=0 不要求 RynnValue；相乘始终需要 Stage。需要但缺失的输入不会自动降级。UI 在数据集配置的 Final Reward 中调整 p、α、κ 和形式，使用最新关键帧与已有模型输出重算，成功后替换该数据集的当前 Final Reward；已启动训练保留快照。编辑器预览使用基础配置，实验曲线以已生成结果为准。独立 CLI 也可修改对应 YAML 字段重算，无须重新标记。

annotate 的作用不是重新判断任务是否成功，也不是训练 RynnValue。它冻结加载 RynnValue-4B，对 prepare 后每条轨迹的 action-chunk 边界执行以下处理：

1. 只读取第三人称 `agentview` 和该轨迹的 BDDL 提示词；不读取 wrist、动作来源或人工/策略标签。
2. 严格按官方 prefix-uniform 推理方式，在每个边界保存 absolute temporal distance、absolute head 原始 logits/entropy、relative temporal distance、relative head 原始 logits，以及整条轨迹的原始 Analysis 文本与生成 token IDs。长轨迹不再对重叠窗口结果求平均；Description / Match / Success 的解析只用于显示。
3. 将逐轨迹的原始模型输出原子写入 `paths.annotation_cache/<content_hash>.npz`，并在 `paths.work_dir/annotations/annotation_manifest.json` 保存当前 prepared dataset 的引用索引。这些 NPZ 不含 sparse、Shape 或 Final Reward。

因此，失败原始轨迹、接管后成功轨迹和重新推理分支都会用同一个冻结模型标注；区别来自 prepare 构造的 transition、实际 `action_source`、环境 terminal 和后继状态，而不是人为给 RynnValue 设置不同类别。每条分支从第 0 步到有效终点完整标注；训练采样阶段才对父轨迹与 sibling 分支重复的自然 rollout 前缀去重。

昂贵的评价配置只包含：

```yaml
reward:
  model: Alibaba-DAMO-Academy/RynnValue-4B
  device: cuda:0
  dtype: bfloat16
  max_frames: 4
  annotation_batch_size: 1
```

- `max_frames`：每个时间点的完整历史前缀均匀重采样成的固定图像槽数量；论文附录 B.3 的离线奖励重标注使用 `4`。官方独立趋势可视化 demo 默认 `64`，那是不同的演示协议，不能据此把训练默认值改成 64。
- `annotation_batch_size`：一次 RynnValue 前向处理的时间前缀数量；16 GB 显存保持 `1`。它只改变执行批量，不改变缓存中的评价语义。

运行轨迹评价：

```bash
conda run -n rynnvalue-reward python \
  vla-adapter-rynn-iql/scripts/annotate_rewards.py \
  --config vla-adapter-rynn-iql/configs/training/iql.yaml
```

快速奖励派生由下列配置决定：

```yaml
reward:
  fusion_mode: additive
  alpha: 0.0
  stage_exponent: 2.0
  gamma: 0.99
  shaping_weight: 0.1
  accumulate_primitive_steps: false
```

- `fusion_mode`：`additive` 为相加；`multiplicative` 为相乘，后者不使用 α。
- `alpha`：相加时 Stage 的比例，范围 `[0,1]`；不缩放 κF。
- `stage_exponent`：Stage 插值指数 p；p=2 使临近 positive/negative 边界的升降更快，p=1 为线性。
- `gamma`：同时影响 PBRS、累计奖励（若开启）及 IQL Bellman target，不能只改 Bellman 而沿用旧奖励数组。
- `shaping_weight`：κ，即 RynnValue 势函数奖励的系数；0 关闭该项，不读取模型输出。
- `accumulate_primitive_steps`：相加模式保留 **cumulative reward** On/Off。Off 将 chunk 当作一个宏动作；On 累计 chunk 内的折扣 primitive-step reward，并沿用实际 L 的 PBRS / Bellman 折扣 γ^L。相乘固定 Off；原 RynnValue 单独评价与旧 YAML 来源不变。

运行奖励派生：

```bash
conda run -n vla-liberox python \
  vla-adapter-rynn-iql/scripts/materialize_rewards.py \
  --config vla-adapter-rynn-iql/configs/training/iql.yaml
```

评价缓存键按单条轨迹内容寻址，包含轨迹/图像 hash、提示词、chunk 边界、RynnValue 模型与 `max_frames`，但不包含融合形式、α、p、γ、κ、cumulative 或整个数据集 hash。因此同一轨迹进入不同数据集可复用；已有 schema-v4/v5 sidecar 也会先迁移原始 head 输出，不重新执行 RynnValue。修改源数据、所需边界、提示词、`max_frames` 或模型版本才使模型评价失效。UI 的 Final Reward 使用当前数据集或全局已保存输出的实际推理配置，不会因基础 YAML 的默认 `max_frames` 不同而要求重跑。

奖励缓存是第二层，记录数据集、评价与关键帧哈希、有效融合参数及奖励实现代码指纹。输入或公式变化只重算派生数组，不改模型输出。γ / cumulative 在数据集和训练页都可配置；训练覆盖时由固定输出与标注快照生成本次训练的私有奖励，不覆盖数据集结果。p、α、κ、形式在数据集配置，训练页不另行覆盖。旧 YAML 的显式 `source` / `rynnvalue` 字段仍按旧语义兼容；使用新融合配置时移除这些旧选择字段。

#### 4.4.3 配置并运行 IQL 后训练

平台支持 **IQL / BC** 两种训练方法，默认仍为 IQL。BC 是等权 masked L1 行为克隆：
使用所选数据集训练划分中的全部有效动作 chunk（包括失败动作；成功后记录是否参与由数据集设置决定），
沿用分支前缀去重、归一化和 action mask；不在训练层筛选成功示教。
任务、来源、成功/失败及规模均在数据集层选择。BC 不读取奖励，不运行 RynnValue、
Robometer、Stage 或 Q/V，因此未评价的数据集也可以训练。

GUI：在「训练配置 → 训练方法」选择 **BC · 等权行为克隆**，其他公共训练参数和
TensorBoard/W&B 监控继续可用；BC 不显示奖励及 critic 参数。模型训练范围由独立配置决定，
默认只更新 action head 和 proprio projector，导出的 BC overlay 可直接用于原仿真推理。

CLI（workspace 根目录）：

```bash
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/prepare_dataset.py \
  --config vla-adapter-rynn-iql/configs/training/bc.yaml
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train.py \
  --config vla-adapter-rynn-iql/configs/training/bc.yaml
```

`configs/training/bc.yaml` 提供无奖励配置的示例。公共参数可写在 `training` 中；同名字段优先于
旧 `iql` 字段，原有 IQL YAML 和 `train_iql.py` 命令继续兼容。终端流水线仅需设置
`overrides.training.method: bc`，数据选择不变，自动跳过 annotate、reward 和 bind 阶段。

BC/IQL 对照应保持相同初始化、样本、micro batch、梯度累积、actor 学习率及更新预算。
`train_steps` 仍表示 micro-batch 次数，不是 epoch；完整累积组的 actor batch 是
`micro_batch_size × gradient_accumulation_steps`。新增 `training.actor_lr_warmup_steps`
单独控制 actor 学习率预热，BC/IQL 预设各自显式配置；仅旧 IQL 的缺省或 `null` 设置在迁移时沿用原 critic warmup，BC 不依赖它。
IQL 后者还决定何时启用 advantage 权重，BC 则始终等权。不要通过扩大 critic warmup
模拟 BC。断点恢复必须使用相同训练方法；界面队列仍从头启动，CLI 支持 BC 断点恢复。

实现分层：`methods.py` 声明方法能力，`algorithms.py` 实现辅助更新、actor objective
和算法状态接口，`training.py` 共用优化、监控、取消和保存；`ActionDataset` 读取动作样本，
`ReplayDataset` 额外提供 IQL transition/reward。新增方法无需复制 UI 或整套训练循环。

模型配置独立于 BC/IQL。在训练页依次选择「基础模型 → 训练方法」，再展开「模型训练配置」及对应方法的高级参数：

- **Backbone 适配方式**：冻结 / LoRA / 全量微调，三选一，避免“冻结但又开启 LoRA”的歧义。
- **Action head / Proprio projector**：分别选择训练或冻结；全部冻结会拒绝开始训练。
- **LoRA**：仅选择该方式时显示 rank、alpha、dropout。参照 [VLA-Adapter 官方 finetune](https://github.com/OpenHelix-Team/VLA-Adapter/blob/main/vla-scripts/finetune.py)，默认 rank=32、alpha=64、dropout=0、Gaussian 初始化、`all-linear`，同时训练 action queries。这里的 Backbone 包含视觉塔、多模态投影、语言模型和 action queries，不仅是 Qwen；Proprio projector 是独立组件。

目前模型目录只注册已支持的 `vla_adapter`（Object-Pro），不会把尚未适配的 foundation model 显示成可用模型。`models.py` 管理轻量能力注册与配置、按模型选择前向后端；`model_adaptation.py` 管理冻结、LoRA、参数范围与 Backbone 保存。前端配置请求不加载 Torch 或模型，模型设置不会改变 reward、BC loss 或 IQL 更新顺序。

YAML 使用顶层 `model`，同时适用于 `training/iql.yaml` 与 `training/bc.yaml`；终端流水线使用 `overrides.model`。旧 `vla.freeze_backbone` 仍兼容，无新配置时 true 对应 frozen、false 对应 full；新 `model.backbone` 优先。默认配置如下，IQL 的双 Q、value 和 target 仍独立训练：

```yaml
model:
  family: vla_adapter
  backbone: frozen  # frozen | lora | full
  action_head: train  # train | frozen
  proprio_projector: train  # train | frozen
  lora:
    rank: 32
    alpha: 64
    dropout: 0.0
  base_checkpoint: VLA-Adapter/LIBERO-Object-Pro
  stats_key: libero_object
  use_pro_version: true

iql:
  critic_image_size: 128
  critic_lr: 0.0003
  value_lr: 0.0003
  critic_optimizer: adamw
  critic_weight_decay: 0.01
  value_optimizer: adamw
  value_weight_decay: 0.01
  critic_max_grad_norm: 10.0
  value_max_grad_norm: 10.0
  expectile: 0.8
  beta: 3.0
  max_advantage_weight: 20.0
  target_tau: 0.005
  critic_warmup_steps: 1000

training:
  actor_lr_warmup_steps: 1000
  policy_peak_lr: 0.00003
  policy_final_lr: 0.000003
  train_steps: 10000
  micro_batch_size: 1
  gradient_accumulation_steps: 32
  checkpoint_interval: 512
  resume_checkpoint: null
  seed: 7
  device: cuda:0
  dtype: bfloat16

logging:
  tensorboard: true
  flush_seconds: 5
  # 首步、每 10 步以及最终一步在终端打印一次进度。
  console_interval_steps: 10
```

参数语义如下：

- 数据与显存：`critic_image_size`只控制Q/V使用的双视角缩放尺寸；VLA actor仍走自身processor。`micro_batch_size`是每次同时送入Q/V和VLA actor的transition数量，不再被人为限制为1；`gradient_accumulation_steps`决定多少个micro-step后更新actor。基础16 GB profile采用 `1×32`，单任务A100服务器profile采用 `8×4`，两者等效actor batch均为32。批处理只允许prepared training split包含唯一 `task_id + prompt`；旧多任务manifest仍可用 `micro_batch_size=1`训练。
- 训练长度：`training.train_steps`表示 micro-batch 次数，不是 epoch；IQL 每次同时更新 critic/value，BC 没有 critic/value。实际抽样transition数为 `train_steps × micro_batch_size`，actor optimizer更新次数约为 `ceil(train_steps / gradient_accumulation_steps)`。因此将 `1×32` 改为 `8×4`并保持相同 `train_steps`会增加数据吞吐和actor更新次数，不应直接与旧run按step数视为相同训练预算。每个变长transition仍被均匀采样，`action_source`和`transition_type`语义没有改变。
- critic/value：`critic_lr` 与 `value_lr` 分别控制双 Q 和 expectile value 的 optimizer；`critic_optimizer`、`value_optimizer` 可选 `adam` 或 `adamw`，对应的 `*_weight_decay` 必须显式配置。默认恢复历史参数 `adamw + 0.01`，即使用 PyTorch AdamW 默认 `β=(0.9,0.999)、ε=1e-8`；切换到官方 Adam 时应同时明确设置 weight decay。`critic_max_grad_norm`、`value_max_grad_norm` 分别限制 Q/V 的总梯度范数，默认均为 `10.0`。每步先以冻结 target Q 更新 V，再用更新后的 V 更新 online Q，最后 Polyak 更新 target Q。`expectile` 越高，value 越偏向高 Q 动作；actor 权重使用 online `min(Q1,Q2)-V`，计算 `exp(beta × advantage)` 后由 `max_advantage_weight` 截断。当前默认 `beta=3`、权重上限 `20`。
- actor 优化：`policy_peak_lr` 到 `policy_final_lr` 使用 warmup 加余弦衰减。默认前 `1000` 个 `critic_warmup_steps` 中 Q/V 正常学习，同时 actor 以权重 `1` 做普通行为克隆；warmup 结束后才切换到 advantage-weighted L1，actor 在 warmup 期间并未冻结。
- 保存与复现：`checkpoint_interval` 是训练 step 间隔，必须整除梯度累积步数；`seed` 控制网络初始化、replay 抽样及相关随机状态。当前 profile 要求单个 `cuda:N` 设备和 `bfloat16` actor，不会在显存不足时静默回退 CPU。
- 终端进度：`logging.console_interval_steps`控制打印间隔。首步和最终一步始终打印；每行包含当前/总step、百分比、BC warmup/IQL阶段、已用时间、ETA、step/s、sample/s、Q/value/actor loss、Q/V/advantage均值、advantage weight、actor学习率及CUDA峰值显存。比较不同batch时应以 `samples_per_second`衡量吞吐，不能只比较step/s。

运行训练：

```bash
conda run -n vla-liberox python \
  vla-adapter-rynn-iql/scripts/train_iql.py \
  --config vla-adapter-rynn-iql/configs/training/iql.yaml
```

每次训练创建新的 `outputs/training/<run>/`，不会覆盖旧实验。`checkpoint_interval` 必须能被 `gradient_accumulation_steps` 整除。断点恢复时把配置改为：

```yaml
training:
  # 其余字段保持与目标实验兼容；train_steps 是恢复后的总目标步数。
  train_steps: 20000
  resume_checkpoint: ../outputs/training/<run>/checkpoints/step_00010000
```

恢复会校验 dataset hash、reward hash、基础 checkpoint、stats key 和模型训练范围（含有效 LoRA 参数），并恢复算法状态、actor、optimizer、随机数状态和 replay sampler。改变冻结范围、LoRA 参数或 BC/IQL 方法需新建训练；旧 checkpoint 仅能按原冻结配置恢复。`train_steps` 是绝对终点，例如从 step 10000 恢复到 `train_steps=20000` 只再执行 10000 步。

LoRA 依赖训练环境中的 `peft==0.11.1`（已加入 `requirements-train.txt`；已有该版本无需重装）。训练开始会输出各组件总参数与可训练参数数目。Backbone 不冻结时需要保留反向传播图，显存和耗时会增加；并不保证原 `1×32` / 16 GB profile 能用于全量微调，不会自动切 CPU 或悄悄冻结。

新导出清单为 schema 3，兼容读取 schema 1/2：冻结 Backbone 时仍只保存两个动作组件；LoRA/full 额外保存完整 `backbone.pt`，LoRA 在最终导出前合并，但训练 checkpoint 保留未合并的 LoRA 增量和 action queries 以便恢复。导出因此比小型 LoRA adapter 大，优势是 CLI/GUI 推理不依赖 PEFT 或训练代码。切换涉及适配后 Backbone 的策略时重新加载基础模型，避免权重残留；两个旧式冻结 Backbone overlay 仍可轻量切换。新版 overlay 需要本次更新后的加载器。

#### 4.4.4 评测 base 与训练后的 overlay

评测不读取训练入口的数据筛选范围，而是由 `configs/inference.yaml` 独立指定任务和 rollout：

```yaml
policy:
  overlay: ../../policy-registry/latest/policy.yaml
evaluation:
  level: LEVEL1
  task_name: EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove
  trials: 1
  max_steps: 300
  open_loop_steps: 8
  seed: 7
  compare_base: true
```

`compare_base: true` 会用相同任务设置分别运行基础 Object-Pro 与 IQL overlay；成功仍只由 LIBERO-X 环境判定。

```bash
conda run -n vla-liberox python \
  vla-adapter-rynn-iql/scripts/evaluate.py \
  --config vla-adapter-rynn-iql/configs/inference.yaml
```

#### 4.4.5 哪些修改需要重跑哪些阶段

| 修改内容 | prepare | RynnValue annotate | reward materialize | train | evaluate |
|---|---:|---:|---:|---:|---:|
| `dataset_sources`、`project_id`、`task_ids` 或源轨迹 | 必须 | 只评价新增/失效轨迹 | 必须 | 必须 | 按需 |
| `success_consecutive_steps`或 action/chunk 边界 | 必须 | 仅缺失所需边界时 | 必须 | 必须 | 按需 |
| 仅 `split_seed`、`validation_fraction` | 必须 | 复用已有逐轨迹评价 | 必须 | 必须 | 按需 |
| RynnValue 模型/版本、`max_frames`、提示词 | 按输入是否变化 | 必须 | 必须 | 必须 | 按需 |
| `rynnvalue`、`gamma`、`shaping_weight`、`accumulate_primitive_steps` | 不需要 | **不需要** | 快速重算 | 必须 | 按需 |
| VLA checkpoint/stats 或任一 `iql.*` 参数 | 不需要 | 不需要 | 相同奖励配置直接复用 | 必须 | 必须 |
| 仅 `logging.*` | 不需要 | 不需要 | 不需要 | 仅影响新训练 | 不需要 |
| 仅 `inference.yaml` | 不需要 | 不需要 | 不需要 | 不需要 | 必须 |

首次跑通或希望严格重建全部阶段时，可以使用编排脚本自动在两个 Conda 环境间切换：

```bash
python vla-adapter-rynn-iql/scripts/run_pipeline.py \
  --config vla-adapter-rynn-iql/configs/training/iql.yaml \
  --inference-config vla-adapter-rynn-iql/configs/inference.yaml
```

只想完成 `prepare → annotate → materialize rewards → train` 而暂不仿真评测时，加 `--skip-evaluation`。

#### 4.4.6 不启动图形界面的远程终端训练

远程服务器不需要启动 FastAPI、React 或浏览器。推荐使用有状态终端流水线，它会在两个 Conda 环境之间依次执行数据选择、Prepare、RynnValue评价、奖励派生、评价绑定和IQL训练：

```bash
python vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml
```

`configs/terminal_pipeline.yaml` 的 `base_config` 引用 `./training/iql.yaml`，BC 只需改为 `./training/bc.yaml`，默认不重复指定方法。旧的 `overrides.training.method` 仍可显式覆盖选择；使用内置入口时会先选择对应方法文件，再读取公共配置，BC 不加载 IQL/reward 默认值。任务、数据规模和训练参数均可在此覆盖，不会修改基础 YAML。默认示例按类别选择5条基础策略失败轨迹和50条人工接管成功轨迹：

```yaml
selection:
  task_id: EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove
  mode: quota
  seed: 7
  source_types: [inference, manual, policy_requery]
  outcomes: [success, failure]
  size: null
  quotas:
    - {source_type: inference, outcome: failure, count: 5, order: random}
    - {source_type: manual, outcome: success, count: 50, order: random}

overrides:
  paths: {}
  data:
    validation_fraction: 0.2
    split_seed: 7
    success_consecutive_steps: 5
  reward: {}
  model: {}
  training:
    train_steps: 20000
    micro_batch_size: 8
    gradient_accumulation_steps: 4
  iql:
    critic_warmup_steps: 1000
    beta: 3.0
    max_advantage_weight: 20.0
  logging:
    tensorboard: true
    wandb:
      enabled: true
      mode: online
      project: vla-adapter-rynn-iql
      entity: null
      run_name: null
      group: terminal-pipeline
      tags: [liberox, rynnvalue, iql, terminal]
      log_interval_steps: 10
    console_interval_steps: 10
```

`quota`用于分别控制不同来源/结果的数据量；配额不足会退出，不会静默缩小数据集。还支持：

- `mode: random`：在筛选后的候选轨迹中按固定seed抽取 `size` 条，同时设置 `quotas: []`；
- `mode: all`：纳入该任务下全部符合筛选条件的轨迹，同时设置 `size: null` 和 `quotas: []`。

任务默认接受完整 `LEVEL1::...` ID；省略LEVEL前缀时，只有在数据源中能唯一匹配才会自动补全。每条流水线只允许一个任务。分支和父轨迹仍按照root ID进入同一个train/validation split，相同物理前缀只在构造ReplayDataset时去重。

脚本首先显示候选/选中数量、类别组成、动作和chunk规模、已有绑定评价、训练参数与阶段执行计划，然后等待确认。无人值守任务使用：

```bash
python vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml \
  --yes
```

其他控制选项：

- `--dry-run`：只扫描、校验和打印计划，不创建流水线、冻结数据集或训练结果；
- `--force-prepare`：忽略匹配的Prepare缓存并重新生成manifest；
- `--force-annotate`：显式放弃评价复用，重新执行 RynnValue，同时强制重建派生奖励。日常调整训练奖励参数不应使用此选项。

默认复用规则如下：

1. 数据成员、源文件哈希、成功阈值、split或chunk结构未变化时，跳过Prepare；仅修改 `iql.*` 不会使Prepare失效。
2. 当前 prepared dataset 已有完整、官方推理配置相同且文件哈希有效的 annotation manifest 时，跳过 RynnValue 评价阶段。这个命中不要求奖励参数相同。
3. 即使新工作目录还没有 annotation manifest，脚本也会按轨迹检查 `rynnvalue_evaluation.json/.npz`、schema-v4/v5 旧缓存和全局 content cache。只要轨迹/图像/提示词/边界/模型契约匹配，就迁移原始 head 输出，不再运行 RynnValue forward。
4. 当前 prepared dataset 已有完整、奖励参数相同的 reward manifest 时直接复用；`rynnvalue` / `gamma` / `shaping_weight` / `accumulate_primitive_steps` 不一致时只执行快速 reward materialize。
5. 评价结束后结果会原子绑定回各轨迹目录，因此删除终端流水线缓存后仍可复用，也能在现有数据详情页查看。
6. IQL训练默认每次创建新的输出和overlay；只有 `overrides.training.resume_checkpoint` 明确指定checkpoint时才恢复。

每次执行的状态写入：

```text
vla-adapter-rynn-iql/outputs/terminal-pipelines/
├── datasets/terminal-ds-<hash>/dataset.json
├── cache/prepared/<hash>/work/
└── runs/<timestamp>__<id>/
    ├── pipeline.json
    ├── effective_config.yaml
    └── train_result.json
```

`Ctrl+C`会转发给当前Conda子进程；训练阶段仍通过安全checkpoint停止，流水线记录为 `INTERRUPTED`。原来的 `run_pipeline.py` 保留为无选择清单、无阶段状态检查的简单编排入口，新远程训练应优先使用 `train_terminal.py`。

#### 4.4.7 使用 TensorBoard 查看训练变化

新训练默认同时保存两种指标：`metrics.jsonl` 是可审计的逐步原始记录，`tensorboard/` 是图表事件。训练开始后可在另一个终端启动：

```bash
cd ~/eclipseaws/vla-liberox-workspace
conda run -n vla-liberox tensorboard \
  --logdir vla-adapter-rynn-iql/outputs/training \
  --host 127.0.0.1 \
  --port 6006
```

浏览器打开 `http://127.0.0.1:6006`。如果训练发生在另一台机器上，不要把 TensorBoard 直接暴露到局域网；从本机建立 SSH 转发后访问同一地址：

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<training-host>
```

已经完成、尚无 TensorBoard 事件的旧训练可从 `metrics.jsonl` 转换，不需要重新训练：

```bash
conda run -n vla-liberox python \
  vla-adapter-rynn-iql/scripts/metrics_to_tensorboard.py \
  --run-dir vla-adapter-rynn-iql/outputs/training/<run>
```

转换结果写入该 run 的 `tensorboard-imported/`。旧日志只能显示当时已经记录的 loss、Q/V、advantage weight、耗时和显存；新版本才会额外记录七个 action 维度的 L1、夹爪预测/目标均值、闭爪样本比例、actor 学习率和梯度范数。

建议重点观察：

- `loss/actor_loss` 与 `action_l1/actor_l1_gripper`：动作头总体误差和夹爪维度误差；
- `gripper/actor_gripper_prediction_mean`、`target_mean` 和 `target_close_fraction`：判断模型是否只学会移动而没有学会闭爪；
- `iql/advantage_weight_mean`：若长期贴近 `max_advantage_weight`，通常表示权重饱和；
- `value/q_mean`、`value/value_mean`、`value/advantage_mean`：判断 critic/value 是否漂移；
- `optimization/actor_grad_norm`、`action_head_parameter_norm` 和 `proprio_projector_parameter_norm`：只在梯度累积真正提交 optimizer 的步骤出现，默认每 32 步一次，用于检查两个可训练组件是否获得梯度以及参数尺度是否异常漂移；
- `system/steps_per_second` 与 `cuda_peak_memory_gib`：查看速度和显存峰值。
- `system/progress_percent` 与 `estimated_remaining_seconds`：查看训练完成比例和滚动 ETA；checkpoint 保存期间的短暂停顿会暂时反映在 ETA 中，后续窗口更新后会恢复。

可在 YAML 中把 `logging.tensorboard` 设为 `false` 关闭事件写入，`metrics.jsonl` 仍会保留。`logging.console_interval_steps` 必须为正整数，例如设为 `1` 可逐步打印，长训练建议保持 `10` 或调大以减少终端日志。

#### 4.4.8 使用 W&B 远程监控

训练器也可以把同一组分层指标发送到 Weights & Biases。先更新训练依赖并登录：

```bash
conda run -n vla-liberox pip install -r \
  vla-adapter-rynn-iql/requirements-train.txt
conda run -n vla-liberox wandb login
```

然后在 `training/iql.yaml` / `training/bc.yaml` 或 `terminal_pipeline.yaml` 的 `logging`/`overrides.logging` 中启用：

```yaml
logging:
  tensorboard: true
  wandb:
    enabled: true
    mode: online                 # 无外网服务器改为 offline
    project: vla-adapter-rynn-iql
    entity: null                 # 团队账号可填写 entity
    run_name: null               # null 时使用本地唯一 run ID
    group: a100-sweep
    tags: [liberox, rynnvalue, iql]
    log_interval_steps: 10
  flush_seconds: 5
  console_interval_steps: 10
```

`online` 模式在训练开始时强制检查登录，失败会直接给出错误，不会静默转成离线记录。训练目录中的 `wandb.json` 保存 W&B run ID、URL和本地目录；W&B 页面使用与 TensorBoard 相同的 `loss/*`、`value/*`、`iql/*`、`optimization/*`、`action_l1/*`、`gripper/*` 和 `system/*` 指标名。`log_interval_steps` 只控制网络提交频率，`metrics.jsonl` 与 TensorBoard 仍逐step写入。

无法联网时设置 `mode: offline`。训练完成并转移日志后再同步：

```bash
conda run -n vla-liberox wandb sync \
  vla-adapter-rynn-iql/outputs/training/<run>/wandb/offline-run-*
```

#### 4.4.9 8×A100 资源配置

当前实现是**单进程、单GPU训练器**。`reward.device`和`training.device`各接受一个`cuda:N`；没有DDP/FSDP，设置8张可见卡不会让单次训练自动使用8卡。单卡内已支持同一任务prompt的批量双视角VLA输入，远程终端默认使用 `micro_batch_size=8`、`gradient_accumulation_steps=4`提高A100显存利用率；Q/V、actor梯度、随机采样和checkpoint尚未做多rank同步。

在共享服务器上，推荐由调度器为每条流水线分配一张A100。用物理GPU 3时：

```bash
CUDA_VISIBLE_DEVICES=3 python \
  vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml \
  --yes
```

此时YAML中的 `reward.device: cuda:0` 和 `training.device: cuda:0` **保持不变**：进程内的 `cuda:0` 已映射到物理GPU 3。不要在只暴露一张卡时写 `cuda:3`。

要利用8张A100，当前最有效的方式是并行运行8个独立实验，而不是让一个实验占8卡：先用一条流水线完成Prepare和RynnValue评价绑定；确认第二次 `--dry-run` 显示这两个阶段可跳过后，再准备8份配置，分别修改 `training.seed`、待比较的超参数、`wandb.run_name`，并保持相同 `wandb.group`。每个进程绑定不同物理GPU。这样缓存评价只计算一次，8张卡用于8组IQL实验，W&B可在同一group中直接比较。

Slurm环境建议每个array job申请一张卡（例如 `--gres=gpu:a100:1`），并继续在YAML内使用 `cuda:0`；Slurm会完成可见设备映射。若目标是用8卡缩短**同一个**训练run，需要另行实现DDP/FSDP，不能只改YAML或启动命令。

### 4.5 数据与奖励语义

RynnValue 只读取正常方向的 `agentview` 和 BDDL 提示词；每个 action-chunk 边界都使用截至该点均匀采样的完整因果前缀并读取最后 value slot，不对长轨迹做重叠窗口平均。环境 `done` 是唯一成功依据，RynnValue 生成的 Success 文本只作诊断。原始轨迹与分支都完整评价；`ReplayDataset` 单独去重父轨迹和 sibling 分支复制的相同自然 rollout 前缀，接管后的 `human` 或 `policy_requery` 后缀完整保存，其训练取样受成功后动作选项控制。

这里的 `float32` 与 `bfloat16` 是浮点计算精度，不是 INT8/4-bit 权重量化。固定的 RynnValue-4B checkpoint 使用 BF16：Qwen 文本隐藏维度为 2560，连续 8 个 `<value>` token 的隐藏状态拼接后形成 value head 的 20480 维输入。官方自定义 value-head 构造函数默认以 FP32 建层，因此适配器在加载后显式把**整个模型**（Qwen backbone、普通 value head 和 relative value head）统一转换为 YAML 固定的 BF16，并在标注前检查所有浮点参数；如果仍混有 FP32 参数会立即报出具体参数名。value bin 解码和 entropy softmax 则按官方实现转为 FP32，以避免低精度概率计算不稳定。第一版 16 GB profile 不接受把 `reward.dtype` 改为 `float32` 或 `float16`。

固定总时长的采集可能在任务成功后继续记录；此时 `done` 既可能一直保持 `True`，也可能因为物体继续移动、短暂离开成功区域而出现 `True → False → True`。`data.success_consecutive_steps` 是成功去抖阈值，默认要求连续 5 个控制步为 `True`（20 Hz 下为 250 ms；改为 10 即 500 ms）。确认成功前出现一次 `False` 会清空计数，必须重新连续满足阈值。第 5 个确认 action 标记成功边界，默认不结束训练轨迹；若整条轨迹都没有达到连续阈值，则按失败轨迹处理，即使源会话曾记录过瞬时 `success=true`。

**默认保留确认成功后的实际动作；可通过数据集“配置 → 训练取样配置 → 成功后动作”选择截断。** 确认成功与结束记录是两个独立边界：`terminal_step` 沿用旧字段名，仅表示连续成功确认 action。保留模式下 `bootstrap_mask` 只在实际记录末尾为 0；截断模式保留到确认动作（含该动作），此处停止 bootstrap，后续 chunk 不进入 BC/IQL 训练。未达到连续成功阈值的轨迹始终完整保留。新 manifest 的 `action_count` 等于 `recorded_action_count`；`trailing_action_count` 继续统计确认成功后的步数，不再表示排除数量。成功阈值与奖励公式不变：Sparse 在确认成功后保持 0，Stage 成功后保持 0，RynnValue 继续使用原有 sparse＋PBRS；成功后再次出现 `done=False` 不会重新启用 `-1` step cost。源 `trajectory.npz`、全部同步 observation 和完整评价曲线均不裁剪。

已有 schema-v4 数据集若保存了完整 `evaluation_chunks` 和奖励数组，新训练可直接将尾段作为训练样本读取，无须重新评价或重标关键帧，也不改写历史 manifest 和奖励。为兼容旧评价缓存，`evaluation_chunks` 中的 `post_terminal_evaluation` 只是历史描述，不能再据此判断该 chunk 是否参与训练；实际采样来源仍为 `policy`、`policy_requery` 或 `human`。缺少完整尾段信息或奖励数组时明确报错，需要重新 Prepare／生成完整评价，不能静默省略。历史训练结果不变；切换成功后取样规则时应新建训练 run，不能从另一种规则的 checkpoint 恢复（训练集不存在成功尾段时两种规则等价）。新 checkpoint 与 provenance 记录 `replay_policy: full_recording_v1 | confirmed_success_v1`。

下面先说明 **Original Final Reward**（融合前的 RynnValue 奖励）。设预测剩余秒数为 `v_t`，势函数为 `Φ_t=-v_t`。默认 `reward.accumulate_primitive_steps: false` 时，长度为 `L` 的 chunk 被视作一条宏动作 transition：

```text
r_sparse(t) = 0，若该 chunk 结束时任务已完成；否则为 -1
r_shape(t)  = γ Φ(t+L) - Φ(t)
r_final(t)  = r_sparse(t) + κ r_shape(t)
y_t         = r_final(t) + γ m_t V(s_{t+L})
```

其中 `m_t` 在非终止 transition 为 `1`、终止 transition 为 `0`。失败轨迹的每个宏动作 sparse reward 都为 `-1`；完成任务的宏动作 sparse reward 为 `0`。轨迹末尾、接管点或 `action_source` 变化产生的短 chunk 都使用 mask 统一到 `8×7` 张量，实际 `L` 决定有效 action prefix 和后继状态 `s_{t+L}`，但不把一次宏动作再次按低层控制步累计 reward 或指数折扣。原始错误策略的完整 chunk 仍保留；被接管分支另建实际执行 prefix 与 human/policy-requery transition，二者不会混在同一 chunk 中。

若设置 `reward.accumulate_primitive_steps: true`，同一个 chunk 改为 Semi-MDP 累计语义：

```text
R_sparse(t) = Σ[h=0..L-1] γ^h r_sparse(t+h)
R_shape(t)  = γ^L Φ(t+L) - Φ(t)
R_final(t)  = R_sparse(t) + κ R_shape(t)
y_t         = R_final(t) + γ^L m_t V(s_{t+L})
```

该布尔值写入 effective config、奖励缓存键和训练记录。改变它只会使用已保存的 absolute/relative distance 等 RynnValue 输出重新计算确定性 Shape/Final Reward，不会再次执行模型前向；改变 `max_frames`、模型或 revision 才需要重新评价。

轨迹评价 schema v6 只保存原始 absolute/relative distance、entropy、logits 和 Analysis，不固化任何训练奖励语义。奖励派生 schema v1 另行保存 `sparse_reward`、未乘 `κ` 的 `pbrs_shaping_reward`、已乘 `κ` 的 `dense_reward` 和 Final Reward `pbrs_chunk_reward`。已有 hash 与模型推理契约匹配的 schema-v4/v5 轨迹评价会复用全部模型输出；即使旧文件带有由不同 `gamma`、`κ` 或累计模式产生的 Final Reward，也只丢弃旧派生数组并在 CPU 上重算，不再运行 RynnValue。

选择 **Final Reward** 训练时读取融合并缩放后的奖励，公式见 §4.4.2；选择 **RynnValue** 则读取独立保存的原始 sparse + κ·Shape Reward，不使用 Stage 融合和 Final Reward 首值缩放。Final Reward 缩放前，相加 α=0、κ=0 对应 sparse-only；α=0、κ>0 对应上述原始 RynnValue 奖励；α=1、κ=0 对应 Stage。随后按完整轨迹首个 chunk 的原始融合值缩放，使首值为 −1。仅改变 reward 数值，不改变 IQL 网络、更新顺序和 macro/cumulative 对应的 Bellman discount。保存结果包含 `original_final_reward`、`final_reward`，后者也保留 `pbrs_chunk_reward` 兼容别名；不要把这个旧字段名理解为融合结果仍是纯 PBRS。

主要中间结果：

- `outputs/work/dataset_manifest.json`：只读 replay 索引、episode/chunk 数与数据哈希。
- `outputs/work/annotations/`：当前 prepared dataset 对已缓存 RynnValue 原始输出的引用索引。
- `outputs/work/rewards/`：按当前 `gamma`、`κ` 和 reward mode 派生的 sparse/Shape/Final Reward 二级缓存。
- `outputs/training/<run>/`：训练指标、effective config、Q/V/target、actor 组件、优化器和 RNG checkpoint。
- `outputs/evaluation/<run>/`：轨迹、`agentview.mp4`、双视角 `vla_views.mp4`、逐回合结果和成功率。

### 4.6 Overlay 推理与 Web UI

训练完成后，`policy-registry/<policy_id>/policy.yaml` 只引用 action head 与 proprio projector，并包含组件 SHA-256 和兼容性哈希。刷新 UI 后即可在“创建仿真”的策略下拉框选择它；同一基础 checkpoint 复用已加载 backbone，只热切换两个小组件。完整设计、输出文件和断点恢复说明见 `vla-adapter-rynn-iql/README.md`。

UI 只接受与当前基础 checkpoint、8×7 action、8 维 proprio 兼容且哈希有效的 overlay。会话开始后策略锁定，回溯分支继承父会话策略；overlay 缺失、被篡改或不兼容时会在仿真开始前报错，不会静默回退到基础模型。

独立推理是否对比基础策略由 `configs/inference.yaml` 控制。每个策略仍使用 LIBERO-X 环境原本的 `done` 判断与成功率，不会使用 RynnValue 文本判断替代任务成功条件。

### 4.7 测试、限制与参考资料

Stage 标注与模型原始评价独立存储，通过 Final Reward 组合，不改变 IQL 更新设计。融合奖励是本项目实验设计，不是 RynnValue 官方模型输出，也不保留一般 PBRS 的策略不变性保证。`docs/` 中的开发调研文档仅保留在本地，不随仓库分发。

当前已有数据即使全部失败也允许完成流程烟测，但会明确警告，不能据此预期策略提升。4B RynnValue 评价和 VLA/IQL 严格串行使用 GPU；任一阶段显存不足会报告具体阶段且不会自动回退 CPU。

不下载模型的 CPU 回归测试：

```bash
cd ~/eclipseaws/vla-liberox-workspace/vla-adapter-rynn-iql
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q
```

完整 GPU 验收应至少包括：一条轨迹的 RynnValue 标注、20 个 IQL 更新步、overlay 导出、一次短 CLI rollout，以及在 Web UI 中选择该 overlay 创建仿真。流程跑通不等于策略已经提升；策略效果仍需独立验证集、多个随机初始状态和足够成功/接管数据评估。

相关材料：

- [RynnValue 论文：时间距离与 PBRS](https://arxiv.org/abs/2608.09853)
- [RynnValue 官方实现](https://github.com/alibaba-damo-academy/RynnValue)
- [Implicit Q-Learning 论文](https://arxiv.org/abs/2110.06169)
- [VLA-Adapter 官方实现](https://github.com/OpenHelix-Team/VLA-Adapter)
- [LIBERO-X 官方实现](https://github.com/meituan/LIBERO-X)
- [本仓库的独立训练说明](vla-adapter-rynn-iql/README.md)

### 4.8 能到达目标但无法抓取时的调优顺序

“能到达碗附近但没有完成闭爪/抬升”通常说明空间接近能力已经学到，瓶颈集中在短暂的抓取转换阶段。不要先盲目增加总训练步数；应先在同一任务、相同 seed 下对 base 与 overlay 各运行多次，并按以下顺序定位：

1. 将 `configs/inference.yaml` 的 `evaluation.open_loop_steps` 从 `8` 改为 `1` 或 `2`。抓取接触阶段每 50–100 ms 重规划，通常比一次盲执行 8 步更稳；若成功率明显上升，主要问题是开环执行而不是奖励或动作头完全失效。
2. 检查失败视频对应的 `trajectory.csv`：raw gripper 应在接触前从接近 `1`（开）切到接近 `0`（闭），环境 gripper action 则应变为 `+1`。若始终不闭合，重点检查示教中“闭爪并保持、随后抬升”的有效 chunk 数，而不是只看总轨迹数。
3. 现有 50 条接管轨迹的长前缀会让“接近目标”的样本远多于真正抓取转换。优先从夹爪接近碗前开始新增短分支，明确包含对准、闭爪保持和抬升；数据准备仍只导入分支后缀，不重复父前缀。
4. 查看训练目录的 `metrics.jsonl`。当前默认 `beta: 3`、`max_advantage_weight: 20`、`critic_warmup_steps: 1000`；若 `advantage_weight_mean` 仍长期贴近上限，说明 noisy advantage 仍让少数 chunk 主导训练，应先检查 Q/V 和 reward 分布，再一次只调整一个参数。
5. 若 gripper 输出方向正确但动作抖动或过冲，再把 `policy_peak_lr` 从 `3e-5` 降至 `1e-5`、`policy_final_lr` 从 `3e-6` 降至 `1e-6`，并保留独立验证轨迹选择 checkpoint，避免 action head 在少量成功数据上过拟合。

摄像头关闭功能适合诊断模型究竟依赖主视角还是腕部视角，不建议把单摄像头消融结果直接当作正式策略提升。训练 overlay 仍是双摄像头模型；若希望永久改成单摄像头结构，需要重新设计并训练模型输入层，而不是只关闭一个槽位。

### 4.9 在 Web UI 中创建数据集、标注与训练

LIBERO Studio 已把 CLI 的 prepare、RynnValue 轨迹评价、奖励派生和 IQL 训练编排为可恢复的后台任务，但仍保持两个 Conda 环境隔离：prepare/奖励派生/训练运行于 `vla-liberox`，只有 4B 模型评价运行于 `rynnvalue-reward`。浏览器不会接收或执行任意配置路径/命令；后端只根据表单白名单生成并保存 `effective_config.yaml`。

使用流程如下：

1. 打开侧栏“数据集”，先选择一个任务。轨迹表将来源明确分为“原始推理”“人工接管”“二次推理”和不可训练的“错误/未完成”；人工/二次推理分支会显示策略前缀、`resume_step` 以及实际进入训练的后缀长度。详情页可把轨迹标记为“测试数据”。测试数据仍可浏览和显式单条评价，但会被新的训练数据集打包、任务批量评价和 offline-RL 导出自动跳过。
2. 点击“创建训练数据集”，选择随机、按时间顺序、分类配额或手动勾选。预览会先排除测试数据，再给出 M、预计 action/chunk 数和分类构成；确认后生成不可变、单任务数据集。修改成员必须使用“派生版本”，不会覆盖旧版本。未标注版本可“取消冻结”，已结束标注的版本可“删除数据集”；存在活动任务或派生子版本时会拒绝删除。若已有训练历史，页面会要求第二次确认；强制删除仍保留训练输出、checkpoint 和 policy overlay，只在训练记录中标记源数据集已删除。删除不会移除源轨迹或全局共享奖励缓存。
3. 点击“验证完整性”会重新计算 `run.json`、trajectory 和双视角 observation 的大小及 SHA-256。普通删除被引用轨迹时返回冲突并列出数据集；确认强制删除后关联数据集立即变为 `BROKEN`，不能继续标注或训练。
4. 创建/派生数据集只冻结成员，**不自动评价**。点击每行右侧“配置”，原地展开圆角表单。选择“成功后动作：保留／截断”，点击旁边的“保存”即可更新当前数据集的训练取样配置。默认保留；截断到连续 N 次 `done=True` 的第 N 个动作（默认 N=5），而不是第一次 True。完整视频、关键帧、评价数组及成员哈希不变，无需重新评价；页面数量仍指完整记录。设置只影响之后创建的任务，已注册、排队和运行的训练使用各自快照。创建/派生数据集时也能选择此项。评价类型有四种：
   - **Final Reward**：一行配置相加/相乘、p、α（仅相加）、κ、γ 和 cumulative reward（仅相加）；只复用已保存模型输出与最新关键帧做 CPU 重算。κ=0 不读取 RynnValue；相加 α=0 不读取 Stage。缺少所需输入或 Stage>0 时明确报错。
   - **RynnValue**：配置 max_frames、评价 batch size、γ、κ 和 cumulative，默认复用兼容模型输出，可显式开启重新运行模型。
   - **Robometer**：配置 fps 和 batch size，前缀固定 4 帧；只用于诊断。
   - **All**：配置融合与两个模型的参数，强制按 RynnValue → Robometer → Final Reward 顺序运行；全部成功才切换三个当前结果，中途失败保留旧结果。模型名称及固定 revision 只读。
   任务窗口关闭或页面刷新不会停止后台任务。
5. 每个数据集分别保存 Final Reward、RynnValue、Robometer，关键帧仍保存在原轨迹。重新评价仅替换所选结果，不覆盖另一个模型的原始输出；All 同时更新三者。详情按数据集同类型结果优先、缺少时继承全局；全局保存首次结果，只有显式“同步覆盖全局评价”才替换。**Original Final Reward** 折线图保留 sparse、Shape、original final 三条曲线；**Final Reward** 独立显示融合结果。两图在有融合结果时使用同一组 γ/cumulative/κ 以便对照。RynnValue 原始距离、entropy 和 Robometer 曲线保持独立，不因融合而改写。
6. 侧栏“训练”选择数据集后，IQL 的“训练奖励”可选择 **Final Reward**（默认，融合并缩放后的奖励）或 **RynnValue**（原始 sparse + κ·Shape Reward，不加入 Stage 融合）。两种结果独立保留；按所选类型优先使用数据集评价，没有专属结果时继承同类型全局评价。切换来源会加载对应参数并绑定对应结果，不改变数据集默认评价，也不重新运行模型。仅有 RynnValue 时可直接选择它训练，无需先生成 Final Reward；Robometer 仍只用于诊断。缺少、损坏或不匹配的所选奖励会阻止训练，不回退到另一种。
   p、α、κ、形式在数据集配置；γ 在数据集与训练页都能调整，cumulative 对 RynnValue 和相加 Final Reward 可用。选择 Final Reward 且继承的全局结果包含相乘成员时，整次 UI 训练统一 macro；独立 CLI 若混合结果与 cumulative=On 冲突则明确报错。训练使用固定语义快照，修改 γ/cumulative 只在该训练目录重算私有数组，不覆盖数据集当前结果或历史训练。页面只读轻量摘要，不反复解压 observations；训练启动前完整校验源文件。旧显式来源 YAML 仍兼容，原始完整轨迹与 IQL 更新公式不变。
7. “训练配置”表单始终显示，任务／难度／提示词筛选在桌面端压缩为一行。配置数据集、batch、训练步数及高级 IQL 参数后点击“注册训练任务”；注册后保留当前参数，直接修改并注册下一批，无需另外新增任务或展开表单。运行中也可以继续注册，不需要等待上一轮完成。每项在注册时固定有效配置和奖励引用，之后修改草稿或重新评价数据集不会改变已注册任务。UI 暂不提供断点恢复；各批次从配置的基础模型独立训练，不继承上一批权重，CLI 的显式恢复功能保留。
8. “批量训练”按注册顺序串行执行。当前任务完成、失败或手动停止后自动运行下一项；“停止本轮”只影响当前任务，“取消排队”只移除选中的等待项。仿真、控制器或其他 GPU 任务占用资源时等待，不抢占正在运行的任务。排队任务引用的数据集不能删除。列表展示待完成任务和最近 20 条训练记录，展开可检查已注册参数；详细日志仍由单个任务监视器展示。
9. 任务监视器实时显示阶段、step、速度、已用时间、滚动 ETA/预计完成时间、Q/value/actor loss、Q/V/advantage、advantage weight、学习率、梯度范数和峰值显存。默认跟随当前训练；手动查看其他任务后可点击“跟随当前训练”恢复。安全停止沿用现有 checkpoint 保存逻辑。调度由后端执行，关闭网页不影响队列；后端关闭时已启动的独立训练进程继续运行，等待任务在后端重新启动后恢复调度（不是恢复失败任务的 checkpoint）。夜间连续训练需保持设备和后端服务运行。完成后可回到仿真平台选择发布的 policy overlay。

TensorBoard 按需由平台用 `vla-liberox` 启动并覆盖所有受管训练目录，固定访问 `http://127.0.0.1:6006/`。它不占用 GPU 任务锁，可与仿真并存；若端口被其他服务占用，页面会明确报错而不会结束那个进程。

需要 GPU 的仿真、模型评价、训练和批量测试共享跨进程 GPU 文件锁；直接启动的冲突请求返回 `409`，已注册的训练与测试则等待资源并按共同注册顺序串行执行。开始这些离线 GPU 任务前平台会卸载驻留 VLA，完成后不自动重载，下一次仿真按需加载。Final Reward 重算不加载模型、不占用 GPU 锁，仍受后台任务调度保护。后台任务使用独立进程组，PID、心跳、日志与状态均落盘，所以 UI 后端重启不会主动终止它。TensorBoard 是只读进程，不占用 GPU 锁。

平台持久化目录为：

```text
dataset-root/projects/libero_x_vla/
├── datasets/<dataset_id>/dataset.json
├── annotation-cache/<content_hash>.{json,npz}
├── training/<training_id>/
├── jobs/<job_id>/{job.json,job.log,effective_config.yaml}
└── runs/...
```

生产 UI 的 `frontend/dist` 仍是本机构建产物。拉取包含此页面的代码后，直接重启 `run_ui.py` 会检测源码指纹并运行 `npm run build`；新机器应先在 `liberox-vla-adapter-terminal/frontend` 执行 `npm ci`。

#### Robometer 独立轨迹评价

Robometer 使用独立的 `robometer-reward` 环境和
`aliangdw/Robometer-4B-LIBERO`，安装步骤见
[`vla-adapter-robometer/README.md`](vla-adapter-robometer/README.md)。数据集页面的评价区可分别勾选
`RynnValue` 与 `Robometer`；同时勾选时后台按勾选顺序串行加载两个 4B 模型，不会让它们同时占用 GPU。批量评价和“评价所选”默认分别跳过已有有效 sidecar；显式开启覆盖后可以更新所选评价器的全局结果，不影响另一评价器。

Robometer 完整读取 `agentview` observation，在 20 Hz 原时间轴上以 3 Hz 选取评价点，始终包含首帧和末帧；每个评价点按官方 `use_frame_steps` 语义使用从起点到当前点的前缀，并均匀选择 4 帧。首次 `done=true` 不会截断评价。每条 episode 独立保存：

```text
robometer_evaluation.json   # 模型/revision/官方代码 commit、配置与输入 hash
robometer_evaluation.npz    # step、真实秒数、progress_pred、success_probs
```

详情页的 Robometer Progress 与 Success Probability 均为模型原始单轨迹输出。对比卡中的 RynnValue 进度
`clip(1-d(t)/d(0), 0, 1)` 和真实秒数插值只用于 UI 显示，不写回 sidecar，也不进入训练。冻结数据集配置可单独生成 Robometer 诊断版本，与训练奖励版本相互独立；不参与 IQL reward、样本筛选或训练 manifest。有效模型输出可跨版本复用，更换输出目录或执行 batch size 不会单独导致模型重载。

首次配置环境：

```bash
conda create -n robometer-reward python=3.10 -y
conda activate robometer-reward

git clone https://github.com/robometer/robometer.git Robometer
git -C Robometer checkout 352d160389daa964788de1ec933d1925f3a6de4f

pip install -e ./Robometer
pip install -e ./vla-adapter-robometer
```

评价参数位于 `vla-adapter-robometer/configs/robometer_evaluation.yaml`：

```yaml
schema_version: 1
paths:
  selection_manifest: null
  output_dir: ../outputs/evaluations
  robometer_root: ../../Robometer
model:
  checkpoint: aliangdw/Robometer-4B-LIBERO
  revision: bb7dce7e6bde3bd236c0fbe0be46fdf19b57b873
  robometer_commit: 352d160389daa964788de1ec933d1925f3a6de4f
  device: cuda:0
  dtype: bfloat16
evaluation:
  control_hz: 20
  fps: 3.0
  prefix_frames: 4
  batch_size: 8
```

各字段含义：

- `selection_manifest`：UI 后台为当前勾选轨迹生成的不可变选择清单。模板保持 `null`；GUI 作业会生成 effective YAML 并自动填入，只有直接运行脚本时才需要手动指定。
- `output_dir`：当前作业的临时输出目录；完成校验后，JSON/NPZ 会原子绑定到源 episode。
- `robometer_root`：官方 Robometer checkout 的路径，不是 `vla-adapter-robometer` 适配项目路径。
- `checkpoint + revision + robometer_commit`：共同定义评价器版本。两个 revision/commit 必须是完整 40 位哈希，任一变化都会使旧评价失效。
- `device`：明确选择一张 GPU，例如服务器上可改为 `cuda:1`；不自动跨卡或回退 CPU。
- `dtype`：第一版固定 `bfloat16`。
- `control_hz`：原始轨迹的控制频率，当前固定 20 Hz，不能用它改变评价密度。
- `fps`：Robometer 评价点密度，默认 3 Hz；25 秒记录约产生 76 个评价点，且强制包含首尾。
- `prefix_frames`：每个评价时刻从完整历史前缀均匀取 4 帧，固定为官方协议，不等同于 batch size。
- `batch_size`：一次送入 GPU 的 prefix 样本数。显存不足时可从 `8` 降为 `4/2/1`；只影响速度和峰值显存，不改变评价时间点及语义。

GUI 后端还需要在 `configs/ui_config.yaml` 保留：

```yaml
robometer_root: ../vla-adapter-robometer
robometer_environment: robometer-reward
```

这里的 `robometer_root` 指向适配项目，用于寻找脚本和默认 YAML；适配 YAML 内的 `paths.robometer_root` 才指向官方仓库。官方 checkout 或 Conda 环境缺失时，页面会禁用 Robometer 并提示原因，但不会影响 RynnValue 评价。

### 4.10 在 Web UI 中批量测试策略

侧栏“测试”位于“训练”之后，用于对基础 VLA 或训练发布的 policy overlay 进行可复现的批量成功率评估。它与主页单次仿真不同：一次测试固定选择**一个任务和一个策略**，不保存可回放轨迹或视觉文件，也不会进入运行记录或训练数据集。

创建测试时配置：

- `trials`：仿真回合数，默认 100，有效范围 `1..1000`；
- `max_steps`：每个回合的固定控制步数；
- `open_loop_steps`：每次 VLA 预测后实际执行的动作数，有效范围 `1..8`；
- 初始状态池：默认使用当前 BDDL 任务的全部合法 `init_state_index`，也可在高级设置中选择子集或固定一个；
- `base_seed` 与 `seed_count`：环境 seed 池为 `base_seed .. base_seed + seed_count - 1`；默认 `seed_count=ceil(trials/init_state_count)`，设为 1 即固定 seed；
- `schedule_seed`：只控制完整测试顺序的可复现打乱；
- 实时模式：默认按墙钟严格限制为 20 Hz；关闭后可加速运行，但 MuJoCo 的 `control_freq=20`、动作与模拟时间语义不变。

这里的“随机环境”只表示所选任务有限的 benchmark 初始状态，不会切换到同一 LEVEL 下其他 BDDL 场景。后端在开始前冻结完整调度，对初始状态池和 seed 池做确定性的均衡组合轮转；任一状态、seed 及可用组合的分配次数差不超过 1。预览区会先显示分布、预计时长和前若干组合，正式运行期间 schedule 不再改变。

点击“预览调度”后可“注册测试任务”；注册时固定本次参数、策略信息和调度哈希。表单保持显示，参数不变时可以再次注册；修改参数后需重新预览。正在运行训练或测试时仍可注册，注册本身不会加载模型或抢占 GPU。“测试队列”按注册顺序运行，与训练共用调度器；关闭网页不影响执行，后端重启后恢复等待任务。排队项可“取消排队”，当前项可“停止本轮”；完成、失败或停止只影响该任务，当前进程退出并释放 GPU 后继续下一项。排队或运行中的测试必须先取消/停止才能删除，其引用的模型也不能通过平台删除或改名。

测试的 headline 成功语义是：环境 `done=true` 必须连续出现 5 个控制步才确认成功。短暂命中后出现 `false` 会重新计数；达到 5 步后成功状态锁存，即使后续 `done=false` 也不撤销。无论是否已确认成功，每个正常回合都继续执行到 `max_steps`，因此该指标不等同于“首次 done 后立即结束”的官方 LIBERO benchmark。单回合错误会记录后继续测试，连续 3 个回合错误时提前结束整次任务；错误回合计入已尝试回合和成功率分母。

运行监视器显示模型加载、当前回合、`init_state_index`、seed、完成进度、成功数、实时成功率、ETA、控制频率和滚动日志。默认跟随当前测试；手动查看排队项后可点击“跟随当前测试”恢复。历史详情包含总体成功率与 Wilson 95% 置信区间、完成覆盖率，以及按初始状态、seed、状态×seed 组合的分组成功率和逐回合关键数值。取消的部分测试只对已尝试回合计算临时成功率，并单独显示覆盖率，不能与完整测试混为一谈。页面刷新会恢复运行中和等待中的测试；已结束记录从历史表选择查看，可在终态下通过重复输入测试 ID 二次确认后永久删除。

每次测试唯一的结果 artifact 是：

```text
dataset-root/projects/libero_x_vla/evaluations/
└── <task_name>/YYYY-MM-DD/<timestamp>__<evaluation_id>/evaluation.json
```

`evaluation.json` 原子更新并保存任务/策略快照、完整有效配置、冻结 schedule 及哈希、逐回合数值、成功率分组、模型加载与总耗时。测试不会生成视频、图像、observation、action、trajectory、NPZ、CSV 或图表。`jobs/<evaluation_id>/` 中的 `job.json`、`job.log` 和 `effective_config.yaml` 只是 detached job 的恢复与诊断信息，不属于测试结果。测试与仿真、RynnValue 标注和 IQL 训练使用同一 GPU 互斥锁；TensorBoard 不受影响。

## 5. 为什么这样适配

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

## 6. 使用 LIBERO-X 训练数据微调


- VLA-Adapter：https://github.com/OpenHelix-Team/VLA-Adapter
- LIBERO-X：https://github.com/meituan/LIBERO-X
- LIBERO-X 数据：https://huggingface.co/datasets/meituan/LIBERO-X
- VLA-Adapter Object-Pro：https://huggingface.co/VLA-Adapter/LIBERO-Object-Pro
