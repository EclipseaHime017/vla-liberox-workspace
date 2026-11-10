# 服务器闭环评测

入口为 `vla-adapter-rynn-iql/scripts/evaluate_server.py`，配置为 `configs/server_evaluation.yaml`（位于同一模块内）。每个导出模型在自己的 Conda 环境中运行，依次执行所有任务。支持当前注册的两个模型家族及其训练导出，不依赖平台后端、前端或数据库。

使用训练结果中的完整 `policy-registry/<模型ID>/policy.yaml` 及同目录权重。`checkpoints/step_*/state.pt` 是训练恢复文件，不能直接作为评测模型。跨设备迁移时先用 `scripts/export_policy.py` 更新基础权重的位置；加载时会检查基础模型身份、子模型权重哈希和冻结的输入输出配置，不回退到其他模型。

## 配置和结果

- `policies`：完整导出模型的 `policy.yaml` 路径列表。
- `tasks`：每项包含 `level`、`task_name`、`max_steps`、`init_state_indices`。指令从对应 BDDL 读取。
- `trials`、`base_seed`、`seed_count`、`schedule_seed`：生成与 main 相同的平衡测试安排；每个模型使用相同安排。
- `open_loop_steps`：`null` 使用模型变体默认值，后训练模型还受监督长度约束；显式填写时检查预测长度上限。
- `success_streak`：默认连续 5 步环境成功，确认后锁定成功，但继续执行完整回合。
- `render_backend`：`egl` 或 `osmesa`。`check_environment` 仅决定 `--check-render` 使用的环境。
- `output_root`：每次运行建立独立结果目录；配置中的相对路径均相对于 YAML 所在目录。

每回合使用主视角和手腕视角的 256×256 图像，在模型预测时重新渲染。视角翻转、图像处理、动作转换来自模型冻结的配置和家族实现；仿真控制频率为 20 Hz，运行速度不受墙钟限速。

`summary.json` 汇总模型与任务；每个模型的 `result.json` 保存模型清单、任务文件哈希、执行安排、渲染器信息、逐回合结果、成功率和置信区间。运行错误单独计数并进入已尝试回合的分母，同时显示完成覆盖率，不能把未执行回合当成正常失败。共享初始状态的试验存在相关性，Wilson 区间仅作为辅助指标，并保留逐初始状态统计。

与 main 的批量评测保持相同的状态恢复、均衡采样、连续成功与全回合执行规则。当前移植参考为 main `c62dcc9`。仿真依赖版本及 LIBERO-X 提交锁定在 `configs/dependency-lock.yaml`，模型推理仍由各自后端维护。

## 无头服务器

LIBERO-X 使用 MuJoCo 的 OpenGL 渲染，不需要桌面或显示器。Linux EGL 提供 GPU 离屏渲染，OSMesa 提供 CPU 软件渲染。[MuJoCo 文档](https://mujoco.readthedocs.io/en/3.3.7/programming.html#using-opengl)

A100/A800 上能否使用 EGL 取决于实际驱动、GPU 分配方式和容器暴露的图形库，不能用 CUDA 推理或 `nvidia-smi` 成功代替渲染验收。Ampere GPU 的 MIG 实例不支持 OpenGL；这类分配需改用整卡或显式使用 OSMesa。[NVIDIA MIG 说明](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html)

容器必须在启动时暴露 NVIDIA `graphics` 能力，例如由容器运行时设置 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`，并提供与宿主驱动匹配的 NVIDIA EGL 库和 GLVND 厂商配置。只在已启动的容器终端中 `export` 这个变量不会补挂缺失的库；不要用 CUDA 包替代图形驱动。[NVIDIA 容器配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)

Ubuntu 的通用加载器可安装 `libegl1 libglvnd0 libgl1`；CPU 渲染另外需要 `libosmesa6`。NVIDIA 的 `libEGL_nvidia`、`libnvidia-eglcore` 和 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 应由匹配的宿主驱动/容器运行时提供。

`--check-render` 实际创建 LIBERO-X 场景，读取两个相机的图像并执行一个控制步，记录 GL vendor/renderer/version。EGL 模式若使用了软件渲染器会明确失败；切换 OSMesa 必须显式修改配置，结果也记录该区别。OSMesa 仅改变渲染方式，模型推理仍使用 GPU。

## PBS 与资源使用

先安装评测依赖并准备 LIBERO-X，再在 PBS 分配的 GPU 计算节点中运行上述预检和评测命令。登录节点检查不能证明计算节点具备 GPU 渲染能力。评测按单卡、逐模型运行，无需 `torchrun`；使用 PBS 分配的 GPU，不在登录节点运行模型。

保持调度器的 `CUDA_VISIBLE_DEVICES` 分配。若需指定渲染设备，在启动进程前设置 `MUJOCO_EGL_DEVICE_ID`，它是 EGL 的设备枚举索引，不要直接当作 CUDA 进程内的逻辑编号。通过预检确认渲染器后再进行正式评测。

评测复用 server 的资源锁，已有训练占用资源时直接提示忙碌；任务可放在训练完成后的作业中。Ctrl+C 或 SIGTERM 取消当前评测，保留已完成回合和错误日志。新入口本身不申请 PBS 资源，也不改变已有训练队列。
