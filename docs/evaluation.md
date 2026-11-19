# 评测

所有评测使用同一套规则：按初始状态均衡安排试验、恢复初始状态后运行完整回合、连续 `success_streak` 步（默认 5）成功才确认成功，确认后仍执行完整回合。仿真控制频率 20 Hz，不按墙钟限速。

## 1. 平台测试页

在「测试」页面选择模型、任务和次数后提交，任务进入平台的统一队列，结果显示成功率、各初始状态的成功率及其平均值，以及逐回合结果。共享初始状态的试验之间存在相关性，Wilson 区间依赖回合独立假设，只作参考。

## 2. 无头闭环评测（服务器）

不依赖平台后端、前端或数据库。每个导出模型在自己的 Conda 环境中依次执行所有任务，支持已登记的两个模型家族及其训练导出。安装：`python envs/setup.py --role eval --family <家族>`。

```bash
python -m cluster evaluate --config configs/cluster/evaluation.yaml --check-render   # 渲染预检
python -m cluster evaluate --config configs/cluster/evaluation.yaml
```

使用训练结果中完整的 `policy-registry/<模型ID>/policy.yaml` 及同目录权重；`checkpoints/step_*/trainer.pt` 是训练恢复文件，不能直接评测。跨设备迁移时先用 `python -m train export-policy` 更新基础权重的位置。加载时检查基础模型身份、子模型权重哈希和冻结的输入输出配置，不回退到其他模型。

**配置（`configs/cluster/evaluation.yaml`）**

| 字段 | 含义 |
| --- | --- |
| `policies` | 导出模型的 `policy.yaml` 路径列表 |
| `tasks` | 每项包含 `level`、`task_name`、`max_steps`、`init_state_indices`；指令从对应 BDDL 读取 |
| `trials`、`base_seed`、`seed_count`、`schedule_seed` | 生成与平台相同的均衡测试安排，每个模型使用同一安排 |
| `open_loop_steps` | `null` 使用模型变体的默认值（后训练模型还受监督长度约束）；显式填写时检查预测长度上限 |
| `success_streak` | 连续成功步数 |
| `render_backend` | `egl`（GPU 离屏）或 `osmesa`（CPU 软件渲染）；`check_environment` 只决定 `--check-render` 使用的环境 |
| `output_root`、`liberox_root`、`vla_adapter_root` | 结果目录与上游路径，相对本 YAML |

每回合在模型预测时重新渲染主视角和手腕视角的 256×256 图像；翻转、图像处理和动作转换来自模型冻结的配置和家族实现。带 critic 的策略（IFQL）每次预测采样 `selector.num_samples` 个动作块，按实际执行的 `open_loop_steps` 打分后选择，单次推理更慢。仿真依赖版本和 LIBERO-X commit 由 `third_party/lock.yaml` 锁定，启动时核对。

**结果**：`summary.json` 汇总模型与任务；每个模型的 `result.json` 保存模型清单、任务文件哈希、执行安排、渲染器信息、逐回合结果、成功率和置信区间。运行错误单独计数并计入已尝试回合的分母，同时给出完成覆盖率，未执行的回合不算作失败。

### 无头渲染

LIBERO-X 使用 MuJoCo 的 OpenGL 渲染，不需要显示器：EGL 用 GPU 离屏渲染，OSMesa 用 CPU 软件渲染（模型推理仍用 GPU）。

- A100/A800 能否使用 EGL 取决于驱动、GPU 分配方式和容器暴露的图形库；CUDA 推理或 `nvidia-smi` 正常不能代替渲染验收。MIG 实例不支持 OpenGL，需改用整卡或 OSMesa。
- 容器启动时必须暴露 NVIDIA `graphics` 能力（例如 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`），并提供与宿主驱动匹配的 NVIDIA EGL 库和 GLVND 厂商配置。在已启动的容器里 `export` 这个变量不会补上缺失的库。
- Ubuntu 的通用加载器：`libegl1 libglvnd0 libgl1`；CPU 渲染另需 `libosmesa6`。NVIDIA 的 `libEGL_nvidia`、`libnvidia-eglcore` 和 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 由宿主驱动或容器运行时提供。
- `--check-render` 实际创建场景、读取两个相机并执行一步，记录 GL vendor/renderer/version。EGL 模式落到软件渲染器时明确失败；切换 OSMesa 必须显式修改配置。

### 资源

- 在 PBS 分配的 GPU 计算节点上运行预检和评测；登录节点的检查不能证明计算节点能渲染。
- 评测按单卡、逐模型运行，不需要 `torchrun`。保持调度器的 `CUDA_VISIBLE_DEVICES`；指定渲染设备时在启动前设置 `MUJOCO_EGL_DEVICE_ID`（EGL 的设备枚举索引，不是 CUDA 逻辑编号）。
- 评测复用训练的资源锁，资源被训练占用时直接提示忙碌。`Ctrl+C` 或 SIGTERM 取消当前评测，保留已完成的回合和错误日志。

## 3. 本地仿真、远程推理

服务器没有渲染环境时，可以在本地运行 LIBERO-X，只把双视角图像（无损 PNG）、机器人状态和任务指令发给服务器。服务器加载导出模型并返回动作块；翻转和动作转换仍由服务器上的模型接口完成。两端使用同一版本的代码；远端只需 `infer` 角色的环境，本地不需要下载策略权重。

1. 生成一个仅本人可读的随机令牌文件，安全复制到两端。
2. 远端在分配的 GPU 上启动服务（只监听 `127.0.0.1`，默认空闲 15 分钟退出；每个服务绑定一个模型和一张 GPU，可用不同端口启动多个副本）：

   ```bash
   python -m cluster serve --policy /path/to/policy-registry/<ID>/policy.yaml \
     --gpu 0 --port 18210 --token-file /private/path/evaluation.token
   ```

3. 本地建立 SSH 转发（需要跳板机时沿用已有的 SSH 配置）：

   ```bash
   ssh -N -o ExitOnForwardFailure=yes -L 18210:127.0.0.1:18210 <训练主机>
   ```

4. 用服务器上 `sha256sum policy.yaml` 的结果运行本地评测：

   ```bash
   python -m cluster evaluate-remote --config configs/cluster/evaluation.yaml \
     --endpoint http://127.0.0.1:18210 --token-file /private/path/evaluation.token \
     --policy-sha256 <MANIFEST_SHA256> --concurrency 4
   ```

此模式的模型由远端服务和清单哈希指定，不读取配置中的 `policies`；任务、次数、初始状态、执行步数和输出位置仍用同一份 YAML，执行步数随每次请求发给服务端，供带 critic 的策略打分。多个 `--endpoint` 分配同一模型的副本。每个回合有独立的随机数状态和递增的请求序号；超时或断线明确报错，不重放请求。`summary.json` 另外记录往返时间、远端推理与排队时间、请求大小和显存。多个回合共享一个服务时推理串行处理，正式评测前先用短回合压测并发量。
