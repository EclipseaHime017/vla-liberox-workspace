# 平台使用

## 启动

```bash
conda activate vla-liberox
python -m studio
```

浏览器打开 **http://127.0.0.1:8000**。前端源码有变化时，启动时自动重新构建 `web/`。地址、端口、数据目录和各环境名在 `configs/studio/ui.yaml`，仿真默认设置在 `configs/studio/studio.yaml`；修改后重启平台。

网页操作都由后端执行，仿真、评价、训练和测试共用一个 GPU 工作队列。

## 页面

**控制台：仿真与人工接管**

- 点击「创建仿真」，依次选择任务、难度、提示词和模型，然后开始；页面上设置运行步数、随机种子和初始状态。
- 已有记录可以按模型家族和具体模型筛选，播放录像、拖到指定位置，再选择重新推理或人工接管。
- 人工接管前选择 SpaceMouse 或 FACTR，完成校准后启动。SpaceMouse 可切换世界坐标／工具坐标，并用滑杆调整灵敏度。接管结果保存为新记录，原记录不变。
- 完成的分支可「设为主线」后继续操作，可以从任意已保存帧继续，包括最后一帧。续跑的总步数包含已保存的前缀。
- 策略统一使用所选任务的完整提示词。任务注册见 [任务](tasks.md)。

**运行记录与关键帧**

- 「运行记录」检索历史仿真和接管记录；「数据集」中的轨迹详情可查看录像、动作、评价曲线和奖励。
- 在详情中进入「切片」，拖到目标位置，添加 positive 或 negative 标记后保存。
- 标注可保存为全局标注，或保存到指定数据集；所选归属会保持，便于连续标注。

**评价与打包数据集**

- 在「数据集 → 轨迹评价」中选择任务和记录，勾选 RynnValue、Robometer 后批量评价。需要重算时勾选「覆盖已有评价」。
- 在「打包训练数据集」中选择记录并命名，预览后「冻结数据集」。新数据集默认复用轨迹已有的全局评价；缺少关键帧时自动继承全局标注。所需奖励或关键帧在全局也不存在时，会列出缺失轨迹并停止相关评价或训练。
- 已打包数据集的「配置」可以：
  - 保留或截断成功后的动作；
  - 重新生成 RynnValue、Robometer 或 Final Reward；
  - 调整 Stage 与 RynnValue 的组合，只改公式时复用已有模型评价和关键帧。
- 「验证完整性」在后台检查原始文件，分别列出全局、数据集专属和实际可用的奖励。
- 「导出数据集」打包数据和评价结果，见 [数据](data.md)。

奖励的定义见 [奖励](rewards.md)。

**训练、模型与测试**

- 「训练」：选择数据集、基础模型和训练方法后创建任务，见 [训练](training.md)。方法列表来自后端的方法目录（`GET /api/methods`）。
- 「模型」：查看基础模型和训练结果。训练模型只关联生成它的那次训练，可查看方法、数据集、步数和种子，并打开或下载当次的有效 YAML。
- 「测试」：选择模型、任务和次数，查看成功率与各回合结果，见 [评测](evaluation.md)。

**工具：IQL Critic 诊断**

先用所选模型采集独立轨迹，再只训练 Q/V，不更新、不导出 actor。结果包括曲线、逐 chunk 的 Q/V 与优势（CSV/NPZ）、critic checkpoint 和配置。新 rollout 不生成奖励，只展示 critic 的估计。IQL 的 V 是数据动作上的 expectile，不是所选策略的真实回报。产物只保存在该工具任务的目录，删除任务即一并清理。

**工具：VF 回报分布**

只对带 Value Flows critic 的已训练模型可用，不训练。先用所选模型跑一条仿真，记录每次查询采样的全部候选块、各自的 critic 分数和执行的那一个（需要 GPU）；再用模型导出的 critic 对每个候选做完整 flow 积分，采样回报分布（只用 CPU）。

- 回放：主视角视频旁显示当前查询的回报直方图（选中块填充，其余候选为轮廓，点线为选择时那一次单步采样的分数）和选中块分布的中位数与 10–90% 区间随时间的变化；点击时间轴可跳转。
- 实际回报：critic 的奖励配方不需要奖励模型的评价时（如 Sparse），画出这条轨迹从每次查询起的折扣回报，作为单个样本对照；否则注明原因。
- 下载：并排视频 `returns.mp4`、分布数组 `returns.npz`、回放数据 `returns.json`、摘要和配置。

产物保存在该工具任务的目录，删除任务即一并清理。

## 接管设备

**SpaceMouse**

安装见 [安装](install.md#4-需要人工完成的步骤)。配置在 `configs/devices/spacemouse.yaml`：

- `mode`：先用 `device` 检查设备，再用 `simulation` 检查仿真。
- `motion_mode`：默认 `exclusive`（平移与旋转按主导意图互斥），`combined` 为六轴联动。
- `control_frame`：`world` 为世界坐标；`tool` 为末端工具坐标，前／左／上对应 Panda 夹爪的 +Y／+X／−Z。
- 两类灵敏度独立保存。网页中也可切换坐标，运行中需松开摇杆回中后生效。

```bash
python -m studio check-spacemouse
```

**FACTR**

配置在 `configs/devices/factr.yaml`，同样先 `mode: device` 再 `simulation`。

```bash
python -m studio setup-factr            # 首次：下载锁定的官方源码，建立隔离运行时
python -m studio setup-factr --check    # 只核对，不访问设备
python -m studio check-factr            # 终端校准与控制测试
```

USB 延迟：网页中点击「校准控制器」时，若 latency 不是 1 ms，会弹出本机系统授权窗口，安装永久规则后继续校准。需要本机桌面和 polkit 认证代理。SSH、无桌面或远程浏览器时，在终端执行下面的命令，然后退出控制程序、支撑主臂并重插 USB：

```bash
python -m studio setup-factr --install-usb-rule
```

规则匹配 YAML 中的 VID/PID。`serial_number: null` 时覆盖所有同 ID 设备（包括其他 FTDI 转接器）；配置了序列号时只作用于那一台。

校准与测试：

1. 将整臂摆到官方 Figure 1 参考构型（七轴角 `[0, -0.7854, 0, -2.356, 0, 1.57, 0]` rad），松开触发器。
2. 输入 `c`，一次采集七轴与触发器零点。有效校准可以复用。
3. 输入 `s` 开始测试。其他按键：
   - `g`：开启重力补偿（保留安全检查）；`d`：关闭补偿；
   - `q`／`Ctrl+C`：先撤力再退出，会立即撤力，请提前支撑好；
   - `i`：查看状态。
4. `simulation` 模式下，实体主臂会缓慢对齐静止的仿真机械臂，倒计时后七个关节 1:1 跟随，周围请留出空间。网页走同一关节路径，并保存标准人工轨迹。

## 命令

| 命令 | 用途 |
| --- | --- |
| `python -m studio` | 启动平台（即 `serve`） |
| `python -m studio check-model --model <ID>` | 平台运行时，对一个模型做仿真、短训练和导出加载检查；结束后自动清理本次产生的记录、数据集、模型和任务 |
| `python -m studio migrate-run-layout [--apply\|--rollback]` | 预览、迁移或回滚旧录制目录，见 [数据](data.md) |
| `python -m studio setup-factr`、`check-factr`、`check-spacemouse` | 接管设备 |
| `python -m studio evaluate-policies`、`critic-diagnostic` | 后台任务进程，由平台队列启动 |
