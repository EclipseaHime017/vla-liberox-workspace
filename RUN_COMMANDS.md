# 快速启动指令

以下命令均从 workspace 根目录执行：

```bash
cd ~/eclipseaws/vla-liberox-workspace
```

## 个人端：启动仿真 UI

配置文件：`configs/config.yaml`、`configs/ui_config.yaml`

```bash
conda activate vla-liberox
python liberox-vla-adapter-terminal/scripts/run_ui.py
```

访问：`http://127.0.0.1:8000`

## 个人端：FACTR 校准与重力补偿测试

配置文件：`configs/factr_test_config.yaml`（先 `mode: device`，通过后再 `simulation`）

```bash
conda activate vla-liberox
pip install -r liberox-vla-adapter-terminal/requirements-factr.txt
python liberox-vla-adapter-terminal/scripts/setup_factr.py  # 首次下载官方源码/配置隔离环境
python liberox-vla-adapter-terminal/scripts/setup_factr.py --check  # 不触碰设备
python liberox-vla-adapter-terminal/scripts/test_factr.py
```

GUI 直接点击原“校准控制器”：latency 已为 1 ms 时不提权；否则在本机系统授权窗口输入密码，自动安装永久规则并应用当前设备，然后继续校准，不自动上力、不需重插。没有新增修复按钮或网页密码框；拒绝/取消/超时不校准。需本机桌面及 `pkexec`/polkit 认证代理；SSH/headless、远程浏览器或 CLI 使用终端备用：

```bash
python liberox-vla-adapter-terminal/scripts/setup_factr.py --install-usb-rule
```

终端安装后先退出控制程序、支撑主臂并重插 USB，再测试。规则匹配 YAML VID/PID，`serial_number: null` 时覆盖所有同 ID 设备（包括其他 FTDI 转接器）；只在配置序列号时限定单台。`add|bind` 时自动设为 1 ms，不固定端口号；手动 `echo ... | sudo tee .../latency_timer` 仅临时应急。

先将整臂摆到官方 Figure 1 参考构型（固定七轴角 `[0, -0.7854, 0, -2.356, 0, 1.57, 0]` rad）并松开触发器，再输入 `c` 一次采集七轴与触发器零点（官方行程 0.8 rad，不再分步采集端点）；只保留这一套校准，不再逐电机标定或任意姿态置零。已有有效校准可复用，然后 `s` 开始测试。重力补偿使用同一校准，`g` 直接请求出力，无 `ENABLE` 二次确认，但保留安全检查；`d` 直接关闭补偿；`q`/`Ctrl+C` 先关闭电机输出再退出整个测试程序，均无二次确认。`i` 查看状态。`Ctrl+C` 会立即撤力，请准备支撑。详见 [FACTR 使用说明](README_CN.md#361-factr-franka-校准手动重力补偿与无-vla-测试)。

`mode: simulation`：先 `c` 校准、`g` 开补偿，再 `s` 启动。实体主臂会缓慢对齐静止的仿真机械臂，请留出空间；倒计时后七关节按 1:1 跟随。独立测试不记录；GUI 走同一关节路径并保存标准人工轨迹，正常完成保留补偿。SpaceMouse 采集不受影响。

## 服务器端：启动终端训练

配置文件：`vla-adapter-rynn-iql/configs/terminal_pipeline.yaml`

`overrides.training.method` 选择 `iql` 或 `bc`；BC 跳过所有奖励评价，数据筛选仍使用 `selection`。

```bash
python vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml
```

脚本显示数据选择、缓存和训练计划后会提示：

```text
Start this pipeline? [y/N]
```

- 输入 `y` 或 `yes`：确认并开始执行；
- 直接按 Enter，或输入其他内容：取消执行。

启动前只检查配置和数据，不执行训练：

```bash
python vla-adapter-rynn-iql/scripts/train_terminal.py \
  --config vla-adapter-rynn-iql/configs/terminal_pipeline.yaml \
  --dry-run
```
