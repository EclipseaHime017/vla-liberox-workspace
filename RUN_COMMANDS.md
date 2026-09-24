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

输入 `c` 按官方文档摆整臂近似参考构型，松开触发器后一次采集七轴与触发器零点（官方行程 0.8 rad，不再分步采集端点）；只保留这一套校准，不再逐电机标定或任意姿态置零。已有有效校准可复用，然后 `s` 开始测试。重力补偿使用同一校准，`g` 直接请求出力，无 `ENABLE` 二次确认，但保留安全检查；`d` 直接关闭补偿；`q`/`Ctrl+C` 先关闭电机输出再退出整个测试程序，均无二次确认。`i` 查看状态。`Ctrl+C` 会立即撤力，请准备支撑。详见 [FACTR 使用说明](README_CN.md#361-factr-franka-校准手动重力补偿与无-vla-测试)。

`mode: simulation`：先 `c` 校准、`g` 开补偿，再 `s` 启动。实体主臂会缓慢对齐静止的仿真机械臂，请留出空间；倒计时后七关节按 1:1 跟随。GUI 走同一关节路径且仅控制、不记录；正常完成保留补偿。SpaceMouse 采集不受影响。

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
