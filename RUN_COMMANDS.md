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

## 服务器端：启动终端训练

配置文件：`vla-adapter-rynn-iql/configs/terminal_pipeline.yaml`

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
