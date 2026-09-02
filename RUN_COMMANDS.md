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

## 服务器端：启动单机多卡训练

仅在 `server` 分支提供。配置文件：
`vla-adapter-rynn-iql/configs/server_pipeline.yaml`

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml
```

脚本显示数据选择、缓存和训练计划后会提示：

```text
Start this pipeline? [y/N]
```

- 输入 `y` 或 `yes`：确认并开始执行；
- 直接按 Enter，或输入其他内容：取消执行。

启动前只检查配置和数据，不执行训练：

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml \
  --dry-run
```

使用哪些物理 GPU 由 YAML 的 `distributed.gpu_ids` 明确指定；脚本会为
`torchrun` 设置 `CUDA_VISIBLE_DEVICES`，无需在启动命令前重复设置。
