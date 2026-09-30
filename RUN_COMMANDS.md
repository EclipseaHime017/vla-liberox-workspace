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

默认打开终端交互界面，选择任务、配置训练参数后启动。
YAML 的 `runs_root` 指向复制来的原始数据及全局评价；服务器不运行评价模型。
方法配置使用 `training/iql.yaml` 或 `training/bc.yaml`。
SSH/nohup/Slurm 非交互运行时加 `--yes --task <完整任务ID>`。

启动前只检查配置和数据，不执行训练：

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml \
  --dry-run
```

多任务目录执行 `--dry-run` 时需加 `--task <完整任务ID>`。
GPU 由 `distributed.gpu_ids` 指定；若已有 `CUDA_VISIBLE_DEVICES`，
索引对应调度器分配的可见卡，否则对应物理卡。
