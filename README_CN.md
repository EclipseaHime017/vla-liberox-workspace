# LIBERO-X 服务器训练

server 跟随 main 的发布版本号，当前两者均为 **v0.7.2**。两个分支独立维护：server 只负责命令行训练，不合并 main 的运行流程。

操作顺序：安装环境 → 上传数据集 → 配置训练 → 运行脚本 → 下载训练模型。除上传和下载外，以下命令均在服务器的仓库根目录执行。

## 1. 环境配置

### 下载 server 分支

```bash
mkdir -p ~/eclipseaws
cd ~/eclipseaws
git clone --branch server https://github.com/EclipseaHime017/vla-liberox-workspace.git
cd vla-liberox-workspace
```

已有服务器仓库时，在工作区干净的情况下更新：

```bash
git fetch origin
git switch server
git pull --ff-only origin server
```

服务器需要 NVIDIA GPU、Conda 和 Git，以及支持 CUDA 12.8 的驱动。按使用的模型安装对应环境。

### VLA-Adapter

```bash
conda create -n vla-liberox python=3.10 pip -y
conda activate vla-liberox
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git third_party/VLA-Adapter
git -C third_party/VLA-Adapter checkout 23fa0c9c159e2aa04341cdd3e924f44061311060
python -m pip install -e ./third_party/VLA-Adapter
python -m pip install -r vla-adapter-rynn-iql/requirements-train.txt
python -m pip install -e ./vla-adapter-rynn-iql
python -m pip install --upgrade torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 setuptools==69.5.1
git -C third_party/VLA-Adapter apply ../../vla-adapter-rynn-iql/vla_adapter_hf_local_autoclass.patch
```

默认基模首次加载时下载到 `models/VLA-Adapter-LIBERO-Object-Pro/`。也可以提前复制同一基础模型的完整权重目录到该位置。

### π₀.₅

```bash
conda create -n pi05 python=3.11 pip -y
conda run -n pi05 python -m pip install uv
conda run --no-capture-output -n pi05 python vla-adapter-rynn-iql/scripts/setup_pi05.py
conda run --no-capture-output -n pi05 python vla-adapter-rynn-iql/scripts/prepare_pi05.py --base-model pi05-libero-base
conda activate pi05
```

安装脚本将固定版本的 OpenPI 放在 `third_party/OpenPI/`；权重默认以 BF16 转换到 `models/pi05_libero_torch/`。已有转换权重时直接复制完整目录，不必再次转换。其他已注册变体使用对应的 `--base-model` ID。

两个环境互不混装。训练入口可在已安装的训练环境中运行，实际训练进程按模型配置选择环境。复用导出的评价和奖励，不需要安装 RynnValue 或 Robometer 环境。

## 2. 上传和导入数据集

准备完整的数据集导出 ZIP 或目录，保留其中的清单、轨迹和评价文件。服务器示例上传位置为 `dataset-root/imports/`。

在本地终端修改服务器地址和仓库路径后执行：

```bash
TRAIN_HOST=user@server
TRAIN_REPO=/home/user/eclipseaws/vla-liberox-workspace
ssh "$TRAIN_HOST" "mkdir -p '$TRAIN_REPO/dataset-root/imports'"
scp ./task-export.zip "$TRAIN_HOST:$TRAIN_REPO/dataset-root/imports/"
```

上传目录时，将最后一行替换为：

```bash
scp -r ./task-export "$TRAIN_HOST:$TRAIN_REPO/dataset-root/imports/"
```

导出的数据集保留原成员和 train/validation 划分。也支持完整的 runs 目录，但会按其中已有的标记选择轨迹。ZIP 在运行时临时解压训练需要的文件，结束后清理，不需要手动解压。

## 3. 配置训练

编辑 [server_pipeline.yaml](vla-adapter-rynn-iql/configs/server_pipeline.yaml)。下面是 VLA-Adapter、IQL、8 卡的配置示例：

```yaml
schema_version: 4
training_config: ./training/iql.yaml
runs_root: ../../dataset-root/imports/task-export.zip
task_id: null
output_root: ../outputs/server-training
cache_root: ../outputs/server-replay-cache
reward_source: final

distributed:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  backend: nccl
  zero_stage: 1
  timeout_seconds: 1800
  data_workers_per_rank: 2
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true

overrides:
  model:
    family: vla_adapter
    base_id: base
    backbone: frozen
  training:
    method: iql
    train_steps: 10000
    micro_batch_size: 8
    gradient_accumulation_steps: 4
    checkpoint_interval: 512
  reward: {}
```

所有 YAML 相对路径按对应配置文件所在目录解析。目录导入时把 `runs_root` 改为目录路径。只有一个任务可保留 `task_id: null`；多个任务时填写完整 task ID，或启动时使用 `--task`。

### GPU 和训练规模

- `gpu_ids` 指定参与训练的 GPU，支持单机 1–8 卡。已有 `CUDA_VISIBLE_DEVICES` 时，编号对应其可见设备列表。
- `micro_batch_size` 是全局 batch，必须能被 GPU 数整除；每卡 batch 为全局 batch ÷ GPU 数。
- actor 每次更新的有效 batch = micro batch × gradient accumulation。示例为每卡 1、全局 8、累积 4，有效 batch 为 32。
- `checkpoint_interval` 必须能被累积步数整除。调整 GPU 数不会自动调整训练步数或学习率。

### 模型和训练方法

使用 π₀.₅ 时，将 `overrides.model` 改为：

```yaml
model:
  family: pi05
  base_id: pi05-libero-base
  backbone: frozen
```

VLA-Adapter 的 `backbone` 可选 `frozen/lora/full`；π₀.₅ 可选 `frozen/full`，其中 frozen 训练动作专家和投影层。其他模型配置见 [Model.md](docs/Model.md)。

使用 BC 时，将 `training_config` 改为 `./training/bc.yaml`，同时设置 `overrides.training.method: bc`。BC 使用数据集中的动作，不读取奖励、不训练 Q/V；筛选训练样本应在准备数据集时完成。

IQL 的 `reward_source` 可选 `final/rynnvalue/stage/sparse`。已有评价直接复用，缺少所选奖励时停止并报错，不会自动运行评价模型。p、α、κ 等配方来自导出的结果；训练只允许通过 `overrides.reward` 调整以下两项：

```yaml
reward:
  gamma: 0.99
  accumulate_primitive_steps: false
```

保留 `reward: {}` 时沿用导出的配置。相乘 Final Reward 的 cumulative 必须为 Off。

## 4. 运行训练脚本

激活已安装的训练环境后，先检查数据和配置：

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml --dry-run
```

开始训练：

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml
```

也可临时指定模型和方法：

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --model pi05-libero-base --method bc
```

入口自动完成输入校验、奖励读取、mmap 缓存和 DDP + ZeRO-1 启动。远程长训练建议在 tmux 中执行，断开 SSH 不会中断训练。

### 查看进度、停止和恢复

默认训练记录在 `vla-adapter-rynn-iql/outputs/server-training/server_<时间>_<ID>/`。终端会打印 run 路径和进度，在另一终端查看：

```bash
RUN="vla-adapter-rynn-iql/outputs/server-training/server_<时间>_<ID>"
tail -f "$RUN/training.log"
tail -n 1 "$RUN/metrics.jsonl"
```

TensorBoard 默认启用。需要 W&B 时先在模型训练环境运行 `wandb login`，再设置 `overrides.logging.wandb.enabled: true`。

Ctrl+C 会请求在安全的梯度累积边界保存 checkpoint。恢复时设置：

```yaml
overrides:
  training:
    resume_checkpoint: /absolute/path/to/checkpoints/step_00000512
```

保留其他原配置，并确保 `train_steps` 大于已完成步数。恢复需保持模型、数据、奖励、全局 batch、累积和 seed 一致；可以改变 GPU 数，但全局 batch 仍须能被整除。

## 5. 导出训练结果并下载到本地

训练完成后，`$RUN/summary.json` 的 `policy_overlay` 指向 `policy-registry/<模型ID>/policy.yaml`。推理需要下载这个文件所在的完整模型目录，不是只下载 YAML 或 checkpoint。

在服务器上生成部署副本。将 `TARGET_BASE` 改为本地设备上对应基础权重的位置：

```bash
POLICY_MANIFEST=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy_overlay"])' "$RUN/summary.json")
POLICY_ID=$(basename "$(dirname "$POLICY_MANIFEST")")
TARGET_BASE=/home/localuser/eclipseaws/vla-liberox-workspace/models/VLA-Adapter-LIBERO-Object-Pro

python vla-adapter-rynn-iql/scripts/export_policy.py \
  --policy "$POLICY_MANIFEST" \
  --output "$PWD/vla-adapter-rynn-iql/outputs/deploy/$POLICY_ID" \
  --base-checkpoint "$TARGET_BASE"
```

π₀.₅ 使用其对应基础权重目录，例如 `models/pi05_libero_torch/`。两台设备的基础权重内容必须一致。部署副本不修改原训练结果，输出目录必须尚不存在。

在本地仓库根目录下载，填写刚才得到的模型 ID：

```bash
TRAIN_HOST=user@server
TRAIN_REPO=/home/user/eclipseaws/vla-liberox-workspace
POLICY_ID=填写模型ID
mkdir -p policy-registry
scp -r "$TRAIN_HOST:$TRAIN_REPO/vla-adapter-rynn-iql/outputs/deploy/$POLICY_ID" ./policy-registry/
```

下载后保留 `policy-registry/<模型ID>/` 的目录名和全部文件。需要继续训练时，另外保留服务器上的 run 和 checkpoints；只用于本地推理则无需下载训练 checkpoint。
