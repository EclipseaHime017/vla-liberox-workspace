# 服务器依赖

所有路径以仓库根目录为起点。训练环境安装见 [README_CN](../README_CN.md#1-环境配置)，常用操作见 [RUN_COMMANDS](RUN_COMMANDS.md)。

## 训练环境和上游源码

| 模型 | Conda 环境 | 上游源码 | 固定版本 |
| --- | --- | --- | --- |
| VLA-Adapter | `vla-liberox`，Python 3.10 | `third_party/VLA-Adapter/` | commit `23fa0c9c159e2aa04341cdd3e924f44061311060`，Torch 2.7.0，Transformers 4.40.1 |
| π₀.₅ | `pi05`，Python 3.11 | `third_party/OpenPI/` | commit `215abfb217dbac7d5f1273282331b9b1866c0479`，Torch 2.7.1，Transformers 4.53.2 |

模型家族 YAML 管理训练环境和接口，变体管理权重及归一化配置。上游源码放在 third_party，不混入服务器训练实现；两个模型环境独立安装。

server 从导出的数据读取轨迹和奖励，不需要安装仿真、控制器或网页依赖。复用已有 RynnValue、Stage、Final Reward 及 Robometer 结果也不需要评价模型环境。

## 本地目录

| 目录 | 用途 |
| --- | --- |
| `models/` | 策略基础权重及下载缓存 |
| `policy-registry/` | 训练后导出的模型 |
| `dataset-root/imports/` | 上传的数据集目录或 ZIP |
| `vla-adapter-rynn-iql/outputs/` | 训练记录、checkpoint、replay 缓存和部署副本 |
| `third_party/` | 本机安装的上游源码 |
| `docs/` | 随仓库发布的使用说明 |
| `logs/` | 本地开发和验收记录，不提交 |

权重、数据、缓存、训练产物和第三方源码不随 Git 提交。需要发布的使用与接口说明放在 docs；调研、架构草稿及排错记录放入 logs。

## 可选离线评价

仅需重新生成模型评价时安装。RynnValue 的源码路径由 `vla-adapter-rynn-iql/configs/runtime.yaml` 管理，模型及源码版本见同目录的 `dependency-lock.yaml`。Robometer 的固定版本见 `vla-adapter-robometer/configs/dependency-lock.yaml`。

Robometer 使用独立环境，不将其 Torch/Transformers 依赖装入训练环境：

```bash
conda create -n robometer-reward python=3.10 -y
conda activate robometer-reward
git clone https://github.com/robometer/robometer.git third_party/Robometer
git -C third_party/Robometer checkout 352d160389daa964788de1ec933d1925f3a6de4f
python -m pip install torch==2.8.0 torchvision==0.23.0 torchao==0.13.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install xformers==0.0.32.post2 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c vla-adapter-robometer/constraints-robometer.txt \
  -e './third_party/Robometer[robometer]'
python -m pip install -e ./vla-adapter-robometer
python vla-adapter-robometer/scripts/verify_environment.py
```

配置文件为 `vla-adapter-robometer/configs/robometer_evaluation.yaml`。Torch、torchao、xFormers 和 Transformers 必须按锁定版本成套安装，不能单独降级 torchao。
