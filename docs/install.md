# 安装：按设备用途

每台设备只安装它的用途需要的环境、上游源码和权重。用途（角色）、安装单元和步骤写在 [`envs/roles.yaml`](../envs/roles.yaml)，上游的锁定版本写在 [`third_party/lock.yaml`](../third_party/lock.yaml)，由 `envs/setup.py` 组合执行。

## 1. 准备

以 Ubuntu、NVIDIA GPU 和 Conda 为例。需要能访问 GitHub、Hugging Face；π₀.₅ 权重转换还需要访问 Google Cloud Storage。

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential pkg-config \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev libhidapi-dev
git lfs install

git clone https://github.com/EclipseaHime017/vla-liberox-workspace.git
cd vla-liberox-workspace
```

以下命令都在仓库根目录执行。`envs/setup.py` 只依赖标准库和 YAML 解析器，用 Conda base 环境的 Python 或系统 `python3` 运行即可，不需要先装本仓库。

## 2. 选择角色

```bash
python envs/setup.py --list                               # 查看角色
python envs/setup.py --role train --family pi05 --dry-run # 只打印计划
python envs/setup.py --role train --family pi05           # 执行
```

| 角色 | 用途 | Conda 环境 | third_party | 权重 | 前端 |
| --- | --- | --- | --- | --- | --- |
| `dev` | 开发与测试 | vla-liberox | VLA-Adapter、LIBERO-X | 无 | 构建 |
| `studio` | 平台机：仿真、采集、接管、网页 | vla-liberox（`--family pi05` 另加 pi05） | VLA-Adapter、LIBERO-X（+OpenPI） | 所选家族的基模 | 构建 |
| `teleop` | 接管设备，附加到 studio | vla-liberox + FACTR 隔离运行时 | FACTR | 无 | — |
| `reward-rynnvalue` | RynnValue 轨迹评价 | rynnvalue-reward | RynnValue | RynnValue-4B | 否 |
| `reward-robometer` | Robometer 评价 | robometer-reward | Robometer | Robometer-4B-LIBERO | 否 |
| `train` | 训练服务器 | 所选家族的环境 | 所选家族的上游 | 所选家族的基模 | 否 |
| `hpc` | PBS 集群节点 | 同 train | 同 train | 同 train | 否 |
| `eval` | 无头闭环评测，附加到 train／infer | 所选家族的环境 + 仿真层 | + LIBERO-X | 待评测策略自带 | 否 |
| `infer` | 远程推理服务 | 所选家族的环境 | 所选家族的上游 | 所选家族的基模 | 否 |

`--family` 取 `vla_adapter`（默认）或 `pi05`，可以重复。角色可以组合，相同步骤只执行一次：

```bash
# 个人平台机：平台 + 接管设备 + RynnValue 评价 + π₀.₅
python envs/setup.py --role studio --role teleop --role reward-rynnvalue --family vla_adapter --family pi05
# 训练服务器：π₀.₅ 训练，并在本机做无头评测
python envs/setup.py --role train --role eval --family pi05
```

`dev` 包含 VLA-Adapter 和训练依赖，因为测试会导入两个模型家族的训练代码；它不下载任何权重。

## 3. 安装做了什么

每个单元在对应环境中按顺序执行：

- 创建环境（已存在则跳过）；
- `clone`：把上游克隆到 `third_party/<目录>/` 并检出锁定 commit。已有检出只核对 commit，不一致时停止，不会重置你的修改。
- `patch`：校验补丁哈希后应用一次（VLA-Adapter）；
- 安装 `envs/layers/*.txt` 中的依赖层；
- `pip install --no-deps -e .` 安装本仓库，并用 `python -I` 核对 `core/sim/rewards/train/cluster/studio` 六个包都从本仓库导入；
- vla-liberox 最后安装 CUDA 12.8 的 PyTorch（支持 RTX 50 系列，需要匹配的驱动），并固定 `numpy==1.26.4`；
- 下载或转换权重：π₀.₅ 由 `python -m train prepare-pi05` 转换（首次下载约 12 GiB，另生成约 7 GiB，占用较多内存）；RynnValue 和 Robometer 用 `hf download` 下载锁定的 revision；VLA-Adapter 在首次加载时下载到 `models/`。

| 依赖层 | 内容 |
| --- | --- |
| `envs/layers/sim.txt` | 平台仿真（MuJoCo、robosuite 等，按 VLA-Adapter 已装依赖固定） |
| `envs/layers/studio.txt` | 后端（FastAPI、uvicorn） |
| `envs/layers/vla.txt` | VLA-Adapter 训练与推理、TensorBoard、W&B |
| `envs/layers/eval.txt` | 无头评测需要的仿真依赖，装进模型环境 |
| `envs/layers/test.txt` | pytest |
| `envs/layers/spacemouse.txt`、`factr.txt` | 接管设备 |
| `envs/rynnvalue.txt`、`envs/robometer.txt`、`envs/robometer-constraints.txt` | 两个奖励模型环境 |

不要安装 `third_party/LIBERO-X/requirements.txt`，它会替换当前环境的依赖；也不需要 FlashAttention。

## 4. 需要人工完成的步骤

`setup.py` 不执行需要 sudo 或人工判断的步骤，只在结束时列出：

- **SpaceMouse**：复制 udev 规则后重插设备。自带规则适用于对应型号的 SpaceMouse Wireless，其他型号按实际 USB ID 调整。

  ```bash
  sudo cp envs/udev/70-3dconnexion-spacemouse-wireless.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

- **FACTR**：先为系统 Python 安装 ROS 2 Jazzy 和 Pinocchio（不要装进 vla-liberox），再由 `python -m studio setup-factr` 建立 `third_party/factr-runtime/`。USB 延迟规则见 [平台使用](studio.md#接管设备)。
- **π₀.₅-LIBERO-X 基模**：先在 Hugging Face 获得 [Adam-YAN/pi05-liberox-base](https://huggingface.co/Adam-YAN/pi05-liberox-base) 的访问权限，再执行：

  ```bash
  conda run --no-capture-output -n vla-liberox hf auth login
  conda run --no-capture-output -n pi05 python -m train prepare-pi05 --base-model pi05-liberox-base
  ```

- **W&B**：需要时执行 `conda run --no-capture-output -n vla-liberox wandb login`，再在训练配置中开启。
- **LIBERO 首次运行**：询问是否自定义数据集路径时输入 `N`。

## 5. 检查安装

```bash
conda run -n rynnvalue-reward python -m rewards rynnvalue-verify   # RynnValue 检出与环境
conda run -n robometer-reward python -m rewards robometer-verify   # Robometer 版本矩阵与 CUDA
conda run -n vla-liberox python -m studio setup-factr --check      # FACTR 源码与运行时，不访问设备
```

开发机运行测试（shell 中 source 过 ROS 时，先去掉它的 `PYTHONPATH`）：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q tests
conda run -n vla-liberox npm --prefix web test
```

VLA-Adapter 的源码、补丁和 TensorFlow 版本由 `tests/core/test_upstream_environment.py` 核对：除字节码和 egg-info 外，检出不得包含补丁以外的修改。

## 6. 从 0.8 目录升级已安装的设备

`third_party/`、`models/`、`dataset-root/`、`policy-registry/` 的位置不变，数据无需迁移。旧的训练包和 Robometer 接入包不再存在，需要在每个用到它们的环境中换成本仓库：

```bash
conda run -n vla-liberox python -m pip uninstall -y vla-rynn-iql
conda run -n vla-liberox python -m pip install --no-deps -e .
conda run -n pi05 python -m pip uninstall -y vla-rynn-iql
conda run -n pi05 python -m pip install --no-deps -e .
conda run -n robometer-reward python -m pip uninstall -y vla-adapter-robometer
conda run -n robometer-reward python -m pip install --no-deps -e .
```

然后运行对应角色的 `--dry-run` 核对计划，并按 [CHANGELOG](../CHANGELOG.md) 的路径对照更新自定义 YAML 和脚本。
