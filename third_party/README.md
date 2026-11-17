# third_party

本仓库依赖的上游源码。Git 只提交三样东西：

- `lock.yaml`：所有上游的锁定信息（仓库地址、commit、本地目录名，以及补丁、权重和运行时校验的包版本）；
- `patches/`：对上游源码的本地补丁，哈希记录在 `lock.yaml`；
- 本说明。

上游检出、FACTR 隔离运行环境和权重都不提交。按设备用途安装时，`python envs/setup.py --role <角色>` 只克隆该角色需要的上游；已有检出只核对 commit，不会被重置。

| 上游 | 目录 | 用途 | 使用者 |
| --- | --- | --- | --- |
| LIBERO-X | `LIBERO-X/` | 仿真场景、BDDL 任务、初始状态 | `sim/`、平台、无头评测 |
| VLA-Adapter | `VLA-Adapter/` | VLA-Adapter 策略家族（打补丁） | `train/models/vla_adapter.py`、平台推理 |
| OpenPI | `OpenPI/` | π₀.₅ 策略家族，按其 `uv.lock` 安装 | `train/models/pi05.py`、`python -m train setup-pi05` |
| RynnValue | `RynnValue/` | 时序价值奖励模型，从源码加载 | `rewards/rynnvalue/` |
| Robometer | `Robometer/` | 进度与成功概率奖励模型 | `rewards/robometer/` |
| FACTR | `FACTR_Teleop/`、`factr-runtime/` | 力反馈接管设备，在独立的系统 Python 环境运行 | `studio/devices/factr*.py` |

代码通过 `core.upstream.upstream_lock(<名称>)` 读取锁定信息，`checkout_path(<名称>)` 给出默认检出目录。

## 更新或新增上游

1. 修改 `lock.yaml` 中对应条目的 `commit`（有补丁时同时更新 `patches/` 和 `sha256`）。
2. 在 `envs/roles.yaml` 中把 `clone: <名称>` 加到需要它的安装单元。
3. 运行 `python envs/setup.py --role <角色> --dry-run` 检查计划，再运行 `pytest tests/core/test_environment_roles.py`。

移植第三方训练代码（例如 IDQL）时，只把核心逻辑移入 `train/`，并在文件头注明来源仓库和 commit；不为此新增检出目录。
