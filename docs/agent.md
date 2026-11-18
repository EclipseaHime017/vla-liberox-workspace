# Agent 指南

给在本仓库工作的 AI agent：开始任务前先读本页，再按需读对应的专题文档。部署、安装和环境核对使用技能 [`skills/deploy`](../skills/deploy/SKILL.md)（Claude Code 通过 `.claude/skills/deploy` 发现）。

## 1. 仓库速览

| 位置 | 内容 | 入口 |
| --- | --- | --- |
| `core/` | 共享基础：配置组合、目录布局、模型登记与契约、方法目录、上游锁、IO 与哈希；不导入 torch | — |
| `sim/` | LIBERO-X 仿真、控制、录制、成功判定、无头 rollout | — |
| `rewards/` | PBRS、Stage、Final Reward、奖励版本；RynnValue、Robometer 接入 | `python -m rewards` |
| `train/` | 模型家族实现、训练方法、回放、训练循环、导出 | `python -m train` |
| `cluster/` | 多卡训练、队列、PBS、闭环评测、远程推理、模型传输 | `python -m cluster` |
| `studio/` | 平台后端（FastAPI） | `python -m studio` |
| `web/` | 平台前端（React + Vite） | `npm --prefix web ...` |
| `configs/` `envs/` `third_party/` `tests/` `docs/` | 配置、按角色的环境、上游锁与补丁、测试、文档 | — |

- 依赖方向：`core ← sim, rewards ← train ← cluster, studio`，同层互不导入，由 `tests/core/test_layout.py` 检查。
- 命令登记在各包 `__main__.py` 的 `COMMANDS`（`"命令": ("模块", "说明")`），实现放在 `<包>/commands/`，每个模块提供 `main()`。例外：`cluster` 的 `transfer-*` 直接指向 `cluster/transfer/` 下兼作库的模块。`core`、`sim` 只作为库，没有命令。
- 路径一律从 `core.paths`（`ROOT`、`CONFIGS`、`THIRD_PARTY`、`MODELS`、`OUTPUTS`、`WEB`）推导；YAML 中的相对路径以该 YAML 所在目录为起点。不要依赖当前工作目录，也不要用超出仓库根目录的 `parents[n]`。
- 详细说明：[架构](architecture.md)、[训练](training.md)、[训练方法](methods.md)、[模型](models.md)、[奖励](rewards.md)、[评测](evaluation.md)、[服务器](cluster.md)、[数据](data.md)。

## 2. 运行与验证

所有命令在仓库根目录执行。shell 中若 source 了 ROS，pytest 会因 ROS 插件崩溃，所以固定这样运行：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q tests          # 约 2 分钟
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q tests/core/test_layout.py   # 结构规则
cd web && conda run -n vla-liberox npx vitest run && conda run -n vla-liberox npx tsc --noEmit -p tsconfig.app.json
```

- 改了 Python：跑完整 `tests`；只动一个包时可先跑 `tests/<包>`，提交前仍跑完整套件。
- 改了 `web/`：跑 vitest 和类型检查。
- 改了被本地实验脚本引用的接口：如果存在 `experiments/`（不提交，只在部分机器上有），也运行 `pytest -q experiments`。
- 测试只写临时目录，不读写真实的 `dataset-root/`、`models/`、`policy-registry/`。
- 不要为了验证而启动 GPU 训练、真实评测或安装依赖，除非用户要求。启动平台检查页面后要停止进程。

## 3. 常见改动的落点

| 任务 | 修改位置 | 参考 |
| --- | --- | --- |
| 新增基础权重 | `configs/models/<家族>.yaml` 的 `variants`；权重放 `models/` | [模型](models.md) §3 |
| 新增模型家族 | `core/families/`、`train/models/`、`studio/policies/`、`configs/models/`、`third_party/lock.yaml`、`envs/roles.yaml` | [模型](models.md) §4 |
| 新增训练方法（IDQL、VF 等） | `core/methods.py`、`train/methods/`、`configs/methods/`、导出与推理链路 | [训练方法](methods.md) |
| 修改奖励公式 | `rewards/pbrs.py`、`rewards/stage/`、`rewards/fusion.py`、`configs/rewards/reward.yaml` | [奖励](rewards.md) |
| 新增任务场景 | `configs/tasks/`、`tasks/` | [任务](tasks.md) |
| 新增接口与页面 | `studio/api/`（在 `studio/main.py` 注册）、`studio/services/`；`web/src/pages/`、`web/src/features/` | [平台](studio.md) |
| 新增命令 | `<包>/commands/<名>.py` + `<包>/__main__.py` 的 `COMMANDS` | — |
| 新增上游或改版本 | `third_party/lock.yaml`、`third_party/patches/`、`envs/roles.yaml` | [third_party](../third_party/README.md) |
| 新增依赖或设备角色 | `envs/layers/*.txt`、`envs/roles.yaml`；`tests/core/test_environment_roles.py` | [安装](install.md) |
| 发布新版本 | `VERSION`、`pyproject.toml`、`studio/main.py`、`web/package.json`、`web/package-lock.json`、`web/src/components/layout/Sidebar.tsx`、README 两份、`CHANGELOG.md` 顶部标题；`test_layout.py` 检查它们一致 | — |

## 4. 必须保持的约定

**结构**

- `core` 不导入 torch；GPU 实现只在 `train/models/`、`rewards/rynnvalue/`、`rewards/robometer/` 中按需导入。
- 以下模块只能依赖标准库，因为它们在依赖装好之前运行：
  - `envs/setup.py` 与 `skills/deploy/scripts/preflight.py`（YAML 除外，二者都兼容 PyYAML 和 ruamel.yaml）；
  - `core/paths.py`，以及 `core/pi05_assets.py` 和它导入的 `core/io.py`、`core/model_storage.py`（`python -m train setup-pi05` 在新建的 pi05 环境中先导入它们）；
  - `studio/devices/factr_usb_rule.py`（以 `python3 -I` 提权运行）。
- 不在平台和前端写死方法名：读 `core.methods.METHODS` 的标志（`requires_rewards`、`critic`、`config_section`、`parameters`），前端读 `/api/methods`。
- 模型家族的差异通过适配器声明表达，不在业务代码里按家族名分支。

**数据与可复现性**

- `dataset-root/`、`models/`、`policy-registry/`、`evaluator/` 的目录名不能改，数据清单和数据库记录了其中的路径。
- 记录不可变：续跑、接管、重新评价都写新记录或新版本，不改写历史文件。
- 奖励缓存按实现文件的名称和内容建立指纹。移动或改动 `rewards/manifest.py`、`pbrs.py`、`stage/`、`fusion.py` 会让派生奖励重新物化（只用 CPU）；RynnValue 原始输出的复用由 `rewards/rynnvalue/outputs.py` 的推理契约决定。改动这些文件时在 CHANGELOG 中说明。
- 第三方源码不直接修改：补丁放 `third_party/patches/`，哈希写进 `third_party/lock.yaml`；读锁定信息用 `core.upstream.upstream_lock`。

**文档与提交**

- 文档以中文为主，`README.md` 是精简英文版；用户可见的变化写进 `CHANGELOG.md`。
- 仓库公开：不提交主机地址、账号、密码、个人路径和数据。本机部署配置放 `configs/cluster/local/`，本地实验放 `experiments/`，两者都不提交。
- 按主题提交；只在用户要求时提交和推送，不强制推送；不加 Co-Authored-By 等署名行。
- 提交信息用英文标题 `type(scope): ...`。普通提交一行即可，改动较多时在正文加几条 `- ` 要点。
- 发布版本时，标题写 `feat: release vX.Y.Z with <概括>`，正文逐条列出这一版的全部功能性改动（每条一行、动词开头），并同时更新版本相关文件。

## 5. 安全边界

- 安装、卸载、`git pull`／推送、`sudo`、远程机器上的任何操作，先征得用户同意；流程见 [`skills/deploy`](../skills/deploy/SKILL.md)。
- 平台、训练和评测共用 GPU 锁与队列。重启平台或改环境前，先确认没有正在运行的任务。
- 凭据只通过已有的 SSH 配置或 askpass 使用，不打印、不写入日志或提交。
- 不清理 `third_party/` 检出和数据目录；发现与锁定版本不一致时报告给用户。
