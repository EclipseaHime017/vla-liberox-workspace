---
name: deploy
description: 在一台设备上部署、升级或核对 vla-liberox-workspace：按用途（dev、studio、teleop、reward-rynnvalue、reward-robometer、train、hpc、eval、infer）安装 Conda 环境、third_party 上游和权重，从 0.8.0 布局升级，并在前后用只读的 preflight 检查。用于"部署／安装／配环境／升级到新版本／在服务器或 PC 上跑起来／检查环境是否正确"等请求。
---

# 部署与环境核对

本仓库按设备用途安装：角色和步骤在 `envs/roles.yaml`，上游版本在 `third_party/lock.yaml`，执行器是 `envs/setup.py`。本技能给出完整流程和一个只读检查脚本 `scripts/preflight.py`。背景见 `docs/install.md`、`docs/agent.md`。

## 规则

- **先查后改**：任何改动前先运行 preflight 和 `--dry-run`，把计划给用户看。
- **需要用户确认的操作**：执行安装或卸载、`git pull`／切换分支、任何 `sudo`、在远程机器（评测 PC、训练服务器、集群）上的任何操作、下载大权重（π₀.₅ 约 12 GiB）。
- **不做的事**：不重置、不清理 `third_party/` 中的检出（commit 不符时停下来问）；不删除 `dataset-root/`、`models/`、`policy-registry/`、`outputs/` 和 0.8.0 的遗留目录；不安装 `third_party/LIBERO-X/requirements.txt`；不强制推送。
- **凭据**：主机地址、账号、密码只放在 `configs/cluster/local/` 等不提交的位置，通过已有的 SSH 配置或 askpass 使用；不打印、不写进日志、提交或回复。
- **不打断任务**：升级前确认没有平台、训练或评测在运行（平台「训练」页的队列、`python -m cluster queue list`、`nvidia-smi`）。

## 流程

**1. 确定目标**：哪台设备（本机或远程）、哪些角色、哪些模型家族。

| 用户想做的 | 角色 |
| --- | --- |
| 开发、跑测试 | `dev` |
| 用网页仿真、采集、训练、测试 | `studio`（需要 π₀.₅ 时 `--family vla_adapter --family pi05`） |
| 人工接管 | 再加 `teleop` |
| 在本机做 RynnValue／Robometer 评价 | 再加 `reward-rynnvalue`／`reward-robometer` |
| 服务器训练 | `train`（PBS 节点用 `hpc`），`--family` 选模型 |
| 服务器上闭环评测 | 再加 `eval` |
| 只做远程推理服务 | `infer` |

**2. 只读检查**

```bash
python skills/deploy/scripts/preflight.py --role <角色> [--family <家族>]          # 文本
python skills/deploy/scripts/preflight.py --role <角色> --json                      # 给程序读
```

用系统 `python3` 或 Conda base 的 Python 运行即可。每项为 OK／WARN／FAIL，并附建议命令；退出码 1 表示有 FAIL。

| 区域 | 检查内容 |
| --- | --- |
| repository | 版本、分支与提交、未提交修改、与上次 fetch 的远端相比是否落后或分叉、0.8.0 遗留目录 |
| environments | 角色需要的 Conda 环境是否存在；六个包是否从本检出导入（`python -I`）；是否残留 0.8.0 的可编辑安装 |
| third_party | 检出是否存在、commit 是否等于锁定值；VLA-Adapter 补丁是否已应用；FACTR 官方文件哈希；其他检出是否干净（忽略字节码和 egg-info） |
| web | 需要前端的角色是否已安装 `web/node_modules` |

**3. 同步代码**（工作区有未提交修改或分叉时先问用户）

```bash
git fetch origin
git status -sb
git pull --ff-only          # 确认后
```

**4. 计划与安装**

```bash
python envs/setup.py --role <角色> [--family <家族>] --dry-run    # 把摘要给用户确认
python envs/setup.py --role <角色> [--family <家族>]              # 耗时长，后台运行并保留日志
```

结束时列出的"Manual steps left"（udev 规则、ROS 2 与 Pinocchio、Hugging Face 授权等）转告用户，不要自行执行 sudo。

**5. 从 0.8.0 升级**（preflight 提示残留可编辑安装或遗留目录时）

```bash
conda run -n vla-liberox python -m pip uninstall -y vla-rynn-iql
conda run -n vla-liberox python -m pip install --no-deps -e .
# pi05 同上；robometer-reward 卸载的是 vla-adapter-robometer
```

用户自定义的 YAML 和脚本按 `CHANGELOG.md` 的 0.8.0 → 0.8.5 路径对照表更新。数据目录不需要迁移。

**6. 验证**

| 角色 | 命令 |
| --- | --- |
| 全部 | 再跑一次 preflight，FAIL 必须为 0 |
| dev | `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q tests`；`conda run -n vla-liberox npm --prefix web test` |
| studio | `conda run -n vla-liberox python -m studio`，打开 http://127.0.0.1:8000；`curl -s 127.0.0.1:8000/api/methods` 返回 iql、bc。用完停止进程 |
| teleop | `conda run -n vla-liberox python -m studio setup-factr --check`（不访问设备） |
| reward-rynnvalue | `conda run -n rynnvalue-reward python -m rewards rynnvalue-verify` |
| reward-robometer | `conda run -n robometer-reward python -m rewards robometer-verify` |
| train／hpc | `python -m cluster train --dry-run` 或 `python -m train pipeline --dry-run` |
| eval | 在分配到的 GPU 节点上 `python -m cluster evaluate --check-render` |
| infer | `python -m cluster serve --help`；正式启动前与用户确认 GPU 和端口 |

**7. 报告**：装了什么、跳过了什么、剩余的人工步骤、preflight 的最终结果。实际执行和只做了 dry-run 的部分要分开说明。

## 远程设备

在远程机器的仓库根目录执行同样的步骤（通过用户已有的 SSH 别名；需要密码时用 askpass，不回显）。临时脚本放在对方的 `/tmp` 并在结束后删除。PBS 集群在计算节点上安装和运行（交互式作业），不在登录节点；NSCC 的 `pi05` 环境需要先装 Conda 版 rerun-sdk，见 `docs/cluster.md`。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| pytest 报 `No module named 'lark'` | shell 里 source 了 ROS：加 `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` |
| `third_party/<X> is at ..., the lock pins ...` | 检出被改动或切换过；问用户后再检出锁定 commit，不要 reset |
| packages imported from elsewhere | 有同名包遮蔽（`core`、`sim`、`train` 是通用名）：卸载遮蔽的分发，再 `pip install --no-deps -e .` |
| 安装后 torch 版本不对 | vla-liberox 的 `torch-cu128` 单元必须最后执行；重跑该角色即可 |
| `Incompatible Robometer runtime` | 按锁定矩阵重建 robometer-reward，不要单独升降 torchao |
| Hugging Face 401／403 | 需要授权的模型（pi05-liberox-base）先 `hf auth login` |
| `--check-render` 报软件渲染器 | 容器或节点没有暴露 NVIDIA EGL，见 `docs/evaluation.md` 的无头渲染一节 |
| LIBERO 首次运行询问数据集路径 | 输入 `N` |
