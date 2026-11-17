# 服务器与集群

`cluster/` 负责平台之外的训练和评测：单机多卡训练、本机训练队列、PBS 作业、闭环评测、远程推理和模型传输。它复用 `train/` 的数据准备、模型加载和训练流程，不改变 BC／IQL 算法。安装：普通服务器用 `--role train`，PBS 节点用 `--role hpc`，需要本机评测时加 `--role eval`（见 [安装](install.md)）。

## 1. 准备数据

在平台「数据集 → 已打包数据集 → 导出数据集」，完成后把显示的 `runs/` 目录复制到服务器，例如：

```bash
rsync -av --progress dataset-exports/<导出ID>/runs/ \
  <服务器>:/path/to/vla-liberox-workspace/dataset-root/projects/libero_x_vla/runs/
```

保留完整的 run 文件夹，不要只复制评价 JSON。导出时各类奖励优先使用数据集专属结果，否则使用全局结果；服务器按所选奖励读取已有结果，不重新评价。模型、数据集和训练输出放在共享持久存储或个人 Scratch，不要放在作业结束后会清空的 `$TMPDIR`，也不要让 checkpoint 占满 Home。

## 2. 单机多卡训练

配置 `configs/cluster/pipeline.yaml`：`runs_root` 指向复制后的目录；`distributed.gpu_ids` 选择 GPU；`model_profiles` 按家族给出默认的全局 batch 和可训练组件；`overrides` 选择方法（`iql`／`bc`，BC 不读取奖励）和训练参数。

```bash
conda activate vla-liberox        # 或本机已安装的 pi05 环境
python -m cluster train --dry-run # 只校验并显示计划
python -m cluster train           # 交互界面
```

启动器可以在任一装有本仓库的环境中运行；训练进程（`torchrun`）按模型家族 YAML 的 `environment` 在对应环境中启动（VLA-Adapter 为 vla-liberox，π₀.₅ 为 pi05）。

交互界面中：

- 方向键选择任务（全部已标记记录），Enter 进入配置，选中 `START` 后按 Enter 开始训练；
- `q`／`Ctrl+C` 安全停止。

`--task`、`--model`、`--method`、`--runs-root` 可以覆盖 YAML。该入口只训练并导出模型，不运行仿真或奖励模型。默认 8 卡、每卡 8 个 chunk；启动前按实际分配的 GPU 修改配置。

**训练队列**：同一台机器上排队执行多项训练，队列持久保存。

```bash
python -m cluster queue add --task <任务> --model <变体ID> --method iql --name <名称>
python -m cluster queue start     # 启动独立的后台 worker
python -m cluster queue list
python -m cluster queue stop|cancel|retry ...
```

不要在 PBS allocation 内启动后台队列。

## 3. PBS 集群

PBS 入口在分配的计算节点上复用上述训练流程；环境安装和训练都在计算节点上进行，不在登录节点上进行。

编辑 `configs/cluster/pbs.yaml`：

- `project`、`queue`、`walltime`、`gpus`：项目、公共路由队列、时限和单节点 GPU 数。
- `experiments`：每项独立选择服务器配置、模型、方法、数据、奖励来源和参数覆盖。
- `gpu_ids`：当前 allocation 内的 GPU 分组，必须互不重叠，并覆盖申请的全部卡。
- `output_root`：请求快照和结果目录（默认 `outputs/pbs/`）。

默认示例是 π₀.₅-LIBERO-X 的 RynnValue 和 Final Reward 两项实验，四卡分两组，每项两卡。修改 GPU 数量不会自动改变训练规模：batch 必须能被每项实验的卡数整除，显存占用需要实测，脚本不会自动缩小 batch。

```bash
python -m cluster pbs prepare                    # 校验配置，生成快照和 job.pbs；不申请资源
python -m cluster pbs submit                     # 正式提交，返回输出目录和作业 ID
python -m cluster pbs submit --config next.yaml --afterok 123.pbs101   # 前一作业成功后运行
qstat -f 123.pbs101
qdel 123.pbs101
```

- PBS 入口只接受单节点和调度器提供的完整 GPU UUID，不占用未分配的卡。
- 并行实验按物理 GPU 加锁，取消一项不会终止另一项。
- `--afterok` 建立作业依赖：前一作业内任一实验失败，下一作业不会开始。
- 收到停止信号，或距离时限剩余五分钟时，各训练会保存 checkpoint；强制结束只能从此前完整保存的 checkpoint 恢复。
- 请求同时记录训练代码和调度代码的指纹，作业运行或排队期间不要覆盖服务器上的代码。

输出目录 `outputs/pbs/pbs_<时间>_<ID>/`：

- `request.json`、`*.training.yaml`、`*.server.yaml`：本次配置和代码指纹；
- `submission.json`、`progress.json`、`result.json`：提交、进度和最终状态；
- `runs/<实验名>/`：各实验的日志、指标、checkpoint 和导出记录。

**NSCC 说明**

- 四卡任务使用 `normal@pbs101`；`ai@pbs102` 需要项目另有权限。不要直接指定内部执行队列。
- 超过 24 小时的 GPU 训练仍提交到 `normal@pbs101`，由调度器路由到长队列，最长 `120:00:00`，以集群政策为准。
- normal GPU 池提供单节点四卡 A100 40GB。单节点八卡需要 `select=1:ngpus=8`，不能用两台四卡节点代替。参考 [官方 GPU 指南](https://help.nscc.sg/wp-content/uploads/2024/05/ASP2A-AI-System_QuickStart-7-March-2024.pdf)。
- glibc 2.28 不能使用锁定的 Rerun PyPI wheel。创建 `pi05` 环境后，先安装同版本的 Conda 构建，再让安装器保留它（安装器会核对版本与上游锁一致）：

  ```bash
  conda install -n pi05 -c conda-forge rerun-sdk=0.23.1=py311h7435ce4_0 numpy=1.26.4 pyarrow=20.0.0
  conda run --no-capture-output -n pi05 python -m train setup-pi05 --preserve-package rerun-sdk
  ```

## 4. 评测与传输

- 闭环评测和远程推理（`python -m cluster evaluate`、`serve`、`evaluate-remote`）见 [评测](evaluation.md)。
- `transfer-*` 命令用于在限定的时间窗口内，把集群上新完成的模型下载到本地、校验、发布并排队评测；`transfer-pbs` 发布已结束 PBS 作业的结果目录，`transfer-seed` 用本地已有的导出初始化接收端。这些命令的配置含主机与账号信息，只放在 `configs/cluster/local/`，不提交。

## 5. 测试

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vla-liberox python -m pytest -q tests/cluster
```

测试使用模拟调度器，不申请 GPU，也不提交 PBS 作业。
