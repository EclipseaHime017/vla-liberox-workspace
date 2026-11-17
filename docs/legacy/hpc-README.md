# NSCC / PBS 训练

在原有服务器训练入口上增加 PBS 提交机制。普通服务器继续使用 `scripts/train_server.py`；集群通过这里申请资源，分配计算节点后复用相同的数据准备、模型加载和训练流程，不改变 BC／IQL 算法。

```text
hpc/
├── train_pbs.py                 # 准备、提交与计算节点入口
├── pbs.py                       # 配置快照、资源分组及作业依赖
├── configs/pbs_pipeline.yaml    # 可编辑的提交配置
├── tests/                      # 调度与隔离测试
└── local/                      # 本机部署脚本和配置，不提交 Git
```

以下命令均在仓库根目录执行。先按主 README 安装所用模型的环境，将模型和导出的数据集上传至服务器共享存储。环境安装和训练都应在集群分配的计算节点执行。

## 环境和存储

普通服务器的环境、数据导入和模型导出见[主 README](../../README_CN.md)。PBS 集群先申请 CPU 作业，在分配的计算节点安装环境；不在登录节点安装或训练。

NSCC 的 glibc 2.28 不能使用锁定的 Rerun PyPI wheel。创建 `pi05` 环境后，先安装同版本的 Conda 构建，再运行安装器：

```bash
conda install -n pi05 -c conda-forge \
  rerun-sdk=0.23.1=py311h7435ce4_0 numpy=1.26.4 pyarrow=20.0.0
conda run --no-capture-output -n pi05 python \
  vla-adapter-rynn-iql/scripts/setup_pi05.py --preserve-package rerun-sdk
```

安装器会核对其版本与上游锁一致。其他模型安装步骤不变。

模型、完整数据集导出和训练输出放在共享持久存储或个人 Scratch，不放在作业结束后清空的 `$TMPDIR`。代码可以留在 Home；配置中填写模型、数据、缓存和输出的实际路径，避免训练 checkpoint 占满 Home。

## 配置与提交 PBS

编辑 [configs/pbs_pipeline.yaml](configs/pbs_pipeline.yaml)：

- `project`、`queue`、`walltime`、`gpus`：项目、公共路由队列、时限和单节点 GPU 数量。
- `experiments`：每项独立选择服务器配置、模型、训练方法、数据集、奖励来源和参数覆盖。
- `gpu_ids`：当前 allocation 内的 GPU 分组，必须互不重叠且覆盖申请的所有卡。
- `output_root`：请求快照和结果目录；所有相对路径均相对于这份 YAML。

默认示例为 π₀.₅-LIBERO-X 的 RynnValue 和 Final Reward 两项实验：四卡分成两组，每项两卡、全局 batch 64、10000 steps、梯度累积 1，每卡 batch 32，各自保存日志、checkpoint 和导出模型。也可使用单节点八卡分成 4+4。修改 GPU 数量不会自动改变训练规模，batch 必须能整除每项实验的卡数；全量微调的显存占用需实测，脚本不会自动缩小 batch。

```bash
# 检查配置并生成快照、job.pbs；不申请资源、不加载模型
python vla-adapter-rynn-iql/hpc/train_pbs.py prepare

# 正式提交；返回输出目录和完整 PBS 作业 ID
python vla-adapter-rynn-iql/hpc/train_pbs.py submit

# 另一组实验在前一作业成功后运行
python vla-adapter-rynn-iql/hpc/train_pbs.py submit \
  --config /path/to/next.yaml --afterok 123.pbs101

qstat -f 123.pbs101
qdel 123.pbs101
```

NSCC 四卡任务可使用 `normal@pbs101`；`ai@pbs102` 需要项目另有访问权限。不要直接指定内部执行队列。程序只使用 PBS 分配的完整 GPU UUID，不自动占用其他卡。

超过 24 小时的 NSCC GPU 训练仍使用 `normal@pbs101`，调度器会路由至 `glong`。例如设置 `walltime: "72:00:00"`；最长支持 `120:00:00`，实际可用时限以集群政策为准。

NSCC 的 normal GPU 池提供单节点四卡 A100 40GB。单节点八卡需要 `select=1:ngpus=8`，不能替换成两台四卡节点；先确认项目可用的八卡路由。参考[官方 GPU 指南](https://help.nscc.sg/wp-content/uploads/2024/05/ASP2A-AI-System_QuickStart-7-March-2024.pdf)。

`prepare` 只固定配置并生成作业文件，不运行模型。正式提交后才在分配的计算节点准备数据、读取已有奖励和并行训练。`--afterok` 建立 PBS 依赖，提交后用 `qstat -f` 核对；前一作业内任一实验失败，下一作业不会开始。

PBS 入口只接受单节点和调度器提供的完整 GPU UUID。并行实验使用按物理 GPU 隔离的锁，不同卡可以并行，同卡或普通独占训练会互斥；取消一项不会终止另一项。不要在 PBS allocation 内启动 `train_queue.py` 后台队列。

## 查看结果

默认输出在 `vla-adapter-rynn-iql/outputs/pbs/pbs_<时间>_<ID>/`：

- `request.json`、`*.training.yaml`、`*.server.yaml`：本次配置和代码指纹。
- `submission.json`、`progress.json`、`result.json`：提交、进度和最终状态。
- `runs/<实验名>/`：独立的训练日志、指标、checkpoint 和模型导出记录。

收到停止信号或距离配置时限剩余五分钟时，会向各训练请求安全保存 checkpoint；强制结束或节点故障只能恢复此前完整保存的 checkpoint。最终模型的导出和下载沿用主 README 中的操作，路径以各训练的 `summary.json` 为准。

请求同时校验训练代码和 HPC 调度代码。运行和排队期间不要覆盖服务器上的这些代码；本地整理目录不影响已经提交的远端任务。

## 测试

```bash
conda run --no-capture-output -n vla-liberox python -m pytest \
  vla-adapter-rynn-iql/hpc/tests -q
```

测试使用模拟调度器，不申请 GPU 或提交 PBS 作业。
