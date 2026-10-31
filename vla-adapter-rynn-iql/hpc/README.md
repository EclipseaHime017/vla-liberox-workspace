# HPC 训练提交

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

## 直接运行服务器训练

```bash
python vla-adapter-rynn-iql/scripts/train_server.py \
  --config vla-adapter-rynn-iql/configs/server_pipeline.yaml
```

此入口不申请 PBS 资源，普通服务器的运行方式保持不变。

## 配置与提交 PBS

编辑 [configs/pbs_pipeline.yaml](configs/pbs_pipeline.yaml)：

- `project`、`queue`、`walltime`、`gpus`：项目、公共路由队列、时限和单节点 GPU 数量。
- `experiments`：每项独立选择服务器配置、模型、训练方法、数据集、奖励来源和参数覆盖。
- `gpu_ids`：当前 allocation 内的 GPU 分组，必须互不重叠且覆盖申请的所有卡。
- `output_root`：请求快照和结果目录；所有相对路径均相对于这份 YAML。

默认示例为四卡分成两组，每项两卡、全局 batch 64、10000 steps、梯度累积 1，每卡 batch 32。修改 GPU 数量不会自动改变训练规模，batch 必须能整除每项实验的卡数。

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

## 查看结果

默认输出在 `vla-adapter-rynn-iql/outputs/pbs/pbs_<时间>_<ID>/`：

- `request.json`、`*.training.yaml`、`*.server.yaml`：本次配置和代码指纹。
- `submission.json`、`progress.json`、`result.json`：提交、进度和最终状态。
- `runs/<实验名>/`：独立的训练日志、指标、checkpoint 和模型导出记录。

任一实验失败，依赖它的 `afterok` 作业不会启动。接近时限或收到停止信号时会请求保存 checkpoint；强制结束或节点故障只能恢复此前完整保存的 checkpoint。模型、数据和输出不要放在任务结束后会删除的临时目录。

请求同时校验训练代码和 HPC 调度代码。运行和排队期间不要覆盖服务器上的这些代码；本地整理目录不影响已经提交的远端任务。

## 测试

```bash
conda run --no-capture-output -n vla-liberox python -m pytest \
  vla-adapter-rynn-iql/hpc/tests -q
```

测试使用模拟调度器，不申请 GPU 或提交 PBS 作业。
