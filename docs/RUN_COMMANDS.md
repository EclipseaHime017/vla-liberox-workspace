# 服务器指令

在工作区根目录执行，训练配置位于 vla-adapter-rynn-iql/configs/server_pipeline.yaml。

```bash
# 预检查
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_server.py --dry-run

# YAML 配置训练
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_server.py

# π₀.₅ BC
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_server.py --model pi05-libero-base --method bc

# 指定导出数据包
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_server.py --runs-root /mnt/data/task-export.zip --dry-run

# 提交训练到队列（可修改 YAML 后继续提交，已入队配置固定）
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_queue.py add --config vla-adapter-rynn-iql/configs/server_pipeline.yaml

# 后台串行训练；断开终端后继续运行
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_queue.py start

# 查看队列
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_queue.py list

# 当前训练完成后停止队列，保留待执行任务
conda run --no-capture-output -n vla-liberox python vla-adapter-rynn-iql/scripts/train_queue.py stop

# 回归
conda run --no-capture-output -n vla-liberox python -m pytest vla-adapter-rynn-iql/tests -q
```
