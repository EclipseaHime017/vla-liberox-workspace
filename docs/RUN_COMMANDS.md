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

# 回归
conda run --no-capture-output -n vla-liberox python -m pytest vla-adapter-rynn-iql/tests -q
```
