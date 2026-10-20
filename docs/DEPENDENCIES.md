# 第三方依赖与目录迁移

所有命令从工作区根目录执行。完整安装流程见 [README_CN](../README_CN.md#2-环境配置)。

## 目录职责

| 目录 | 内容 |
| --- | --- |
| `third_party/` | 下载的上游源码、源码内自带的资源及 FACTR 隔离环境 |
| `models/` | 策略基础模型权重 |
| `evaluator/` | 独立标注工具的模型权重 |
| `policy-registry/` | 平台训练后导出的模型 |
| `docs/` | 随仓库发布的使用指南和接口说明 |
| `logs/` | 本地调研、开发记录、排错和验收报告，不提交 |

`third_party` 是第三方源码目录的常见命名。本平台的适配、接口和训练实现仍放在原来的项目目录，不混入上游仓库。移动源码不改变模型名称、模型文件身份或训练算法。

## 上游目录与配置

| 上游 | 目录 | 配置入口 |
| --- | --- | --- |
| LIBERO-X | `third_party/LIBERO-X/` | `configs/config.yaml`、训练项目的 `configs/runtime.yaml` / `configs/inference.yaml` |
| VLA-Adapter | `third_party/VLA-Adapter/` | 同上 |
| OpenPI | `third_party/OpenPI/` | `scripts/setup_pi05.py` 默认安装位置；运行时通过 `pi05` 环境导入 |
| RynnValue | `third_party/RynnValue/` | 训练项目的 `configs/runtime.yaml` |
| Robometer | `third_party/Robometer/` | `vla-adapter-robometer/configs/robometer_evaluation.yaml` |
| FACTR | `third_party/FACTR_Teleop/`、`third_party/factr-runtime/` | `configs/factr_test_config.yaml` |

未使用的上游不必安装。Git 只保留 `third_party/.gitkeep`，源码、运行环境和权重不随平台提交。新增外部仓库也放在 `third_party/<仓库名>/`，并在对应模块配置其路径，不在业务页面硬编码路径。

## 环境一致性检查

VLA-Adapter 固定为 commit `23fa0c9c159e2aa04341cdd3e924f44061311060` 和 TensorFlow `2.15.0`，并应用 README 中的完整加载补丁。`vla-adapter-rynn-iql/configs/dependency-lock.yaml` 同时记录源码、补丁路径及 SHA256。

在 `vla-liberox` 环境安装 pytest 后，执行：

```bash
env CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=vla-adapter-rynn-iql/src \
  python -B -m pytest vla-adapter-rynn-iql/tests/test_upstream_environment.py \
  vla-adapter-rynn-iql/tests/test_vla_actions.py -q
```

真实像素测试先校验环境，再比较训练和上游推理的 JPEG、缩放及裁剪结果，不加载模型权重。除字节码和 egg-info 等生成物外，源码不得包含未记录的修改；缺少上游源码时跳过真实像素测试。检查支持 tensorflow、tensorflow-cpu 等发行包，缺失、版本冲突或与锁定版本不符时会明确报错。main 与 server 各自在本机执行同等检查。

## 已安装设备迁移

先停止平台和训练进程，再移动已有目录；目标目录已存在时先核对内容，不合并覆盖：

```bash
mkdir -p third_party
for repo in LIBERO-X VLA-Adapter OpenPI RynnValue Robometer; do
  if [ -d "$repo" ]; then
    if [ -e "third_party/$repo" ]; then
      echo "目标已存在，请先核对：third_party/$repo"
      break
    fi
    mv -- "$repo" "third_party/$repo"
  fi
done
```

按设备已安装的环境重新登记源码路径，`--no-deps` 保留原来的运行依赖版本：

```bash
conda run -n vla-liberox python -m pip install --no-deps \
  -e ./third_party/VLA-Adapter -e ./third_party/LIBERO-X \
  -e ./third_party/LIBERO-X/packages/openpi-client

# 安装过 π₀.₅ 时执行
conda run -n pi05 python -m pip install --no-deps \
  -e ./third_party/OpenPI -e ./third_party/OpenPI/packages/openpi-client

# 安装过 Robometer 时执行
conda run -n robometer-reward python -m pip install --no-deps \
  -e ./third_party/Robometer
```

RynnValue 通过 YAML 中的源码路径加载，不需要重新安装。FACTR 已在 `third_party/`，无需移动。

LIBERO 在 `~/.libero/config.yaml` 中另外保存资源路径；使用了 `LIBERO_CONFIG_PATH` 时检查其指向目录中的 `config.yaml`。将其中指向旧 `工作区/LIBERO-X/` 的路径改为 `工作区/third_party/LIBERO-X/`，保留其他自定义数据路径。自定义训练 YAML 中的上游路径也需要同步，历史训练结果和评价记录无需改写。

更新后的 `main` 使用新目录；旧分支或旧提交仍可能配置旧目录，切换后运行前需核对，Git 不会自动移动被忽略的第三方源码。

迁移后按 [启动指令](RUN_COMMANDS.md) 启动平台，再按 [模型测试](Model.md#5-测试新模型) 验证推理、短训练和导出模型加载。
