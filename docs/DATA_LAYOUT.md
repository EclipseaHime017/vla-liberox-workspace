# 数据与奖励格式

main 导出的目录或 ZIP 包含：

```text
export.json                 # 成员、split、相对路径、文件哈希
dataset.json                # 冻结数据集身份及配置
runs/<task>/<run_id>/
  run.json
  episodes/episode_000/
    trajectory.npz
    trajectory_observations.npz
    stage_annotation.json
    rynnvalue_evaluation.json / .npz
    trajectory_reward.<source>.json
    trajectory_reward.<hash>.npz
    robometer_evaluation.json / .npz
```

所选奖励需要完整评价文件；Robometer 仅用于诊断，BC 不读取奖励。

轨迹包含 N 个动作和 N+1 个观测，保存 time_seconds、EEF、gripper、raw_action、env_action、done、action_source。action_codec 区分 vla_adapter_v1 与 libero_env_v1。训练使用实际执行动作，再进行目标模型的归一化。

显式导入保留成员、split 和成功阈值。data.include_post_success 控制成功后动作是否训练，原始记录和评价保持完整。只核对训练必需文件，不读取视频或未选评价。

支持逐轨迹 reward snapshot schema 1、RynnValue sidecar schema 5/6 和现有 Stage 标注。每条轨迹保留奖励配方，调整 gamma/cumulative 只重算派生奖励。

每次训练保存 inputs/selection.json、prepared manifest、rewards/、reward_manifest.json 和 export_provenance.json。checkpoint 固定数据、奖励和模型身份。
