# Stage-based 奖励：调研与实验设计

第 1–5 节保留前期调研与候选实验；第 6 节记录目前已接入的人工关键帧直接奖励。它独立于 RynnValue / Robometer，不新增 VLM 阶段分类器，也不修改 IQL 更新、成功判断或采集数据。FACTR 控制器接入是另一项独立工作。

## 1. 要解决的问题与已有工作

当前实验的问题不是让图上的 reward 更平滑，而是让奖励对“真正完成了哪一步”更可靠。阶段标签将一个全局连续回归问题拆成若干局部判断，但不能预先断言 VLM 的离散分类一定更准：夹爪是否稳定接触、物体是否悬空、是否真正释放，仍可能受遮挡和分布变化影响。需要先用人工标签建立可检查的对照。

相关工作已经覆盖了这个方向的不同部分，不能把“按阶段给奖励”本身当作新方法：

这里的 stage 首先是 reward 的结构先验，不是已经新增了一个 goal-conditioned policy。只有把子目标作为策略输入，或引入高低层策略，才涉及后者的模型结构变化；第一轮实验不同时做这件事。

| 工作 | 与本项目相关的机制 | 不应混同的边界 |
| --- | --- | --- |
| [SARM](https://arxiv.org/html/2509.25358v2) / [OpenSARM](https://github.com/xdofai/opensarm) | 显式阶段分类与阶段内进度预测；使用语义子任务标注，RA-BC 再根据预测进度变化重加权示教 | SARM 仍含连续进度头，不是只判断 keyframe；其阶段内训练标签使用时间插值。本项目不能把插值直接当作真实抓取进度，也不把 RA-BC 当成 IQL |
| [STDR](https://arxiv.org/html/2606.31377v1) | VLM 离线切分阶段，学习阶段判别与阶段内进度；加入 OOD 和稳定抓取验证，输出组合奖励 | 论文主要验证在线 RL，并非当前 VLA-Adapter 离线 IQL；它直接组合阶段/进度分数，不等价于对该分数做 PBRS 差分 |
| [Reward Machines](https://github.com/RodrigoToroIcarte/reward_machines) | 以显式状态机表达任务事件、顺序与奖励；为“当前画面相同但历史不同”提供状态表达方式 | 仍需可靠的事件识别器，不会自动解决视觉泛化问题；本轮不引入新的分层策略或反事实 replay |
| [Relay Policy Learning](https://proceedings.mlr.press/v100/gupta20a.html) | 从示教重标注子目标，学习 goal-conditioned 层级策略，再用 RL 改善长时任务 | 不只是修改 reward；如果改变 actor 的子目标输入或高低层结构，就已经超出本轮奖励消融范围 |

SARM 的标签构建会排除未完成子任务序列及包含错误的轨迹；当前采集数据恰恰需要保留失败、回退及接管，不能直接照搬其筛选规则。STDR 使用 VLM 作为弱监督而非真值，也设计了抓取验证/OOD 保护；这支持加入核验机制，但不能证明我们的新初始状态上的准确率。[SARM §3.1](https://arxiv.org/html/2509.25358v2#S3.SS1)、[STDR §III](https://arxiv.org/html/2606.31377v1#S3)

## 2. 先定义可核验的 stage

建议先以黑碗抓放任务建立标注协议，而不是立刻套用到所有任务：

| 状态 | 需要看到的证据 | 常见误判 |
| --- | --- | --- |
| 接近抓取区域 | 夹爪接近目标且具备抓取方向 | 经过目标附近不等于抓取 |
| 稳定抓取并抬起 | 夹爪闭合、目标离开支撑面、连续数帧相对位置稳定 | 空夹、碰撞抬起、短暂卡住 |
| 搬运至目标区域 | 仍稳定持有目标，进入放置区域 | 移动过程中掉落，不能继续计为搬运完成 |
| 放下并释放 | 目标受到支撑，夹爪释放且不再携带目标 | 碗在目标上方悬空或尚未松手 |
| 稳定完成 | 环境成功条件满足，且通过既定稳定性检查 | 短暂经过 success 区域 |

表中是语义状态，不是直接按固定百分比切出的五段。具体 keyframe、连续确认帧数、空间阈值应在标注协议中固定，不能逐条凭训练结果倒推。阶段目标不能覆盖或替代 LIBERO 的 `done` 和现有成功确认规则。

建议未来的独立标注保存 `observation_step`、真实时间、`stage_id`、关键事件、确认/未知标记、标注来源和协议版本，并绑定源轨迹内容哈希。人工接管前的 rollout、接管后片段以及成功后的实际记录都要可见，不能再次裁掉前缀。训练 replay 是否去重仍由原数据管线决定，不由 stage 标注偷偷改变。

允许“抓取 → 掉落 → 重新接近”和“已放下 → 再次拿起”等回退。需要区分：

- **当前状态**：此刻是否仍持有物体，可随失败回退。
- **历史事件**：是否曾经抓起过物体；它不能单独证明当前仍在正确阶段。
- **未知**：遮挡/模糊时不强行补成前进，也不把模型置信度低等同于失败。

只有成功示教内部的时间顺序是不够的。暂不使用按时间线性增长的伪 stage，也不以单调滤波掩盖掉落。若事件需要历史才能判断，未来应使用明确的状态机状态或因果观察窗口；不能在 critic 看不到这些信息时，声称 reward 已满足完整的 Markov 假设。

## 3. 两种不同的候选奖励

以下公式是本项目待验证的候选设计，不是声称已复现上述论文。令 `u` 为已核验的阶段进度索引，`K` 为最终完成索引，`L` 为当前实际执行 chunk 长度。所有对照必须同时明确 reward 和 Bellman 的时间单位。

### A. Stage potential + PBRS

先构造有界势函数：

\[
\Phi(s,u)=-\frac{K-u}{K},\qquad u\in\{0,\ldots,K\}.
\]

用相同的 transition discount 构造奖励和 Bellman target：

\[
R_t=R_t^{\mathrm{sparse}}+\lambda\bigl(\Gamma_t\Phi_{t+L}-\Phi_t\bigr),
\qquad y_t=R_t+\Gamma_t m_tV(s_{t+L}).
\]

- **Macro-step**：每个 chunk 是一次决策，`Γ = γ`，稀疏部分使用同一宏步定义。
- **Primitive cumulative / Semi-MDP**：`Γ = γ^L`，稀疏部分为 `Σ(h=0..L−1) γ^h r(t+h)`。
- 两者不能交叉拼接；不能只把稀疏部分改成累加，却仍固定以 `γ` bootstrap。

这一路径延续 PBRS 的结构，只替换 potential 的来源。满足相应 Markov/扩展状态、折扣与终止条件时，折扣后的 shaping 项会望远镜式抵消；但当前视觉近似、未知状态处理、有限时域截断仍需单独验证。真终止的势函数边界通常取零；达到采集步数上限却仍可继续的 timeout，不能未经分析当作同一类终止。不能对这些条件尚未检查的工程实现直接承诺最优策略不变。

**它不保证即时 reward 单调升高。** 如果 `Φ=-0.5`、`γ=0.99`，在同一 stage 自循环也有 `0.005λ` 的 shaping；晋级时才有较大的额外差分。曲线仍可能是阶段内小幅变化、边界突变。目标是更可信的信用分配，不是把 final reward 画成一条斜线。PBRS 的理论出处见 [Ng et al., 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。

若未来引入置信度，也应先定义统一的状态势函数，再做两端差分；直接用每个 transition 不同的置信系数乘 shaping 差分，会破坏原本的抵消关系。第一轮人工标签实验先保留 `unknown`，报告覆盖率，不假设未标注区域已经正确。

### B. Stage-dependent time cost

直接让越靠后阶段的非成功动作付出更小的代价，例如：

\[
r_t=-c(u_t),\qquad c(u)\in\{1,0.75,0.5,0.25\},
\qquad r_{\mathrm{success}}=0.
\]

这里四个代价仅用于展示形式，不是五个语义状态的最终映射或默认超参数。确定 stage 数量后应单独配置对应 `c(u)`。Macro / cumulative 下分别按宏步或实际步数归约，并与 Bellman discount 配对。

这可以直接产生台阶式上升的即时 reward，但**改变了任务的时间代价**，不再只是 PBRS。后半程拖延的代价变小，可能让策略更慢地释放/完成；折扣又会影响阶段之间的取舍。应把完成时间与成功率一起评估，不能仅凭曲线形状选择它。

还需排除两个易混淆方案：

- 每次跨越里程碑只加一次奖励：仍属于事件稀疏奖励，必须防止反复跨越刷分。
- 完成里程碑后每步持续 `+1`：会奖励停留；不等价于“奖励过去的正确动作”。

## 4. 最小消融与可证伪目标

第一轮只对比奖励，不同时改 actor 结构、IQL 优化器或数据数量：

| 对照 | 奖励 | 回答的问题 |
| --- | --- | --- |
| Sparse | 当前稀疏基线 | 没有语义评价时的基线 |
| RynnValue PBRS | 当前 RynnValue potential 与既定归约 | 原有连续评价的收益 |
| Manual-stage PBRS | 人工 stage 构造势函数，采用方案 A | 用更可检查的阶段信息是否优于原势函数 |
| Stage-weighted cost | 相同人工 stage，采用方案 B | 改变阶段时间代价是否有效，是否导致变慢 |

固定数据成员、root 分组 split、前缀去重、成功阈值、chunk 边界/mask、reward reduction、有效 batch、critic/actor 更新次数、优化器/学习率/梯度裁剪、初始化与 seed。Macro 与 primitive 若也要比较，应另设一个实验轴，而非夹带在奖励对照中。各奖励量级与 `λ` 需要报告，不能用未披露的大尺度变化解释收益。

不仅查看训练 loss，还应记录：

- 相同初始状态列表与环境 seed 下的成功率，分别报告已见/未见 init state；
- 完成所需控制步数和真实仿真时间、掉落次数、阶段停留时间与回退次数；
- 阶段误判、边界误差、unknown 覆盖率和人工复核一致性；
- sparse/shaping/final reward 分布，以及 Q/V、实际 actor advantage 和权重截断比例；
- 多训练 seed 的离散程度，而不是只挑成功率最高的一次。

如果人工阶段标签也没有收益，就不能靠更复杂的 VLM 阶段分类器解释问题；先检查奖励尺度、状态信息和离线数据覆盖。如果人工阶段有效，再研究 VLM 自动标注、置信度校准和 OOD 拒识，单独量化自动标注相对人工标注损失多少性能。

## 5. 本轮交付与后续边界

以上是前期调研的边界。下面的实施版本选择人工关键帧直接奖励，而非方案 A 的 PBRS；原有 RynnValue / Robometer sidecar 和 IQL 算法保持不变。

FACTR 整臂参考姿态校准、末端跟随与手动重力补偿的测试命令见 [README_CN §3.6.1](../README_CN.md#361-factr-franka-校准手动重力补偿与无-vla-测试)。该控制器只影响采集动作来源，与本文件的奖励研究彼此独立。

## 6. 已实现：人工关键帧直接奖励

### 6.1 标注与归一化

数据详情页点击“切片 / 标记关键帧”，拖动主视角录像或逐帧定位，选择 `positive` / `negative` 后保存。切片仅保存标记，不删除视频帧、动作或 observation；完整接管前缀和成功后的记录仍然可见。标记绑定原始 observation step：N 个动作对应 0…N 共 N+1 个状态。原录像若只有 N 帧，最后一个状态会明确提示没有额外编码帧。

令 P、N 分别为手动 positive、negative 的数量（这里 N 是负关键帧数，不是动作数）。成功沿用连续 `data.success_consecutive_steps` 次环境 done 的首次确认，默认 5；确认动作后的 observation 自动成为 success 锚点，不需要也不能重复手动标记。令每个事件的绝对增量为：

\[
w=\begin{cases}
1/(P-N+1),&\text{确认成功}\\
1/(P+1),&\text{未确认成功}
\end{cases}
\]

从 `z_0=-1` 开始，每个 positive 加 w、negative 减 w，自动 success 再加 w，成功处精确取 0。失败末尾保持最后一个锚点的分数，**不强制回到 −1**。例如成功 P=3、N=1 时 w=1/3；失败 P=1、N=1 时 w=1/2。

严格保留公式结果，不裁剪到 `[-1,0]`：negative 在前可能低于 −1，某些顺序可能在成功前达到非负值。这是用户选择的实验语义，不代表成功判定；持续正分数可能鼓励停留，需在消融中检查。成功公式的分母必须大于 0。重复帧、超界帧、success 时刻及其后的手动标记拒绝保存；失败轨迹允许显式保存空标注，整条为 −1，但“没有标注文件”不等于空标注。

### 6.2 阶段间插值与 IQL reward

相邻锚点位于 s、e，采用：

\[
z(t)=z_s+(z_e-z_s)\left(\frac{t-s}{e-s}\right)^p,
\qquad p\ge1.
\]

默认 p=2；positive 前上升更快、negative 前下降更快，p=1 则为线性插值。当前数据严格 20 Hz，按 observation step 插值等价于按其真实时间插值。最后一个锚点后的分数保持不变；成功后记录为 0。这是基于未来人工边界的离线奖励重标注，不能宣称阶段间每一帧都有独立视觉进展证据。

**该分数直接作为奖励，不做 potential 差分，也不再叠加 sparse step cost。** 对起点 t、实际长度 L 的 chunk：

\[
\begin{array}{ll}
\text{macro:}&R_t=z(t+L),\quad y_t=R_t+\gamma m_tV(s_{t+L});\\
\text{cumulative:}&R_t=\sum_{h=0}^{L-1}\gamma^h z(t+h+1),\quad
y_t=R_t+\gamma^L m_tV(s_{t+L}).
\end{array}
\]

`m_t` 仍使用现有 terminal 规则。成功后记录用于详情显示，不重新进入现有 replay 的成功后采样；前缀去重、动作 mask、Q/V 更新顺序和 actor advantage 均不改变。

### 6.3 配置、版本与训练

```yaml
reward:
  source: stage  # sparse | rynnvalue | stage
  stage_exponent: 2.0
  gamma: 0.99
  accumulate_primitive_steps: false
```

训练页面“高级 IQL 参数”提供相同选项。详情中 p 控制保存后的预览；训练使用当前 YAML/表单的统一 `stage_exponent`，改变 p 只重算奖励，不需要重标关键帧。Sparse 不依赖评价，RynnValue 沿用既有 PBRS，Stage 不加载两种奖励模型。

每条轨迹旁保存 `stage_annotation.json`：schema、轨迹 hash、关键帧、确认成功 step、成功阈值、p 和标注内容 hash。保存只读取小型控制 NPZ，结果原子发布；拖动进度条不请求后端，不解压双视角 observation，也不调用 MuJoCo。并发编辑通过 revision 拒绝覆盖旧版本。

Stage 训练开始前检查全部数据成员，包含 validation 和被前缀去重覆盖的记录。缺失、失效、轨迹 hash 或成功阈值不一致时列出记录并停止，不跳过也不退回其他奖励。更改成功阈值后须检查标记并重新保存。GUI/终端任务冻结完整标注快照；训练目录保留该快照和 reward manifest，后续编辑不影响已启动训练。原始 NPZ 与标注一起迁移/打包可保留绑定；现有 UI 的轻量 CSV/视频 ZIP 会归档标注，但重建 NPZ 后 hash 改变，不能直接复用该标注。用于异机 Stage 训练应保留原始 trajectory.npz，不自动篡改绑定 hash。

第一轮建议固定数据、split、训练步数及评测初始状态，仅比较 Sparse、RynnValue、Stage 三种来源；分别报告成功率、耗时、分数范围与实际 actor 权重，不以曲线更平滑作为有效性的结论。
