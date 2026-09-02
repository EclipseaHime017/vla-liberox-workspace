# RynnValue 与 IQL 技术报告

本文依据 RynnValue 论文、其官方代码仓库以及 IQL 原论文整理。需要先明确两个边界。第一，RynnValue 是一个从视觉和语言估计任务时间价值的模型，不是执行机器人动作的策略。第二，RynnValue 论文中的离线策略是 π₀.₅，而本仓库后续实现使用 VLA-Adapter；两者遵循相同的奖励塑形和 IQL 更新思想，但模型输入、动作表示、损失函数和可训练参数并不完全相同。

## 1. RynnValue

### 1.1 要解决的问题

机器人强化学习首先需要回答“当前状态和动作有多好”。传统方法通常为每个任务手工编写奖励函数，或者只在任务完成时给出稀疏成功信号。手工奖励难以跨任务、机器人形态和场景迁移；稀疏奖励又无法区分任务早期、接近完成、发生退步以及最终失败等状态，因此很难为长时程操作提供充分的学习信号。

通用奖励模型试图从视频和语言中自动判断轨迹质量，但常见监督形式仍有明显限制。偏好学习需要成对比较数据，进度预测通常把每条轨迹内部归一化到固定区间，参考轨迹方法则依赖额外的成功示范。这些监督都带有任务内或轨迹内的锚点，不同数据源的动作速度、控制频率、视频长度和任务边界发生变化后，数值含义往往不再统一。模型即使学到一条平滑的进度曲线，也不一定真正理解画面中的任务状态。

RynnValue 将问题改写为语言条件下的时间距离预测：给定任务提示词、机器人形态信息和视觉观察，模型预测从当前观察到语义完成点还需要多少秒。时间戳是不同数据源都天然具有的物理量，因此不需要人工偏好排序，也不需要为每个任务定义专用的进度尺度。论文将这种量称为 temporal distance，也可以理解为面向指定语言目标的剩余时间或 cost-to-go。它不是环境动力学意义上的精确完成时间，而是模型依据当前视觉证据对“距离语言目标还有多远”作出的统一估计。

### 1.2 时间距离监督如何产生

原始机器人数据不能直接把视频末帧视为任务完成点，因为演示中可能包含完成后的停留、回撤和整理动作，也可能在一个长回合中连续完成多个子任务。RynnValue 先把原始轨迹整理为带有单一语言目标的片段，再为每个片段确定语义完成截点。默认可以使用片段终点，必要时依据数据源的结构或持续时间规则修正截点。截点之前的观察以“截点时间减去当前时间”作为绝对时间距离，截点及其后的观察被标记为零。这种处理把异构数据转换成了相同单位的密集状态监督。

### 1.3 模型输入与输出结构

RynnValue 建立在 RynnBrain 多模态骨干之上。一次输入包含任务指令、绝对与相对时间问题的系统提示、机器人形态元数据以及一组视觉观察。主要训练设置从每个视频片段采样八帧。视觉和文本被组织成一条交错的多模态序列，每个观察后插入一组绝对时间查询，从第二个观察开始还插入相邻观察之间的相对时间查询，序列末尾再加入自然语言分析与任务核验提示。每次时间预测不是只使用一个查询 token，而是使用八个同类型查询 token。组内 token 可以相互注意，骨干输出的八个隐藏状态会被拼接，再送入专门的数值头。

### 1.4 防止模型走捷径

时间标签具有很强的序列规律。如果总是均匀取帧并按时间顺序输入，模型可能只根据帧在序列中的位置拟合一条下降曲线，而不分析画面。RynnValue 通过不规则随机采样打破固定时间间隔，又通过时间顺序扰动打破“越靠后越接近完成”的位置关系。一半训练序列不按时间排序，另一半采用总体向前但允许回退的随机游走；相对时间标签则按照实际呈现顺序重新计算，因此既可以为正也可以为负。为了加强语言与视觉的绑定，训练中有 10% 的样本会替换为不匹配的任务指令。此时原轨迹的完成截点对新指令已经无效，因此绝对时间损失被屏蔽；相对时间仍描述两帧的时间位移，所以继续参与训练；语言分支则学习输出不匹配和未成功。

### 1.5 从时间距离到奖励

RynnValue 的原始绝对输出是非负剩余时间 \(v_t\)。剩余时间越小越好，而强化学习通常把越大的数值解释为越有价值，因此论文定义观察势函数 \(\Phi_t=-v_t\)。尚未完成时势函数通常为负，接近完成时趋近于零。这个符号变换保留了秒这一时间尺度，没有把不同任务分别归一化到零到一。

势函数本身不直接作为每一步奖励。论文采用 potential-based reward shaping，以相邻决策状态的折扣势函数差分构造形状奖励：

\[
r_t^{\mathrm{shape}}=\gamma\Phi_{t+1}-\Phi_t.
\]

这一设计衡量的是一次决策使“到目标的时间距离”发生了怎样的变化。它不会强制最终奖励形成平滑上升曲线；只要模型估计存在波动、状态停滞或发生退步，形状奖励就会相应波动或变为负值。

## 2. IQL

### 2.1 算法目的

离线强化学习只能使用固定数据集，不能通过与环境继续交互来修正错误。它的核心困难是分布外动作，Q 网络可能因为缺少真实监督而给出任意高估，误差会被反复放大。

Implicit Q-Learning 的目标是在不对数据集外动作求值的前提下完成策略改进。它训练 Q、V 和策略三个部分，但所有 Q 监督及策略监督都只使用数据集中真实出现的状态—动作对。所谓“implicit”不是不学习价值函数，而是不显式计算 \(\max_a Q(s,a)\)，也不在 critic 更新时采样当前策略的新动作。对高价值动作的偏好由 expectile value 和优势加权行为克隆隐式表达。

### 2.2 Q 与 V 分别表示什么

Q 函数 \(Q(s,a)\) 估计在状态 \(s\) 执行数据集动作 \(a\) 后，并继续按离线数据所隐含的可行行为推进时能够获得的折扣回报。V 函数 \(V(s)\) 不接收动作，它需要把同一状态附近不同数据动作的 Q 分布压缩成一个状态价值。IQL 不取该分布的硬最大值，而是通过非对称平方损失拟合一个较高的 expectile。

设差值为 \(u=Q(s,a)-V(s)\)。expectile 损失在 \(u>0\) 时使用权重 \(\tau\)，在 \(u<0\) 时使用权重 \(1-\tau\)。当 \(\tau>0.5\) 时，低估高 Q 动作的代价更大，V 会朝数据集中较好的动作价值移动；它仍是一个平滑统计量，而不是可能被噪声主导的硬最大值。这样既能体现策略改进方向，又不需要把未见动作输入 Q 网络。

实际实现通常维护两个在线 Q 函数 \(Q_1,Q_2\)，并在需要单一价值时取两者较小值。两个 Q 都拟合同一个 Bellman 目标，但参数初始化和训练误差不同；取最小值可以降低单个 critic 偶然高估对 V 和策略权重的影响。“更新 Q1/Q2”指分别优化两个估计器，“更新 Q”只是对整个双 Q 模块更新的简称，并不存在第三个额外 Q。

### 2.3 三组价值参数如何更新

IQL 的一次迭代先用缓慢变化的 target Q 为数据集动作提供参考，再更新 V。其核心目标可以写为让 \(V(s)\) 拟合 \(\min(Q_1^{target}(s,a),Q_2^{target}(s,a))\) 的高 expectile。target Q 不参与梯度更新，它是在线 Q 参数的延迟副本，因此这一阶段只有 V 的参数变化。

随后使用更新后的 V 构造 Q 的 Bellman 目标：

\[
y=r+\gamma mV(s'),
\]

其中 \(m\) 是 bootstrap mask，终止 transition 为零，非终止 transition 为一。目标 \(y\) 会停止梯度，Q1 和 Q2 分别以均方误差逼近它，因此这一步只有在线 Q 参数变化。Q 更新完成后，target Q 通过 Polyak 平均缓慢靠近在线 Q；target Q 没有独立优化器，其作用是降低 V 监督目标随每次 critic 更新剧烈移动的程度。

这条更新链形成一个闭环：target Q 决定 V 应该靠近数据动作价值分布的哪一部分，V 为下一状态提供无需查询新动作的 bootstrap，在线 Q 学习新的回报估计，再把较稳定的信息逐步传给 target Q。网络之间传递的是数值目标，不是相互贯通的反向传播路径，因此 critic、value 和 target critic 的参数职责清晰分离。

### 2.4 从价值学习到策略学习

价值网络训练完成后，数据动作的优势写为 \(A(s,a)=\min(Q_1(s,a),Q_2(s,a))-V(s)\)。正优势表示该动作优于状态下由 expectile 表示的基准，负优势表示它低于该基准。IQL 将优势变为指数权重 \(w=\exp(\beta A)\)，并通常设置最大权重避免少量估计误差造成极端梯度。

策略仍然只拟合数据集中已经执行过的动作，但高优势动作对监督损失的贡献更大，低优势动作的贡献更小。因此策略改进不是通过直接最大化 Q 或生成动作后再由 Q 筛选，而是通过 advantage-weighted behavioral cloning 完成。参数 \(\beta\) 控制选择强度：数值越大，策略越集中于 critic 判断为较好的数据动作，同时也越容易受到 Q 误差影响；expectile 参数决定 V 基准的高度，权重上限决定单个样本能够放大的最大程度。这三个参数共同决定离线数据利用与保守性之间的平衡。

IQL 的稳定性来自“只在数据支持内学习价值和策略”，但这也决定了它的能力边界。数据中没有覆盖的关键动作不能仅靠算法凭空产生；奖励错误、成功数据不足、视觉状态混淆和动作分布失衡都会传递到 Q、V 和策略权重中。Q 与 V 的绝对数值取决于奖励尺度、折扣和剩余时长，不能只根据其是否接近某个固定常数判断收敛；更重要的是 TD 误差、Q/V 相对关系、优势权重分布以及独立验证轨迹上的策略表现。

## 3. RynnValue-IQL

### 3.1 整体连接关系

RynnValue 论文中的离线学习由三个相互隔离的模型部分组成。冻结的 RynnValue 读取轨迹视觉历史和任务指令，为每个策略决策边界产生绝对时间距离，再转换为观察势函数和形状奖励。IQL 的 Q/V 网络读取离线轨迹中的视觉观察和已执行 action chunk，学习每个数据动作的回报与相对优势。π₀.₅ VLA 继续承担动作生成，只是其原有流匹配监督损失改由 IQL 优势加权。RynnValue 不接收策略梯度，Q/V 不参与部署，部署时真正执行的仍是经过加权后训练的 VLA。

论文为每个任务收集混合质量的离线数据，将成功与失败轨迹合并。附录报告的四个真机任务共有 398 条成功轨迹和 12 条失败轨迹。所有奖励变体使用相同的轨迹、成败标签、策略初始化与优化设置，仅替换势函数来源和形状奖励系数，因此实验比较的是奖励信息本身，而不是数据成员变化。

### 3.2 π₀.₅ VLA 的策略接口

论文使用流匹配形式的 π₀.₅ 作为基础视觉—语言—动作策略。VLA 输入包括缩放并填充到 224×224 的多相机图像、图像有效性 mask、任务指令和流匹配时间变量；不输入 proprioception 或机器人状态向量。策略一次预测长度为 16 的 action chunk，动作维度统一填充到 32。单臂任务的真实控制量是七个绝对关节位置和一个相对夹爪命令，双臂任务则包含两组关节与夹爪控制，其余维度只用于统一张量接口。

单臂任务的 VLA 和 IQL critic 都使用左侧第三人称与左腕相机，双臂任务再加入右腕相机。低层控制以 10 Hz 执行动作。RynnValue 奖励服务使用另一条右侧第三人称 RGB 序列和任务指令，对完整轨迹作因果历史评分。每个时刻的历史被均匀下采样为四帧输入；论文说明这样设置是为了与对比奖励模型保持相同推理协议。奖励标注只调用绝对时间头，不调用语言生成分支，然后在策略决策边界采样势函数序列。

### 3.3 action chunk transition 与奖励

离线轨迹在策略决策边界上切成 transition，记为 \((o_h,a_h,o_{h+1},m_h)\)。这里 \(a_h\) 不是单个低层动作，而是 VLA 输出的完整 action chunk；\(m_h\) 在终止 transition 上为零，其余为一。失败轨迹的所有 transition 都保留，成功与失败的区别通过环境终止和 sparse reward 表达，不由 RynnValue 重新分类。

论文把未完成 transition 的 sparse reward 设为 -1，把执行 chunk 后完成任务的 terminal reward 设为 0；失败轨迹始终为 -1。RynnValue 对边界观察给出绝对剩余时间 \(v_h\)，势函数为 \(\Phi_h=-v_h\)，形状奖励为 \(r_h^{shape}=\gamma_{off}\Phi_{h+1}-\Phi_h\)。最终进入 IQL 的奖励是 \(r_h^{off}=r_h^{sparse}+\kappa r_h^{shape}\)，其中论文对 RynnValue 使用 \(\kappa=0.1\)，离线折扣为 0.99。

论文附录的 \(h\) 是策略决策索引，因此公式把一个 action chunk 视作一个宏动作，并在相邻决策边界之间使用一次折扣。当前官方代码还提供了固定低层 horizon 的 Semi-MDP 写法：先把 chunk 内逐步奖励折扣累积为 \(R_t\)，再以 \(\gamma^H V(s_{t+H})\) bootstrap；逐步 PBRS 项会望远镜式化简为 \(\gamma^H\Phi(s_{t+H})-\Phi(s_t)\)。两种写法都可以自洽，但前提是奖励累计与 bootstrap 使用同一个时间单位。已经按宏动作构造的 reward 不能再次按低层步数折扣，否则会重复改变奖励尺度。

### 3.4 Q、V 与 target Q 的网络结构

论文中的辅助价值网络只服务于后训练。Q 接收当前多相机观察和展平的 16×32 action chunk，输出该观察—动作组合的标量回报；V 只接收当前多相机观察，输出状态标量价值。视觉编码器采用 ResNet-18、GroupNorm、spatial softmax 和 50 维瓶颈，后接隐藏维度为 256、256 的 MLP。当前观察和下一观察都使用随机裁剪，不使用颜色抖动。

两个 Q 以 ensemble 形式维护。官方结构让 Q1/Q2 使用同一 critic 视觉表示，再进入参数独立的 Q MLP；V 则拥有独立的视觉编码器和 MLP。这样 Q ensemble 可以在相同状态特征上提供两个独立回报估计，而 V 不会与 Q 共享会被不同目标同时拉动的 encoder 参数。target Q 是整个在线 critic 的延迟参数副本，并不是第三个接受梯度训练的网络。

论文的 critic/value 路径不接收 proprioception，也不复用 π₀.₅ 的视觉语言隐藏状态。VLA、Q/V 和 RynnValue 因而各自维护视觉表示：VLA 表示用于生成动作，Q/V 表示用于离线回报估计，RynnValue 表示用于时间距离预测。这增加了训练成本，但把奖励建模、价值估计和动作生成的目标隔离开来。

### 3.5 一次 IQL 更新的准确顺序

每个优化 step 首先更新 V。对 batch 中真实执行的 action chunk，target Q1 和 target Q2 计算两个回报估计并取最小值，V 以 expectile 0.8 拟合这个参考值。此时 target Q 完全停止梯度，只有 V 及其视觉编码器更新。使用 target 而不是快速变化的 online Q，目的是给 V 提供较平稳的回归目标。

第二步更新 online Q1/Q2。更新后的 V 在下一观察上给出 \(V(o_{h+1})\)，Bellman 目标由离线 reward、bootstrap mask 和折扣后的下一状态价值组成。目标值停止梯度，两个 online Q 分别以均方误差拟合它，因此梯度不会流入 V。之后 target Q 参数按 0.005 的 Polyak 比例向最新 online Q 移动，使它保留历史平滑效果。

第三步计算策略权重。官方实现用更新后的 online 双 Q 最小值减去当前 V，形成每条数据 action chunk 的优势，再计算 \(\min(\exp(\beta A),w_{max})\)。论文设置 \(\beta=10\)、\(w_{max}=100\)。权重被视为固定监督系数，不把 VLA 的梯度反向传播进 Q/V。前 200 个优化 step 仍训练 Q/V，但策略权重强制为一，使 π₀.₅ 先执行普通的行为克隆式流匹配更新，待价值估计具有基本尺度后再启用优势筛选。

最后更新 VLA。标准 IQL 通常训练一个显式概率策略并最大化加权对数似然；RynnValue 论文不另外建立 IQL actor，而是保留 π₀.₅ 原有的流匹配训练目标。模型仍学习从带噪动作和流时间预测正确的流向量，只是每条样本的流匹配损失先在 action horizon 上归约，再乘对应的 IQL 优势权重。由此，Q/V 不负责生成动作，而是决定哪些已记录 action chunk 应对 VLA 参数产生更强的更新。

这四步的参数流向彼此独立：V 损失只更新 value 网络，TD 损失只更新 online Q，Polyak 操作只复制并平滑 target Q，优势加权流匹配只更新 VLA。奖励和数值估计在模块间向前传递，梯度不会从 VLA 穿过优势权重进入 critic，也不会从 critic 穿过奖励进入 RynnValue。

### 3.6 训练配置与算法目的

论文每个任务训练 10,000 个优化 step，batch size 为 64。Q/V 使用学习率 \(3\times10^{-4}\) 的 Adam；VLA 使用 AdamW，峰值学习率 \(3\times10^{-5}\)、最终学习率 \(3\times10^{-6}\)，先线性 warm-up 2,000 step，再余弦衰减，并对策略参数维护 0.99 的 EMA。策略梯度范数裁剪为 1。离线 IQL 和监督微调都从同一个预训练 π₀.₅ checkpoint 独立开始，IQL 并不是先完成 SFT 再继续训练。论文报告每个任务使用两张 80 GB GPU，IQL 训练约 16 小时。

RynnValue-IQL 的核心目的不是让 RynnValue 直接替代策略，也不是让 critic 搜索任意新动作，而是把三种能力组合起来：RynnValue 把异构视觉轨迹转换成目标条件时间势函数；PBRS 把势函数转换成与任务成败兼容的中间奖励；IQL 只在离线数据支持内学习哪些 action chunk 更有价值；流匹配 VLA 再通过加权模仿吸收这些高价值行为。系统最终部署时只需要后训练后的 VLA，奖励模型与 Q/V 网络可以退出执行路径。

### 3.7 与本仓库 VLA-Adapter 适配的边界

本仓库的 `vla-adapter-rynn-iql/` 沿用 RynnValue 的绝对时间距离、势函数塑形、双 Q、expectile V、target Q 和优势加权策略更新，但它不是论文 π₀.₅ 实验的逐层复制。当前执行策略是 VLA-Adapter/LIBERO-Object-Pro，最大 action horizon 为 8，动作是 7 维 normalized OSC_POSE，并额外输入 8 维 proprioception；策略损失使用 VLA-Adapter 连续 action head 的 masked L1，而不是 π₀.₅ 流匹配损失。为适应单卡资源，critic 的图像分辨率和编码器规模也与论文的 224×224 ResNet-18 配置不同。

参数更新范围同样不同。论文对 π₀.₅ 进行流匹配策略后训练，没有把结论限定为只更新一个 action head；本项目冻结视觉与语言 backbone，只允许 continuous action head 和 proprio projector 接收优势加权损失的梯度。action head 把 VLA 隐藏表示转换为连续动作序列，proprio projector 把机器人自身状态映射到模型隐藏空间。IQL 优势不会直接写入这两个模块，而是作为每条样本损失的乘数：加权损失反向传播后，普通优化器才据此改变 action head 和 projector 参数。

因此，在分析实验结果时应区分“算法语义一致”和“网络实现相同”。奖励势函数、离线 Bellman 更新与优势加权的因果关系可以保持一致，但不同动作空间、chunk 长度、可训练参数、视觉编码器、loss 形式和显存配置都会改变优化动态。论文结果能够证明这条组合路径有效，却不能直接作为当前 VLA-Adapter 配置的数值复现保证。

## 4. 具体实现差异分析

### 4.1 对比边界

本节区分三套容易被混用的实现。第一套是 IQL 论文和 RynnValue 官方 `pi_iql` 代码所定义的参考实现；第二套是本仓库提交 `f82fbe3f`，它曾训练出相对较好的 VLA-Adapter checkpoint；第三套是当前 `oldR` 分支提交 `d9152cf`。`oldR` 继续保留 chunk 内逐步累计的 discounted reward 和实际长度 \(\gamma^L\) bootstrap，但已经把 value、critic 和策略权重的更新恢复为官方 IQL/RynnValue 语义。

这三套实现共享双 Q、expectile V、target Q、PBRS 和优势加权行为克隆等总体结构，但“共享总体结构”不代表数值训练过程等价。一次优化 step 内先读取哪组参数、策略权重使用哪一组 Q、每步实际包含多少 transition，以及 Adam/AdamW 的内部状态如何演化，都会影响小数据集和有限训练步下的瞬态优化结果。

### 4.2 V、online Q 与 target Q 的更新顺序

IQL/RynnValue 官方实现先用冻结的 target Q 更新 V。对数据集中的真实动作取双 Q 最小值：

\[
\bar Q_k(s,a)=\min\left(Q^{target}_{1,k}(s,a),Q^{target}_{2,k}(s,a)\right),
\]

再通过 expectile loss 得到本轮更新后的 \(V_{k+1}\)。随后用这个新 V 在下一状态上的输出构造 Bellman target：

\[
y_k=R_t+\gamma^{L_t}m_tV_{k+1}(s_{t+L_t}),
\]

并据此更新 online Q，最后才执行 target Q 的 Polyak 更新。顺序可以概括为：

```text
target Q_k → V_{k+1} → online Q_{k+1} → target Q_{k+1}
```

`f82fbe3f` 使用的是 Q-first 顺序。它在本轮 V 更新之前读取 \(V_k(s')\)，先构造 Bellman target 并更新 online Q，再使用仍冻结的 target Q 更新 V，最后更新 target Q：

```text
V_k → online Q_{k+1} → V_{k+1} → target Q_{k+1}
```

因此二者的 V 更新参考都来自 target Q，但 Q 所使用的下一状态价值不同。官方实现使用刚更新的 \(V_{k+1}(s')\)，`f82fbe3f` 使用本轮开始时的 \(V_k(s')\)。当学习率足够小且 V 变化缓慢时，二者可能接近；在随机初始化、batch 很小、奖励长期为负和训练步有限的条件下，这一个优化 step 的滞后可能改变 Q/V 的早期下降速度，并进一步影响策略权重。当前 `oldR(d9152cf)` 已恢复官方的 V-first 顺序，而此前的 `4c2aeb0` 仍是为复现历史曲线而保留的 Q-first 实验版本。

### 4.3 Advantage 的定义与参数时刻

IQL 的策略提取使用数据动作相对于当前状态基准的优势：

\[
A(s,a)=\min\left(Q_1^{online}(s,a),Q_2^{online}(s,a)\right)-V(s).
\]

RynnValue 官方代码在本轮 V 和 online Q 都完成更新后计算该优势，再形成策略权重：

\[
w(s,a)=\min\left(\exp(\beta A(s,a)),w_{max}\right).
\]

`f82fbe3f` 没有使用此处的 online Q，而是使用 Polyak 平滑后的 target Q：

\[
A_{f82}(s,a)=\min\left(Q_1^{target}(s,a),Q_2^{target}(s,a)\right)-V(s).
\]

target-Q Advantage 变化较慢，可能降低权重的短期抖动，但它也会延迟 critic 新信息传给 action head，并让初始化阶段的旧 Q 估计保留更久。这是一种具有稳定化动机的工程变体，不是 IQL 论文或 RynnValue 官方代码规定的策略目标。当前 `oldR(d9152cf)` 已改为使用本轮更新后的 online Q 和当前 V；target Q 只负责为 V 提供稳定的 expectile 参考，不再直接决定 actor 权重。

两种 Advantage 不能只比较均值。由于策略使用指数映射，即使 \(A\) 的平均值接近，方差、正值比例以及被 \(w_{max}\) 截断的比例也可能完全不同。实验对比至少需要同时记录 `actor_advantage_mean/std/min/max`、`advantage_weight_mean/max`、权重为一附近的比例和达到上限的比例，否则无法判断实际施加到 action head 上的监督强度。

### 4.4 train step、micro batch 与梯度累积

本项目中的一个 `train_step` 表示一次 Q/V 优化，而不是遍历一遍数据集，也不是一次 actor 参数更新。设 micro batch 为 \(B\)，梯度累积步数为 \(G\)，总训练步数为 \(N\)，则 Q/V 每一步立即更新一次，每次使用 \(B\) 条 transition；action head 和 proprio projector 则累计 \(G\) 个 micro batch 后更新一次。近似训练规模为：

\[
N_{critic\ updates}=N,
\]

\[
N_{critic\ samples}=N B,
\]

\[
B_{actor\ effective}=B G,
\]

\[
N_{actor\ updates}\approx\left\lceil\frac{N}{G}\right\rceil.
\]

`f82fbe3f` 强制 `micro_batch_size=1`，默认 `gradient_accumulation_steps=32`、`train_steps=10000`。因此 critic 约处理 10,000 条随机抽样 transition并更新10,000次，actor有效batch约为32，并更新约313次。强制batch为1不是IQL算法要求，而是当时 VLA processor、双视角配对和隐藏状态提取只实现了单样本输入；梯度累积只扩大actor有效batch，没有扩大critic batch。

当前实现支持单任务内的真正批处理。Replay样本经过 `default_collate` 形成 \([B,\ldots]\) 张量；prompt、agentview和wrist图像以相同batch顺序送入processor；两个相机产生的 token batch经过一致性校验后再组合；冻结VLA一次前向生成 \(B\) 条隐藏表示。Pixel-IQL也直接接收这 \(B\) 条transition，每条样本保留自己的action mask、实际 \(L_t\)、reward和bootstrap mask。当前仍要求一个micro batch中的任务prompt相同，避免不同token长度破坏action-query位置。

扩大 \(B\) 后如果仍沿用10000个train step，训练规模会同步扩大。例如 `B=8,N=10000` 会让critic处理约80,000条transition，而不是原来的10,000条；如果同时保持 \(G=32\)，actor有效batch会从32扩大为256，但actor更新次数仍约313次。反过来，简单把 \(N\) 除以8虽然能保持总样本数，却会把critic参数更新次数从10,000降到1,250，Adam状态、学习率日程和target Q软更新次数也随之减少，因此仍不与原训练等价。

论文中的10,000 step、batch size 64表示每次价值更新都基于64条样本，不能直接与本项目早期的 `1×10000` 比较。复现实验必须同时报告 \(N\)、\(B\)、\(G\)、critic总样本数、critic更新次数和actor更新次数。只写“训练10000轮”无法确定实际训练规模。

### 4.5 Reward的时间单位、归约方式与评价来源

RynnValue论文的离线策略在决策边界上把一个完整action chunk视为一个宏动作。未完成chunk的稀疏奖励为-1，成功终止chunk为0，形状奖励使用相邻决策状态的势函数差：

\[
r_h^{shape}=\gamma\Phi_{h+1}-\Phi_h,
\]

最终用于IQL的单个宏动作reward为：

\[
r_h^{final}=r_h^{sparse}+\kappa r_h^{shape}.
\]

对应Bellman target只跨越一个宏决策步：

\[
y_h=r_h^{final}+\gamma m_hV(s_{h+1}).
\]

主分支一度采用这套macro-action reduction。它的优点是不同长度chunk的稀疏部分都保持约-1，RynnValue势函数差相对于稀疏项不会因为chunk包含更多低层step而被额外稀释；但如果chunk长度变化，一个宏动作代表的真实物理时间也随之变化，单次固定折扣不再严格表达20 Hz时间尺度。

`f82fbe3f` 与当前`oldR`采用逐低层step的Semi-MDP reduction。对实际执行长度为 \(L_t\) 的chunk，先累计每个primitive step的稀疏代价：

\[
R_t^{sparse}=\sum_{h=0}^{L_t-1}\gamma^h r_{t+h}^{sparse},
\]

再利用PBRS的望远镜性质计算完整chunk的势函数项：

\[
R_t^{shape}=\gamma^{L_t}\Phi(s_{t+L_t})-\Phi(s_t),
\]

最终reward和Bellman target分别为：

\[
R_t^{final}=R_t^{sparse}+\kappa R_t^{shape},
\]

\[
y_t=R_t^{final}+\gamma^{L_t}m_tV(s_{t+L_t}).
\]

这种写法在数学上把action chunk视为持续 \(L_t\) 个低层时间步的Semi-MDP动作，reward累计和bootstrap使用同一时间单位。长度不足8的接管中断chunk、terminal chunk和source切换chunk分别使用自己的真实 \(L_t\)，action mask只负责补齐张量，不会把Bellman折扣强制改成 \(\gamma^8\)。`4c2aeb0`修复的正是曾经出现过的“reward按 \(L_t\) 累计、bootstrap却只使用 \(\gamma\)”的不一致。两种reward reduction都能形成自洽的目标，但数值尺度明显不同。当 \(\gamma=0.99,L=8\) 且chunk内尚未成功时，oldR的稀疏部分约为：

\[
-\sum_{h=0}^{7}0.99^h\approx-7.73,
\]

而论文宏动作写法为-1。PBRS部分在两种写法中仍然主要由chunk两端势函数差决定，不会因为chunk含有8个primitive step就自动放大8倍。因此在 `shaping_weight=0.1` 不变时，oldR中稀疏项相对于Shape Reward更强，RynnValue对Final Reward的相对影响会比macro-action reduction更弱。

即使 `f82fbe3f` 和当前oldR采用相同的逐步归约公式，它们的势函数输入仍不等价。`f82fbe3f` 配置为 `max_frames=32, window_overlap=8`，长轨迹通过带重叠窗口评价并合并结果；当前评价依据RynnValue离线协议，对每个决策边界使用从轨迹起点到当前时刻的因果前缀，并均匀抽取4帧，读取官方absolute temporal-distance head。32帧窗口提供更多局部运动历史，4帧前缀则更贴近论文协议，但对接触、夹爪闭合和短暂失败等细粒度状态的估计可能具有不同噪声。即使选择同一批trajectory并执行“覆盖评价”，得到的 \(v_t\)、\(\Phi_t\) 和PBRS reward也可能显著不同。

因此，要判断reward是否导致 `f82fbe3f` 与当前模型的性能差异，至少需要固定相同的trajectory成员和chunk边界，并分别保存两套RynnValue输出与两套确定性reduction。只比较 `reward_sha256` 能确认输入是否相同，但不能说明差异来自4帧/32帧评价还是macro/oldR归约

### 4.6 优化器、weight decay 与梯度尺度

论文和RynnValue官方配置对 Q/V 使用 Adam，学习率为 \(3\times10^{-4}\)；VLA策略使用 AdamW，参数为 `betas=(0.9,0.95)`、`eps=1e-8`、近似为零的 `weight_decay=1e-10`。当前 `oldR` 默认采用这组配置，并允许在YAML中显式选择 Q/V 的 `adam|adamw`、weight decay和梯度裁剪阈值。Q/V梯度裁剪为10，actor梯度裁剪为1。

`f82fbe3f` 对 Q/V 直接调用未显式指定weight decay的 `torch.optim.AdamW`，actor也使用PyTorch默认AdamW。其实际默认参数包含 `betas=(0.9,0.999)` 和 `weight_decay=0.01`。因此它与当前实现至少存在两类重要差异。首先，`beta2=0.999` 对平方梯度采用更长时间尺度的平滑，面对剧烈变化的Advantage监督时通常比0.95反应更慢；其次，0.01的weight decay会设计持续收缩Q/V encoder以及action head参数，在当前小数据集上可能形成有效正则化，也可能造成不必要的偏置。当前actor几乎不施加weight decay，不能把两次训练都笼统称为“使用AdamW”并认为等价。

双Q损失的归约也发生过变化。`f82fbe3f` 使用两个MSE之和：

\[
L_Q=L_{Q_1}+L_{Q_2},
\]

当前实现使用ensemble均值：

\[
L_Q=\frac{1}{2}\left(L_{Q_1}+L_{Q_2}\right).
\]

Adam会在一定程度上抵消整体梯度缩放，因此这项差异通常小于reward、Advantage来源或优化器超参数变化，但在梯度裁剪、epsilon、weight decay和训练初期存在时仍不严格等价。若要解释 `f82fbe3f` 的性能，必须把“AdamW默认正则化”“actor的beta2”和“双Q loss缩放”拆成独立消融项，不能一次全部改动后只归因于某一个算法选择。

### 4.7 当前结论与对照原则

`f82fbe3f` 的较好效果只能说明那一整套训练条件在已有实验中产生了更好的checkpoint，不能单独证明Q-first或target-Q Advantage更合理。除本节讨论的更新顺序、Advantage和优化器外，它还使用32帧带重叠的旧RynnValue评价流程，而当前评价使用4帧因果前缀；分支轨迹的Replay去重和reward sidecar结构也发生了变化。这些差异会改变训练样本及reward本身。

后续对照应固定同一份selection manifest、trajectory/observation哈希、reward数组、随机种子和基础checkpoint，再一次只改变一个因素。建议依次比较：官方V-first与历史Q-first；online-Q与target-Q Advantage；当前Adam配置与 `f82fbe3f` 默认AdamW；actor的两组beta与weight decay；最后再单独比较micro batch。每次运行同时保存dataset/reward哈希、chunk来源比例、reward分布、Advantage与权重分布以及实际参数更新次数，才能把训练曲线差异和最终仿真成功率建立可验证的因果联系。

| 分支 | Reward时间单位 | 更新顺序 | Actor Advantage | 定位 |
|---|---|---|---|---|
| `main` | 每个chunk作为一个宏动作 | V → online Q → target Q | online Q − V | 最接近IQL/RynnValue官方设计 |
| `oldA` | 每个chunk作为一个宏动作 | V → online Q → target Q | target Q − V | 只测试旧Advantage |
| `Qfirst` | 每个chunk作为一个宏动作 | online Q → V → target Q | target Q − V | 同时测试旧更新顺序和旧Advantage |
| `oldR` | chunk内primitive step折扣累积 | V → online Q → target Q | online Q − V | 只测试旧reward时间尺度 |

## 参考资料

- [RynnValue: Scaling Robotic Value Foundation Models with Temporal Distance](https://arxiv.org/abs/2608.09853)
- [RynnValue 官方代码仓库](https://github.com/alibaba-damo-academy/RynnValue)
- [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- [IQL 官方代码仓库](https://github.com/ikostrikov/implicit_q_learning)
