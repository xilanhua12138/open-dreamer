# CoinRun world-model 实验全链复盘

> 时间范围：2026-07-28 至 2026-07-31
>
> 范围：Tokenizer scale、直接训练 Dynamics、PPO 轨迹采集、Dynamics 数据混合与修复、Live Demo
>
> 目的：解释每次实验为什么发生、改变了什么、学到了什么，以及哪些失败来自模型、数据、训练协议、运行时或验证工具。

## 一句话结论

这条链并不是“把 Dynamics 做大就会变好”。真正的依赖顺序是：

```text
可逆且清晰的视觉表示
    → 覆盖合理状态—动作分布的真实轨迹
    → 足够长且优化充分的 action-conditioned Dynamics
    → 不污染状态、正确复用缓存的实时推理运行时
    → 人直接检查画面、时间连续性和动作响应
```

我们前后遇到的主要问题分别属于这五层：

1. 最初的 `0.17M` Tokenizer 只能证明管线能跑，不能保留 CoinRun 的小人、障碍和边缘细节。
2. Tokenizer 的第一次 scale pilot 给不同规模分配了极不相同的训练步数，把模型容量和优化程度混在了一起。
3. 直接用随机动作和小预算训练 Dynamics，虽然能得到 PSNR，却没有覆盖“持续向右跑、跳跃、拿金币”的状态—动作分布。
4. 第一版 PPO 明显低于公开 CoinRun 曲线；对齐公开配方后恢复，单因素实验进一步把主要退化定位到不适合当前无归一化残差编码器的 orthogonal `sqrt(2)` 初始化。
5. PPO checkpoint 的正确问题不是“四个 checkpoint 分别训练四个世界模型”，而是在固定总数据量下混合不同策略阶段的轨迹。
6. `20k / k_max=8 / 64-frame record` 的 Dynamics pilot 太短，而且 64 帧记录配 64 帧窗口让 reward-biased crop 完全失效。
7. 参考式修复第一次直接读取 RGB，每个 optimizer update 都重复运行冻结的 16.6M Tokenizer，算力浪费在重复编码上；因此改成一次性离线 latent。
8. Live Demo 先后暴露了 dtype 漂移、阻塞式逐帧 RPC、重复 context prefill、错误 worktree import、checkpoint 路径语义和输入 revision 时钟单位等运行时问题。这些问题会让一个模型看起来“完全没反应”，但不能归因给模型权重。

连续运行时已经工作；这只证明交互外壳正确。`CR-DYN-0011` 现已完成
`200k` Dynamics 训练、shortcut、256-step full diffusion 和 action controls。
所有预注册数值门通过，但终态视觉验收仍未完成，因此不能把数值通过写成可用 Demo。

## 1. 先把问题拆成四个独立对象

| 对象 | 它回答的问题 | 不能替代什么 |
|---|---|---|
| Tokenizer | 单帧能否压进 latent 后清晰还原 | 不能说明未来预测正确 |
| PPO / trajectory corpus | 数据是否覆盖合理动作和可达状态 | 不能说明世界模型学会了这些动作 |
| Dynamics | 给定历史和未来动作，latent future 是否正确 | 平均 PSNR 不能说明动作可控或长期稳定 |
| Live Demo | 状态、缓存、输入和传输是否按实时语义工作 | `/health`、HTTP 200 和低延迟不能说明模型好 |

早期最大的认知混淆，是把这些局部指标串成了过强结论：

```text
health 通过 ≠ 连续交互可用
PSNR 上升 ≠ 动作响应正确
动作响应 API 正常 ≠ 模型使用了动作
Tokenizer clean PSNR 高 ≠ Dynamics rollout 清晰
训练完成 ≠ 复现 OpenDreamer 官方 CoinRun 原型
```

## 2. Tokenizer scale：从“能跑”到“公平比较”

### 2.1 `CR-TOK-0001`：最小模型只用于管线闭环

最早使用的是发布标签 `0.17M`、本地精确参数量 `163,392` 的 Tokenizer。
它在原生 scaling budget 下训练了 56,928 steps，EMA held-out clean PSNR
只有 `24.2447 dB`。

这次实验的价值是确认：

- 数据读取、训练、checkpoint、重建和评估链可以在一张 A10 上闭环；
- 它不是“视觉质量足够”的证据；
- 后来的第一次 Live Demo 也直接验证了：小人、平台和边缘细节不够清晰。

所以 `CR-TOK-0001` 是工程 smoke，不是 Dynamics 的长期表示选择。

### 2.2 `CR-TOK-0002`：第一次 scale pilot 为什么不公平

第一次 scale 实验的五档精确参数和实际分配步数是：

| 标签 | 精确参数 | 分配 updates |
|---|---:|---:|
| 0.17M | 163,392 | 62,675 |
| 1.1M | 1,048,192 | 9,358 |
| 3.7M | 3,342,528 | 2,938 |
| 8.6M | 7,734,528 | 2,550 |
| 16.6M | 14,912,320 | 2,656 |

这个设计回答的是“固定某种 scaling compute 分配时，各档表现如何”，不是
“模型更大是否重建更好”。最大模型只训练约 2.6k steps，而最小模型训练 62.7k，
容量和优化程度完全混杂。

此外，旧 runner 还保留了一个 conditional quality gate，只完成前三档就退出。
因此该实验被正确关闭为 `aborted / inconclusive`，而不是拿残缺结果解释 scale。

这一轮最重要的方法论修正是：

> 如果目标是做容量消融，所有模型必须无条件跑完，并在相同 completed updates 上比较学习曲线。

### 2.3 `CR-TOK-0003`：固定 20k 的五档公平曲线

修正后，五档全部从头训练恰好 20,000 optimizer updates，并在
2.5k / 5k / 10k / 20k 使用同一 512-clip held-out 集评估。

最终 EMA：

| 标签 | Clean PSNR | Edge PSNR | Temporal-change PSNR |
|---|---:|---:|---:|
| 0.17M | 12.0610 | 11.9328 | 11.4600 |
| 1.1M | 27.7460 | 20.9956 | 21.2332 |
| 3.7M | 32.7710 | 24.5359 | 25.2065 |
| 8.6M | 34.8962 | 25.9151 | 26.5915 |
| 16.6M | 36.1691 | 26.7739 | 27.4284 |

这次才建立了可信的固定步数容量曲线：

```text
16.6M > 8.6M > 3.7M > 1.1M > 0.17M
```

其中三个指标的含义：

- `clean`：完整 held-out 图像的平均重建质量；
- `edge`：只强调边缘区域，近似检查小人、平台和障碍轮廓；
- `temporal-change`：强调相邻帧发生变化的区域，减少静态背景把平均分“抬高”。

### 2.4 `CR-TOK-0004`：补齐“28.7M”发布标签

仓库里的“28.7M”是发布规模标签，本地 depth-6、`d_model=384` 实现的精确参数量
是 `25,564,032`，不能把标签当成本地精确计数。

在相同固定 20k 协议下，最终 EMA 为：

| 模型 | Clean | Edge | Temporal-change |
|---|---:|---:|---:|
| 16.6M | 36.1691 | 26.7739 | 27.4284 |
| “28.7M”标签 | 36.8657 | 27.0661 | 27.6592 |
| 增量 | +0.6965 | +0.2921 | +0.2308 |

它数值上最好，但相对参数增长，边缘和变化区收益已经很小。后续 Dynamics 固定
16.6M，是算力和表示质量的折中，不是因为更大的 Tokenizer 无效。

这次还暴露了两类工程故障：

- 零步 `ConfigAttributeError: dataset.mouse_repr`：validation helper 直接读取了未显式
  materialize 的 dataclass 默认值。先写红测试后修复，科学协议未变。
- 训练到 1,179 updates 时 JAX 报 `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`。
  没有 NVIDIA Xid 或 OOM，保留失败 attempt 后，只从本实验 step-0 checkpoint 恢复。

### Tokenizer 阶段最终决策

- 不再用 `0.17M` 支撑可用 Demo；
- Dynamics 主线固定 `16.6M EMA`；
- “28.7M”保留为数值上更好的上界，但不为约 0.2–0.7 dB 的收益立即付出后续全部算力；
- Tokenizer 与 Dynamics 必须保持 latent 接口一致：`n_latents=16`、
  `d_bottleneck=16`。

## 3. 直接训练 Dynamics：为什么 20k scale 能排序，却仍不可用

### 3.1 `CR-DYN-0001`：随机动作最小闭环

第一版 Dynamics 约 `0.55M` 参数、15k steps、4 帧 context、16 帧 future，
使用随机动作数据。Dynamics-only mean PSNR 为 `16.6342 dB`。

它证明 action-conditioned pipeline 可以训练、保存、rollout 和打分；没有证明：

- 数据包含合理跑跳行为；
- 模型理解动作；
- 递归 rollout 稳定；
- 用户能把它当 CoinRun 玩。

这与 OpenDreamer 原文的经验一致：随机动作轨迹训练出的 Dynamics 很差，后来需要
简单 RL agent 采集更合理轨迹。

### 3.2 `CR-DYN-0002`：固定 FLOPs 把大模型饿死

固定 `C=1e15` 时：

| 模型 | 参数 | updates |
|---|---:|---:|
| tiny | 155,840 | 15,217 |
| small | 545,920 | 4,768 |
| medium | 3,931,392 | 474 |

medium 只有 474 total updates，bootstrap 到 237 才开始。这个实验只能说明
“在极小固定计算预算下，大模型没有优化充分”，不能说明“大模型更差”。

因此它被保留为 `aborted / inconclusive`，随后改做固定 20k。

### 3.3 `CR-DYN-0003/0004`：固定 20k 的容量曲线

相同 20k 配方下：

| 模型 | 参数 | PSNR@16 | SSIM@16 |
|---|---:|---:|---:|
| tiny | 0.16M | 12.77 | 0.337 |
| small | 0.55M | 14.17 | 0.465 |
| medium | 3.93M | 16.02 | 0.542 |
| large | 12.90M | 15.64 | 0.531 |

tiny → small → medium 单调提升；large 又低于 medium。这里的合理解释不是
“scale 定律失效”，而是：

- 数据量很小；
- 每档只有 20k；
- 大模型更可能欠优化；
- 任务仍使用随机或较弱行为分布；
- 评估只覆盖当前 pilot 的 held-out 定义。

所以 medium 只是该协议下的描述性 winner。

### 3.4 `CR-DYN-0005`：历史长度不是越长越好

固定 checkpoint、固定同一个 future hash：

| Context | PSNR@16 | SSIM |
|---:|---:|---:|
| 4 | 16.88 | 0.569 |
| 16 | 17.15 | 0.592 |
| 32 | 17.02 | 0.589 |

16 帧优于 4 帧，32 帧没有继续提升。后续使用 context 16。

### 早期 Dynamics 的错误结论风险

这些实验能用于比较相对趋势，但不能将 `16–17 dB` 描述为可用。第一次 Live Demo
已经给出直接反例：数值最高的 checkpoint 仍然模糊、漂移、对动作反应不可信。

## 4. PPO：为什么要训练，以及低 reward 到底出了什么问题

### 4.1 PPO 的角色只是采集 Dynamics 数据

目标不是在 learned world model 里训练 agent，也不是做 behavior cloning。
PPO 只负责在真实 CoinRun 环境中产生比随机动作更合理的：

```text
(frame_t, action_t, reward_t, terminal_t)
```

轨迹，然后冻结 policy，采集 train/eval corpus 给 Dynamics。

### 4.2 `CR-PPO-0001`：第一版 policy 确实偏弱

协议为 25,165,824 transitions、500 train levels。最佳 deterministic milestone
在 22.02M 达到：

- mean return `5.15625`
- success `51.5625%`

最终 checkpoint 反而回落到：

- mean return `2.96875`
- success `29.6875%`

改用 512-episode stochastic full-distribution evaluator 后：

- best checkpoint：`5.390625`
- final checkpoint：`4.9609375`

说明旧 deterministic evaluator 有偏差，但测评偏差不足以解释全部低分。
好的一面是数据工程闭环有效：最终 policy 仍采集出
4,096 train + 512 eval、每条 64 帧的轨迹，split/action/frame pair audits 全通过。

运行环境还暴露了两个兼容性问题：

- CUDA 12.9 namespace 与 JAX 0.4.35 不兼容，固定到 CUDA 12.4.131；
- logger 假设 `jax.distributed.is_initialized()` 总存在，旧 JAX 没有该 API，
  改成明确的 optional capability 检查。

### 4.3 `CR-PPO-0002`：先对齐公开 easy-200 曲线

在相同 25.17M transition 预算下，对齐：

- 200 train levels；
- reward normalization gamma `0.99`；
- 每 minibatch advantage normalization；
- Glorot IMPALA backbone；
- stochastic `num_levels=0` evaluation。

最终 512 episodes：

- mean return `8.671875`
- success `86.71875%`

这证明 PPO package 可以跑到公开 CoinRun 曲线区间，但由于一次改了多个设置，还不能说
具体哪一个是“bug”。

### 4.4 `CR-PPO-0003`：单因素定位

每个 arm 只恢复一个旧设置，全部跑到 6.29M transitions：

| Arm | Mean return | Success | 相对 reference |
|---|---:|---:|---:|
| reference | 7.7344 | 77.34% | — |
| levels500 | 8.3594 | 83.59% | 更好 |
| reward gamma 0.999 | 8.3594 | 83.59% | 更好 |
| whole-batch advantage | 7.7734 | 77.73% | 基本不变 |
| orthogonal `sqrt(2)` | 5.3906 | 53.91% | -2.3438 / -23.44 pp |

只有 orthogonal `sqrt(2)` 跨过预注册的 collapse threshold。

这不是“orthogonal initialization 普遍有 bug”。具体机制是当前 IMPALA encoder：

- stem + 6 个 residual blocks + dense；
- 没有 normalization；
- 每个 block 是 `x + F(x)`；
- residual branch 没有缩放；
- 所有卷积和 dense 同时用 orthogonal gain `sqrt(2)`。

于是每一层都放大信号，残差又继续相加。

### 4.5 `CR-PPO-0005`：机制 probe 与缓解

相对 Glorot，orthogonal `sqrt(2)` 在零步 probe 中放大：

- encoder RMS：`26.2468×`
- JVP gain：`27.5101×`
- synthetic PPO global gradient：`776.2791×`

orthogonal gain 1 把三者降到：

- `1.5875×`
- `1.6523×`
- `2.3879×`

实际训练到 6.29M：

| 初始化 | Return | Success |
|---|---:|---:|
| Glorot anchor | 7.734375 | 77.34375% |
| Orthogonal √2 anchor | 5.390625 | 53.90625% |
| Orthogonal gain 1 | 6.640625 | 66.40625% |

gain 1 恢复了约 `53.33%` 的差距，是部分缓解，不是完全恢复。用户随后停止更深入
的 initializer arms，因此 depth-scale、zero-last、SkipInit 没有完整可比结果。

最终可安全陈述的是：

> 对这个无归一化、未做 residual scaling 的 IMPALA encoder，统一 orthogonal
> `sqrt(2)` 不是架构无关的安全默认值；它造成了巨大的初始激活和梯度放大。

不能把它推广成所有环境、所有网络、所有 PPO 都成立。

## 5. 用 PPO 数据重新训练 Dynamics

### 5.1 先纠正实验问题：checkpoint 独立训练 → checkpoint mixture

最初的 `CR-DYN-0006/0007` 把 1.05M / 6.29M / 12.58M / 25.17M PPO checkpoint
各自当成独立 corpus。用户及时指出真正的问题是：

> 固定总轨迹量时，世界模型应只看成熟 policy，还是混合“不会玩 → 逐渐会玩”的分布？

因此旧协议在开跑前终止并保留，改为 mixture ablation。

### 5.2 `CR-DYN-0008`：三种固定总量 mixture

固定 2,048 records、16.6M EMA Tokenizer、3.93M Dynamics、20k updates：

| Mixture | 1.05M | 6.29M | 12.58M | 25.17M | Mean PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| final-only | 0 | 0 | 0 | 2048 | 17.0731 | 0.71345 |
| uniform | 512 | 512 | 512 | 512 | 16.5679 | 0.69770 |
| recency weighted | 256 | 256 | 512 | 1024 | 17.0137 | 0.70131 |

预注册的 recency-weighted 假设被拒绝；final-only 在冻结选择指标上获胜。
这不代表 checkpoint mixture 普遍没有价值，只说明在当前固定总量和 evaluator 下，
早期弱 policy 轨迹没有改善最终 held-out future。

### 5.3 `CR-DYN-0009`：胜出 mixture 上再做 Dynamics scale

| Dynamics | 参数 | Mean PSNR | SSIM |
|---|---:|---:|---:|
| tiny | 0.15584M | 12.6694 | 0.55351 |
| small | 0.54592M | 13.3966 | 0.58091 |
| medium | 3.93139M | 17.0731 | 0.71345 |
| large | 12.90278M | 15.7237 | 0.67481 |

medium 再次胜过 large。更重要的是，用户直接玩 Demo 后判定：

- 时间运动不可信；
- 动作响应不可信；
- `17.073 dB / 0.7135` 完全不足以表示“可用”。

### 5.4 20k recipe 的三个根本缺口

事后审计找到：

1. 每条 PPO record 只有 64 帧，训练 window 也是 64 帧，所以合法 crop start 只有 0。
   `p_include_reward=0.5` 配了也完全不起作用。
2. OpenDreamer 通用 Dynamics recipe 接近 200k updates、`k_max=256`；pilot 只有
   20k、`k_max=8`，优化和 diffusion schedule 都被大幅压缩。
3. 选择指标没有要求 aligned actions 必须优于 shuffled、shifted 或 all-noop，
   所以高 PSNR 的模型可能主要依赖视觉惯性而忽略 action。

还要注意评估协议改变会改变绝对分数。旧 medium 在新鲜固定 futures 上重评为
`18.4025 dB / 0.7200`，不等于模型突然变好；这说明不同 held-out future 的
绝对 PSNR 不能直接横向比较。

## 6. Dynamics reference repair：在线 Tokenizer 为什么让训练变慢

### 6.1 `CR-DYN-0010`：参考式修复 bundle

修复协议改为：

- 4,096 train + 512 held-out；
- 每条 160 帧；
- 64/128-frame schedule；
- 200k optimizer updates；
- `k_max=256`；
- 100k 开始 bootstrap；
- aligned / shuffled / shifted / all-noop action controls；
- 模型仍固定 3.93M，Tokenizer 仍固定 16.6M EMA。

新数据的合法 crop：

- 64-frame window：97 个 start；
- 128-frame window：33 个 start。

所有 15 个 Procgen actions 都出现，action 4 明确为 no-op。

旧 checkpoint 在同一固定 future 上的 action control：

- aligned vs shuffled：horizon-16 `+2.7765 dB`
- aligned vs shifted：`+0.7872 dB`
- aligned vs all-noop：`+2.6401 dB`

这证明旧模型的数值预测并非完全不依赖动作，但仍不推翻用户对可见动作响应的拒绝。

新模型训练到 10k 时，online shortcut：

- horizon 1：`26.07 dB`
- horizon 3：`22.41 dB`
- horizon 8：`19.69 dB`

但预计完整 200k 还要约 13 小时。原因不是 Dynamics 本身特别大，而是每个
optimizer update 都把选中的 RGB 帧重新送进冻结的 16.6M Tokenizer encoder。
同一帧会被反复编码成相同 latent，浪费绝大多数计算。

因此在 11,320 updates 人工停止，保留 10k video 和全部日志。它是
`aborted`，不是模型失败。

### 6.2 `CR-DYN-0011`：离线预编码 latent

新协议只改变输入表示的计算位置：

```text
之前：每个 training update 重新 RGB → frozen Tokenizer → latent
现在：每条 160-frame record 一次性编码并审计 → training 直接读取 latent
```

严格保持不变：

- 原始 train/eval tree；
- record 数量和顺序；
- actions / rewards / terminals；
- reward-biased crop；
- latent normalization；
- 3.93M Dynamics；
- batch 16；
- 64/128 schedule；
- 200k 和 `k_max=256`；
- 最终仍回到 raw RGB 做视觉评估。

4,096 + 512 条 latent record 的完整 pair audit 已通过，latent shape 为
`160 × 16 × 16`。

这一阶段还有一个零步 recorder bug：预处理步骤拥有的是 immutable
`metadata.json`，通用 training recorder 却要求 `runtime-identity.json` 和
`run-state.json`。这是生命周期归属错误，不是数据或模型错误。精确红测试确认后，
将离线预处理与训练 recorder 分离。

首次 attempt 在 metric update 62,601 后被外部关机 timer 中断；显式授权的恢复
从 step 50k 继续，并把 replay 的 50,001–62,601 保留为 deviation。恢复最终完成
`200,000` updates 和 step 199,999 checkpoint。

- 1k–10k recorder median throughput：raw RGB `4.0157` updates/s，offline latent
  `6.5986` updates/s，即 `1.6432x`；
- shortcut：`24.065977 dB / 0.833979 SSIM`；
- 256-step full diffusion：`24.053035 dB / 0.846804 SSIM`；
- horizon-16 aligned 相对 shuffled / all-noop：`+7.806056 / +6.480551 dB`；
- 数值门全部通过，终态 claim 仍为 `awaiting_visual_review`。

## 7. Live Demo：模型 bug、运行时 bug和验证器 bug要分开

### 7.1 `CR-DEMO-0001`：最早的阻塞式 Demo

先后遇到：

| 症状 | 根因 | 修复 |
|---|---|---|
| 启动时报 iterable 不是 iterator | `ThreadPrefetchIterDataset` 未包 `iter()` | 只修 Demo loader |
| 第一步成功、第二步报 JAX dtype 错 | generated latent 是 float32，写回 bfloat16 context 后提升了缓存 dtype | 写回前保持 context dtype |
| API 能跑但动作不可信 | Procgen 实际 `D15[]`，config 写 16；通用 `shift_actions` 用 `dim // 2 = 8`，但 CoinRun no-op 是 4 | 后续修正 action contract 并重训 |

修复后 cached step 约 0.67 秒，但用户仍判定不可用。

### 7.2 `CR-DEMO-0002`：技术闭环不等于模型通过

PPO-mixture medium checkpoint 能：

- remote `/health`；
- remote step 1；
- SSH tunnel local step 2；
- cached latency约 `764.5 ms`。

中间还发生：

- SSH port-forward 多次 timeout；
- DSW shutdown timer 在本地验证前停机；
- 重启后持久化的 dead PID 阻止新进程；
- 保留 stale attempt 后才安全恢复。

最终 runtime closure 成功，但用户再次拒绝 temporal coherence 和 action response。

### 7.3 `CR-DEMO-0003`：真正的连续世界运行时

旧 Demo 每生成一帧都：

1. 浏览器发一个阻塞 HTTP 请求；
2. 服务端重新建立/refill Dynamics context；
3. 请求 pending 时 release event 可能丢失；
4. 不按键时世界不前进。

修复后：

- SSE 连续流；
- context 只 prefill 一次；
- persistent Dynamics 和 decoder KV cache；
- 服务端在有 subscriber 时持续推进；
- 输入是“当前所有 held keys”的最新完整状态；
- release 恢复 no-op；
- pause/reset 有明确状态；
- 1440×900 下舞台和侧栏完整可见。

稳定生成约 `22–25 ms/frame`。这是 runtime 的正确性和性能，不是模型质量修复。

### 7.4 新 checkpoint 首次启动的三个运维 bug

把 `CR-DYN-0011` 的 25k checkpoint 接进 Demo 时又发现：

1. **Checkpoint path 语义**

   loader 需要 `checkpoints/` 根目录并自行选 latest，不能直接传
   `checkpoints/25000`。直接传子目录会 `FileNotFoundError`。

2. **Editable install 污染 source identity**

   venv 的 editable install 指向 `/mnt/workspace/open-dreamer`，不是 continuous-demo
   worktree，导致加载到旧版 `shift_actions`，报
   `shift_actions() takes 2 positional arguments but 3 were given`。
   启动时必须显式设置执行 worktree 的 `PYTHONPATH`。

3. **JAX 显存策略**

   Demo 与训练并行时必须关闭预分配并限制 memory fraction。当前 Demo 约 0.5 GiB，
   没有抢占正在训练的 Dynamics。

### 7.5 “一点反应都没有”的真正根因：revision 时钟单位

连续 Demo 用全局单调 `input_revision` 拒绝乱序输入：

```python
if revision > self.input_revision:
    self.input_revision = revision
    self.input_action = action_id
```

浏览器发送微秒尺度：

```text
Date.now() * 1000 + counter
```

独立验证脚本却发送：

```text
time.time_ns()
```

纳秒数比浏览器微秒数大约 1000 倍。验证器一旦写入服务端全局状态，之后浏览器输入虽然
HTTP 200，`revision > input_revision` 永远为假，所以被静默忽略。页面看起来就像
Dynamics 对任何动作都没反应。

修复是：

- 验证器改成 `time.time_ns() // 1000`，与浏览器同为微秒；
- 重启 Demo，清空已污染的全局 revision；
- 重新验证 no-op `302/303/304`、right `307/308`、release `311`；
- 用户随后确认交互已经工作。

这次故障的本质是**测试工具污染在线状态 + 服务端静默拒绝**，不是模型没有学到动作。

后续应把以下不变量固化到代码和测试：

- revision 必须使用声明过的统一单位或 session-local sequence number；
- `/api/input` 必须返回 `accepted` 和 server-side current revision；
- verifier 不得在共享 Demo 上写入永久压制真实客户端的状态；
- reset 或新 session 应隔离输入时钟；
- E2E 必须检查观测 action 随 press/release 改变，不能只检查 HTTP 200。

## 8. 根因矩阵

| 表面症状 | 所属层 | 真正根因 | 处理 |
|---|---|---|---|
| Tokenizer 模糊 | 表示 | 0.17M 容量不足 | 固定 20k 做完整 scale，固定 16.6M |
| 最大 Tokenizer 看似训练不足 | 实验设计 | unequal-step scaling 把容量与优化混杂 | 五档无条件固定 20k |
| Dynamics 大模型反而差 | 优化/数据 | 20k、小数据、large 欠优化 | 不做普遍 scale claim |
| 随机数据 world model 很差 | 数据分布 | 不覆盖有意义跑跳状态 | 训练真实 CoinRun PPO 采集 |
| PPO reward 低 | 训练 recipe | 多设置未对齐；核心单因素是 orthogonal √2 放大 | 先 parity，再单因素和机制 probe |
| PPO evaluator 前后差很多 | 评估 | deterministic 固定 levels 有偏 | 统一 stochastic full-distribution evaluator |
| mixture 没提升 | 科学结果 | 当前固定总量下 early-policy 数据降低 final-future 质量 | 选择 final-only，不伪造正结果 |
| 17 dB 但 Demo 不可用 | 指标 | 平均像素质量不测长期连续性和可控性 | 加 action corruption + 人看视频 |
| reward-biased crop 没效果 | 数据协议 | 64-frame record 与 64-frame window 只有 start 0 | 改 160-frame record |
| 200k 预计太慢 | 算力路径 | 每步重复运行 frozen Tokenizer | 一次性离线 latent |
| 第二步 JAX dtype 报错 | Runtime state | float32 latent 污染 bfloat16 context | 写回保持 context dtype |
| 不按键世界不动 | Runtime architecture | blocking per-frame RPC | continuous SSE + persistent cache |
| 新 Demo 加载错误代码 | Runtime identity | editable install 指向旧 worktree | 显式 `PYTHONPATH` + identity preflight |
| 浏览器按键完全没反应 | 验证工具/协议 | ns verifier revision 永久压过 µs browser revision | 统一单位、重启清状态、返回 accepted |

## 9. 哪些结论已经成立

### 已建立

- Tokenizer 在固定 20k 下随容量从 0.17M 到 16.6M 单调改善；
- “28.7M”标签继续数值改善，但边际收益很小；
- 直接随机动作 Dynamics 只能用于 smoke；
- 20k Dynamics 在当前数据上以 medium 最好，large 未继续提升；
- context 16 在固定 future 上优于 4，32 没有继续提升；
- PPO parity recipe 能恢复到公开 CoinRun 曲线区间；
- 当前无归一化 IMPALA encoder 上，orthogonal √2 引起巨大初始放大；
- PPO final-only 轨迹在当前 2,048-record mixture 比 uniform/recency 更好；
- 64-frame corpus 让 reward-biased crop 失效；
- 离线 latent 完整保留 actions/rewards/terminals/order；
- 离线 latent 把同协议 1k–10k median throughput 提高到 1.6432x；
- `CR-DYN-0011` 完成 200k，并通过 shortcut、full diffusion 和 action controls 的数值门；
- continuous Demo runtime、cache、press/release 和 step-100k preview 输入链已工作。

### 尚未建立

- `CR-DYN-0011` 200k 终态画面是否达到可用视觉质量；
- 200k 画面是否长期不漂移；
- 玩家是否认为动作改变未来的方向、跳跃和碰撞足够可信；
- 结果是否跨 seed、跨 level 或跨游戏成立；
- 任何“可以发 paper”的通用初始化结论。

## 10. 当前下一步和停止条件

当前顺序已经冻结：

1. 先用 NanoDreamer 分文件复现正式 Tokenizer、PPO、数据/latent audits 和
   `CR-DYN-0011` Medium；
2. 每段使用同一 held-out evaluator 和预注册 tolerance，不用 training metric 代替；
3. Nano continuous Demo 必须通过真实 held-input/state progression，并由真实
   Nano checkpoint 生成 README GIF；
4. 只有完整 Nano 链通过，才恢复 `CR-DYN-0012` XLarge；
5. 继续把指标通过与用户视觉验收分开，不同时改变数据、context、sampler 和 scale。

最终 Demo 验收必须同时满足：

- **表示**：角色、平台、障碍和金币可辨认；
- **时间**：静止、移动、跳跃过程中不闪烁或无因漂移；
- **控制**：同一历史下 right/jump/noop 产生方向正确且可重复的差异；
- **运行时**：持续推进、press/release/reset 正确，验证器不污染状态；
- **性能**：首次 JIT 单独报告，steady-state 单独报告。

## 11. 实验基建还需要收敛的一处

PPO 的最终单因素和初始化机制记录位于
[PR #2](https://github.com/xilanhua12138/open-dreamer/pull/2)，Dynamics 与 Demo
记录位于 [PR #3](https://github.com/xilanhua12138/open-dreamer/pull/3)。
PR #3 当前携带的是 PPO 早期账本快照，因此只读一个分支会误判 `CR-PPO-0003`
仍在运行，并看不到 `CR-PPO-0005` 的终态。

这不是科学结果问题，而是账本真源分叉。两个 PR 评审完成后，应让实验账本在主分支
重新汇合，并增加：

- 跨分支 experiment index 校验；
- 每个依赖实验记录 source branch / commit / PR；
- retrospective 自动检查“依赖状态是否比本分支快照更新”；
- execution worktree 与 ledger-generated files 分离，避免自动证据写入让执行 source
  看起来 dirty。

## 证据入口

- Tokenizer：[CR-TOK-0001](../experiments/CR-TOK-0001/README.md)、
  [CR-TOK-0002](../experiments/CR-TOK-0002/README.md)、
  [CR-TOK-0003](../experiments/CR-TOK-0003/README.md)、
  [CR-TOK-0004](../experiments/CR-TOK-0004/README.md)
- 早期 Dynamics：[CR-DYN-0001](../experiments/CR-DYN-0001/README.md) 至
  [CR-DYN-0005](../experiments/CR-DYN-0005/README.md)
- PPO：[CR-PPO-0001](../experiments/CR-PPO-0001/README.md)、
  [CR-PPO-0002](../experiments/CR-PPO-0002/README.md)、
  [CR-PPO-0003 final](https://github.com/xilanhua12138/open-dreamer/blob/feat/coinrun-ppo-collector/experiments/CR-PPO-0003/README.md)、
  [CR-PPO-0005 final](https://github.com/xilanhua12138/open-dreamer/blob/feat/coinrun-ppo-collector/experiments/CR-PPO-0005/README.md)
- PPO mixture 与 scale：[CR-DYN-0008](../experiments/CR-DYN-0008/README.md)、
  [CR-DYN-0009](../experiments/CR-DYN-0009/README.md)
- Reference repair：[CR-DYN-0010](../experiments/CR-DYN-0010/README.md)、
  [CR-DYN-0011](../experiments/CR-DYN-0011/README.md)
- Demo：[CR-DEMO-0001](../experiments/CR-DEMO-0001/README.md)、
  [CR-DEMO-0002](../experiments/CR-DEMO-0002/README.md)、
  [CR-DEMO-0003](../experiments/CR-DEMO-0003/README.md)
- PPO 初始化技术笔记：
  [Orthogonal √2 is not an architecture-independent default](https://github.com/xilanhua12138/open-dreamer/blob/feat/coinrun-ppo-collector/docs/ppo-orthogonal-initialization-note.md)
