# Orthogonal √2 不是架构无关的默认值

这是一份基于 `CR-PPO-0003` 和 `CR-PPO-0005` 的内部技术笔记。结论很窄：

> 在当前没有 normalization、没有 residual scaling 的 IMPALA-style CoinRun
> encoder 中，把 orthogonal gain `sqrt(2)` 无差别用于 stem、六个 residual
> block 和后续 dense，会在初始化时显著放大表示、局部 Jacobian 和 PPO
> 梯度；将 gain 改为 `1` 能部分恢复策略质量。

它不是“orthogonal initialization 普遍有害”的结论，也不是多环境、
多 seed 的论文级结果。

## 1. 我们最初看到什么

`CR-PPO-0003` 保持训练预算、数据分布、PPO 超参数和最终 evaluator 不变，
只回退一个历史设置。其他三项单因素变化没有造成明显退化，只有 backbone
初始化从 Glorot 改为 orthogonal `sqrt(2)` 后，最终 256 局独立评估明显下降：

| 配置 | Mean return | Success |
|---|---:|---:|
| Glorot reference | 7.734375 | 77.34375% |
| Orthogonal √2 | 5.390625 | 53.90625% |
| 差值 | -2.343750 | -23.43750 pp |

这只能定位“哪个开关与退化共现”，还没有解释机制。于是
`CR-PPO-0005` 在训练前固定 32 个 CoinRun observations，并对 16 个参数 seed
测量信号、JVP 和 synthetic PPO gradient。

## 2. 初始化探针给出的机制证据

以下数字都是 16 个参数 seed 的均值；六个配置使用同一个 observation
SHA256 `fd36a901...04feb`：

| 初始化 | Encoder RMS | Encoder JVP RMS gain | PPO global grad L2 | 最后一层 branch / skip |
|---|---:|---:|---:|---:|
| Glorot | 0.4960 | 0.6493 | 4.6084 | 0.5380 |
| Orthogonal √2 | 13.0191 | 17.8623 | 3577.4231 | 1.0570 |
| Orthogonal gain 1 | 0.7874 | 1.0729 | 11.0045 | 0.5300 |
| Orthogonal √2 + `1/sqrt(6)` branch scale | 2.5210 | 3.4602 | 77.5217 | 0.4308 |
| Orthogonal √2 + zero-last | 1.4561 | 2.1225 | 40.1874 | 0 |
| Orthogonal √2 + SkipInit | 1.4561 | 2.1225 | 33.6389 | 0 |

相对 Glorot，未缓解的 orthogonal √2 把：

- encoder RMS 放大到 `26.2468×`；
- encoder JVP RMS gain 放大到 `27.5101×`；
- synthetic PPO global gradient L2 放大到 `776.2791×`。

只把 orthogonal gain 从 `sqrt(2)` 改为 `1`，三个比值就降到
`1.5875×`、`1.6523×` 和 `2.3879×`。这说明在已测试的两个因素中，主要问题
不是“矩阵正交”，而是把适合单条 ReLU 路径的 `sqrt(2)` gain 直接复用到
带 identity addition 的未归一化残差栈。

完整原始探针见
[initialization-probe.json](../experiments/CR-PPO-0005/raw/probe/initialization-probe.json)。

## 3. 为什么 residual connection 会改变初始化问题

普通前馈层只有一条主路径。He-style `sqrt(2)` gain 的直觉是补偿 ReLU
截断，使单条路径的方差不要快速衰减。

残差块却计算：

```text
x_(l+1) = x_l + F_l(x_l)
```

如果 `F_l(x_l)` 初始化时已经和 skip 同量级，那么每个 block 都不是“接近
identity 的小修正”，而是在相加两条强信号。粗略忽略相关性时：

```text
Var[x_(l+1)] ≈ Var[x_l] + Var[F_l(x_l)]
```

六个 block 连续叠加后，即使每条 residual branch 单独看并没有数值溢出，
整网表示和 Jacobian 也可能被逐层放大。PPO 又同时依赖 policy、value、
advantage normalization 和 clipped updates，过大的共享 backbone 梯度会改变
早期优化几何，最后表现为样本效率下降。

这也是为什么 residual 网络的初始化通常需要考虑“接近 identity”：

- [Fixup](https://arxiv.org/abs/1901.09321) 按深度缩放 residual branch，并让
  某些 residual 层从零开始；
- [SkipInit / BatchNorm biases residual blocks toward identity](https://arxiv.org/abs/2002.10444)
  讨论了显式的可学习 residual multiplier；
- [RISOTTO](https://arxiv.org/abs/2210.02411) 从 dynamical isometry 角度设计
  residual 初始化。

它们的共同点不是“禁止 orthogonal”，而是初始化必须和拓扑、深度、
normalization 与 residual 参数化一起设计。

## 4. 一个已经验证的缓解

`orthogonal_gain1` 从头训练了精确 `6,291,456` transitions，并使用和历史锚点
相同的 stochastic、full-distribution、seed-4242、256-episode evaluator：

| 配置 | Mean return | Success |
|---|---:|---:|
| Orthogonal √2 degraded anchor | 5.390625 | 53.90625% |
| Orthogonal gain 1 | 6.640625 | 66.40625% |
| Glorot reference | 7.734375 | 77.34375% |

预注册的“恢复一半差距”阈值是 mean `6.5625` 且 success `65.625%`。
gain 1 两项都通过，恰好恢复了两个差距的 `53.3333%`。

因此当前最可靠的工程结论是：

1. 不要把 orthogonal `sqrt(2)` 当成残差 encoder 的无条件默认值；
2. 如果希望保留 orthogonal，可先用 gain `1`，并在零步检查表示 RMS、JVP
   和 global gradient；
3. gain `1` 是部分恢复，不是 Glorot parity。

最终评估原件见
[final-metrics.json](../experiments/CR-PPO-0005/raw/arms/orthogonal_gain1/final-metrics.json)。

## 5. 哪些事情还没有证明

用户在机制和第一个有效缓解出现后停止了后续实验：

- depth-scaled arm 停在 `302/384` updates，没有可比较的最终评估；
- zero-last 没有启动；
- SkipInit 没有启动。

所以不能比较这三者的最终 policy，也不能声称 gain 1 是“最佳方案”。此外，
当前只有一个训练 seed 和一个环境。要形成更强结论，至少还需要：

- 多训练 seed；
- 不同 residual encoder 深度和 normalization 方案；
- Atari、其他 Procgen games 或连续控制环境；
- 把 representation/JVP/gradient 探针与最终 sample efficiency 做跨任务相关分析。

RL 实现细节本身就可能产生很大影响；相关背景可参考
[Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729)
和
[What Matters in On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990)。

## 6. 实用检查清单

在给 RL residual encoder 换初始化时，先做这些低成本检查：

1. 固定同一批 observation 和多个参数 seed。
2. 逐 block 记录 skip RMS、branch RMS、output RMS。
3. 测 encoder output RMS 和局部 JVP gain。
4. 用同一 synthetic PPO batch 测 backbone/global gradient norm。
5. 再做单变量、同预算、同 evaluator 的短训练消融。
6. 不用训练 rollout return 代替独立最终评估。

对应实验账本：

- [CR-PPO-0003](../experiments/CR-PPO-0003/README.md)：单因素定位；
- [CR-PPO-0005](../experiments/CR-PPO-0005/README.md)：机制探针与 gain-1 缓解；
- [Draft PR #2](https://github.com/xilanhua12138/open-dreamer/pull/2)：实现和原始证据。
