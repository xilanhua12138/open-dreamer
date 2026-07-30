# OpenDreamer 最小 CoinRun 世界模型实验复核

## 定位

这组实验训练的是 action-conditioned latent dynamics，属于世界模型本体，不是 tokenizer-only 实验。

官方没有发布 CoinRun dynamics 的最小配置、checkpoint 或指标。官方仓库默认 dynamics 配置面向 Minecraft：30 层、`d_model=1920`、20 万步。CoinRun tokenizer scaling 图中的 0.17M～28.7M 参数点不能当作世界模型参数点。

## 已完成的端到端链路

1. 用 Procgen CoinRun 采集带动作的视频轨迹。
2. 训练视频 tokenizer。
3. 在线把视频编码成 latent 并计算 latent 统计量。
4. 训练以历史 latent 和动作作为条件的 dynamics。
5. 给 4 帧历史和后续动作，autoregressive rollout 16 帧未来。
6. 用 tokenizer decoder 把预测 latent 还原成 64×64 视频。

## 数据

- 训练集：2,048 条 × 64 帧，共 131,072 帧
- 留出集：256 条 × 64 帧，共 16,384 帧
- 动作：Procgen 的 16 类离散动作
- 数据策略：随机动作

## 最小 dynamics 配置

- 参数量：545,920
- Transformer：2 层
- `d_model=128`
- `n_heads=2`
- `n_kv_heads=1`
- `n_register=16`
- `packing_factor=2`
- `context_length=64`
- `k_max=8`
- 训练步数：15,000
- Shortcut bootstrap：第 5,000 步后启用，比例 0.25
- 优化器：Muon
- 单张 A10 训练耗时：约 11 分钟

## 结果

最终训练时的 60 帧 rollout：

| 采样方式 | PSNR@1 | PSNR@3 | PSNR@8 |
|---|---:|---:|---:|
| Online diffusion | 19.86 dB | 18.07 dB | 16.84 dB |
| EMA diffusion | 19.93 dB | 18.26 dB | 16.42 dB |
| Online shortcut | 18.68 dB | 16.88 dB | 15.52 dB |
| EMA shortcut | 18.68 dB | 17.54 dB | 16.07 dB |

另取 4 条留出轨迹、4 帧上下文后 rollout 16 帧，将预测与 tokenizer 解码的真值比较：

- 平均 dynamics-only PSNR：16.6342 dB
- 平均 SSIM：0.6513

单条轨迹的 16 帧 PSNR 分别为 19.06、16.75、17.28、13.45 dB。

## 结论

这条世界模型链已经完整跑通：模型能从历史 latent 和动作生成多帧未来，场景的洞穴、平台、地面和整体移动具有连续性。但它还不是“好玩”的 CoinRun sandbox：

- 随 rollout 变长会出现模糊和结构漂移。
- 小角色和障碍物细节容易消失。
- 随机动作数据中有效跑跳行为太少，动作控制性不足。
- 官方也报告随机动作训练出的 CoinRun 体验很差，后来改用 RL agent 收集轨迹，再训练 dynamics 和 behaviour-cloning policy；相关 RL/BC 代码没有发布。

因此当前结论是“最小世界模型训练和 rollout 闭环成功”，不是“官方世界模型质量复现成功”，也不是“可交互 agent 已训练成功”。

## 产物

- `dynamics/.hydra/config.yaml`：完整 dynamics 配置
- `test_videos/ema_shortcut/pred_*.mp4`：模型 rollout
- `test_videos/ema_shortcut/gt_decoded_*.mp4`：tokenizer 解码真值
- `test_videos/ema_shortcut/original_*.mp4`：原始视频
- `video_sheets/comparison_*.png`：逐帧对比图
- `pipeline.log`：完整训练日志和最终 rollout 指标
