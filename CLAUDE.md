# OpenDreamer 实验治理规范

本文件适用于整个仓库。任何人或 agent 只要计划、启动、监控、评估、解释或复用实验，都必须先完整阅读本文件和 `experiments/README.md`。

目标不是“保存几个数字”，而是让任意后来者能够回答：

1. 当时为什么做这次实验？
2. 它试图证伪什么假设？
3. 与哪个 baseline 比，唯一主动改变的变量是什么？
4. 实际运行的代码、数据、参数和算力是什么？
5. 哪些结果是原始观测，哪些是解释，哪些结论仍不能声称？
6. 如果结果异常，异常来自模型、实验设计、数据、算力还是执行中断？
7. 下一次实验为什么由这次结果推出？

## 1. 实验风格

本仓采用受 Kaiming He 论文实验结构启发的纪律，而不是声称复刻某个研究组的内部流程：

- 先写机制假设和可证伪条件，再开 GPU。
- 先建立强且简单的 baseline，再做规模扩张。
- 一次只主动改变一个主要变量；多变量一起变化时，必须明确标记为探索性实验。
- 所有比较必须写出“保持不变”的条件。
- 负结果、异常结果和被中断的实验不能删除。
- 先用小实验定位机制，再用规模实验验证趋势。
- 结论强度不能超过证据强度。

参考的公开实验范式：

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)：围绕一个机制假设组织一系列受控消融。
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)：保持核心设计简洁，以同一主干上的系统消融支持设计选择。

## 2. 开始实验前：必须预注册

每次实验先从 `experiments/_template/` 复制一个目录，分配不可复用的 ID：

- `CR-TOK-NNNN`：CoinRun tokenizer。
- `CR-DYN-NNNN`：CoinRun dynamics/world model。
- `CR-DEMO-NNNN`：交互或推理 demo。
- 其他数据集使用清晰的新前缀。

必须先写完 `manifest.json`，至少包括：

- `question`：实验只回答一个核心问题。
- `reason`：为什么现在要做；它由哪个先前结果或缺口触发。
- `hypothesis`：预期会发生什么以及原因。
- `falsification`：什么结果会使假设不成立。
- `baseline_id`：受控比较的基准；没有基准时写明原因。
- `controlled_variable`：唯一主动改变的变量。
- `constants`：数据、seed、训练预算、评估 future、采样器等保持不变的量。
- `source`：仓库 URL、commit、branch、dirty 状态和 runner 路径。
- `data`：数据来源、split、样本数、帧数、动作策略和可用哈希。
- `compute`：GPU 型号/数量/显存、区域、预计预算和关机策略。
- `protocol`：完整模型与训练/评估参数。
- `success_criteria`：在看到结果前写下的判定规则。
- `planned_outputs`：checkpoint、日志、metrics、视频和 manifest 的预期位置。

禁止先看结果再补写假设。历史回填必须设置 `retrospective: true`，明确它不是预注册记录。

## 3. 运行时：源代码与状态必须可定位

### 3.1 代码

- 正式实验必须引用一个 commit，而不是只引用 branch 名。
- 如果运行时工作树为 dirty，记录 `git diff --binary` 的 SHA256，并保存 patch；不能只写“有本地修改”。
- 一次性 runner 也必须进入本仓库。远程临时脚本不是长期真相源。
- 不允许在同一个实验 ID 下悄悄改代码或参数后重跑。改变实验含义时新建 ID；纯故障恢复可保留 ID，但必须追加 deviation/event。

### 3.2 状态

`results.json.execution_status` 只能取：

- `planned`
- `queued`
- `running`
- `completed`
- `failed`
- `aborted`
- `blocked`
- `invalid`

进程退出码和科学结论必须分开记录：

- `execution_status=completed` 只表示预定 run 全部执行完成。
- `scientific_status` 才表示 `supports_hypothesis`、`rejects_hypothesis`、`inconclusive` 或 `not_evaluated`。
- wrapper 写出 `COMPLETE` 但某个比较臂缺失时，整体实验不得标成 `completed`。

所有阶段追加写入 `events.jsonl`。至少应有：

- `planned`
- `started`
- 每个训练/评估阶段的 `stage`
- 故障恢复时的 `deviation`
- 最终 `completed`、`failed`、`aborted` 或 `blocked`

事件只追加，不重写历史。纠错用新的 `correction` 事件，并在 `results.json.revision_history` 说明。

### 3.3 远程算力

- 不创建未获授权的新收费实例，不擅自换 GPU 规格。
- 启动 GPU 前先确认数据、代码、checkpoint 和 runner 已就绪。
- 每个收费实例必须有自动关机策略，并记录创建时间和预计关机时间。
- 库存不足是 `blocked`，不是模型实验 `failed`。
- 账号余额、AccessKey、SSH 私钥、代理订阅和 token 永远不能进 Git。

### 3.4 运行身份和可观测性

- 正式训练必须使用仓库 logger 生成稳定 `run-id.json`、每次进程尝试的
  `runtime-identities/*.json`、原子 `run-state.json`、`metrics.jsonl`、
  `telemetry.jsonl` 和 `artifacts.jsonl`。
- 新 runner 必须通过 `scripts/experiments/run_recorded.py` 启动训练，确保成功和
  失败退出都会自动物化账本 evidence；成功但没有结构化 runtime evidence
  必须视为 runner 错误。
- `runtime-identity.json` 必须记录 source/dirty patch、依赖文件 hash、
  Python/JAX/CUDA/GPU 身份和脱敏后的启动命令。禁止记录代理 URL、API key、
  token 或完整环境变量。
- PID 存活不是训练健康证据。监控必须同时检查结构化 state、最近 completed
  update、最近 telemetry 时间、GPU/磁盘状态和预期产物。
- W&B 是可选的在线镜像，不是唯一真相源。未获得非空在线 run URL 时不得声称
  已上传；W&B 不可用也不能导致本地 metrics、telemetry 或 media 丢失。
- 完整字段、认证方式和恢复语义见 `experiments/OBSERVABILITY.md`。

## 4. 评估：比较对象必须一致

### 4.1 Tokenizer

必须区分：

- training batch PSNR
- held-out clean reconstruction PSNR
- held-out masked reconstruction PSNR
- online 权重与 EMA 权重

它们不是同一个指标，不能互换。官方图中估读的数字必须标记为 `plot_estimate`，不能作为精确 ground truth。

正式 tokenizer run 若启用周期验证，必须使用与训练 split 分离的固定 held-out
dataset、seed、batch/clip shape 和内容 SHA256。验证图片和视频必须来自该固定
validation set，不得用当前 training batch 冒充验证集。每个 milestone 的本地
JSON/PNG/MP4 先落盘并记录 hash，再镜像到 W&B。

### 4.2 Dynamics

默认报告：

- `mean_frame_psnr_db`
- `mean_video_psnr_db`
- `mean_ssim`
- `psnr_by_horizon_db`，至少 `@1/@3/@8/@16`
- 每条 held-out trajectory 的原始分数

必须写清 metric 的 target：

- `tokenizer_decoded_ground_truth`：隔离 dynamics 误差。
- `original_rgb`：同时包含 tokenizer 和 dynamics 误差。

两类 target 的数字不得放在同一列直接比较。

### 4.3 Context 消融

`4/16/32` 历史帧比较必须固定：

- 同一个 checkpoint。
- 同一组 episode 和起始位置。
- 同一段未来帧与未来动作。
- 同一个 seed/PRNG key 或可证明等价的 deterministic sampling。
- 同一采样器、denoise 步数和 horizon。

如果 context 改变导致 future 也平移，则该实验无效，标记 `execution_status=invalid`。

## 5. 证据和产物

每个实验目录固定包含：

```text
experiments/<ID>/
├── README.md       # 面向人的问题、设计、结果、解释
├── manifest.json   # 运行前设计与全部参数
├── results.json    # 运行后观测、偏差、结论与决策
├── events.jsonl    # 只追加的生命周期记录
└── raw/            # 足够小的原始 metrics/config；禁止 checkpoint 和大视频
```

产物策略：

- JSON、Hydra override、关键日志摘录和小型配置直接入 Git。
- checkpoint、完整日志、数据集、MP4 和 tarball 留在对象存储/DSW/本地归档。
- 所有不入 Git 的物料记录 URI、字节数（可得时）和 SHA256。
- 远程绝对路径只是位置，不是完整证据；路径消失后，哈希仍应能确认恢复出的文件。
- 不要提交生成的视频、checkpoint、凭证或完整代理配置。

## 6. 结论分级

`results.json.claim_level` 只能使用：

- `smoke_test`：只证明代码路径能运行。
- `pipeline_closure`：证明数据→训练→评估/rollout 的端到端链路闭环。
- `internal_result`：在本仓自定义协议下得到可复查结果。
- `partial_reproduction`：与公开实验有明确重合，但数据、recipe、规模或指标仍有差异。
- `strict_reproduction`：公开 recipe、数据、模型规模、训练预算和评估协议均匹配。

“生成了 MP4”“训练完成”“PSNR 看起来不错”均不能单独支持 `strict_reproduction`。

README 的结论必须分成：

- `Observed`：文件或日志直接支持的事实。
- `Interpretation`：基于事实的推断。
- `Not established`：本实验仍未证明的主张。
- `Decision`：下一步做什么以及为什么。

## 7. 结束实验后的强制检查

完成或终止一次实验时：

1. 更新 `results.json` 和 `events.jsonl`。
2. 保留所有 deviations、失败原因和缺失比较臂。
3. 将足够小的原始指标复制到 `raw/`，并校验哈希。
   对支持结构化 recorder 的 run，优先使用
   `scripts/experiments/run_recorded.py` 自动生成 run evidence、README 表格和
   results artifact 引用；仅在导入历史 run 时手动调用
   `materialize_run_evidence.py`。禁止由 agent 在多个文件中重复手抄数字。
4. 更新 `experiments/index.json` 和 `experiments/README.md`。
5. 运行：

```bash
uv run --no-project python scripts/validate_experiments.py
```

6. 通过校验后再提交。

不允许只在聊天、notebook、临时目录或远程 `logs/` 中留下最终结论。

## 8. 修改已有记录

- 不删除失败实验。
- 不覆盖原始 metrics。
- 数字纠错必须引用新的证据，增加 `revision_history` 和 `correction` event。
- 重新解释旧结果可以修改 README，但必须保留原解释及修改原因。
- 任何影响比较公平性的变化都新建实验 ID。

## 9. 当前实验入口

当前索引见 `experiments/README.md` 和 `experiments/index.json`。2026-07-28 的 DSW 运行与库存阻塞记录见 `operations/2026-07-28-pai-dsw-a10.md`。
