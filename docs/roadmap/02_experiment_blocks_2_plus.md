# 后续实验块（E2–E5）规划

日期：2026-11-15｜前置：E1 完成｜性质：**方向性规划，可按实际情况增补，不预先过度设计**

> 编号规则：`D` 是施工批次，`E` 是实验块。E1 见
> [01_experiment_block_1_benchmark.md](01_experiment_block_1_benchmark.md)。
> 每一块结束时必须回来更新本文件的状态与下一块范围。

## 总览

| 块 | 主题 | 核心问题 | 依赖 |
| --- | --- | --- | --- |
| E2 | 特征与模型扩展 | 换特征、换模型时，数据层是否真的不用动？ | E1（固定 prepared/split） |
| E3 | 统计、多分类与跨库基准 | 结论稳不稳？能不能对外比较？ | E1 的 run 包 + E2 的模型 |
| E4 | 任务与协议扩展 | 朗读之外（元音/DDK/对话/事件）怎么进同一框架？ | E3 的指标与任务身份 |
| E5 | 平台化与长期维护 | 加库、加模型、跑套件能否规模化且不腐化？ | E2–E4 的经验 |

---

## E2：特征与模型扩展（"即插即用"的第一次真实检验）

### 目标

在**数据层与划分完全不变**的前提下，加入第二种特征、第二种模型，并让三组结果可直接比较。

### 为什么是现在

E1 只有 MFCC + MLP。如果不在同一批数据上验证"换特征/换模型不动上层"，所谓低耦合只是口头承诺。

### 交付

1. **log-Mel 特征配方**：`feature_type: mel`
   - 复用同一 `FrontendGeometry`（帧长/帧移已归特征层）；
   - 缓存需要支持两种形状：定长向量（MFCC 统计量）与帧序列 + 有效帧数（mel）；
   - 约定：缓存 `meta.json` 增加 `layout: vector|frames`，`index.csv` 增加 `n_frames`；
   - `feature_hash` 覆盖 mel 参数（n_mels、f_min/f_max、log 类型、norm）。
2. **模型能力声明**
   - 每个模型声明 `input_layout`（vector / frames）、`expected_feature_type`（可空=任意）、`min_samples`；
   - Runner 在训练前校验：特征布局不匹配直接报错，**不做静默转换**。
3. **两个新模型插件**
   - `mfcc + lr`：最便宜的对照，验证"同特征、不同模型"；
   - `mel + cnn`：验证"不同特征、不同输入形状"。
4. **报告**：同一份 prepared/split/view 下的三行对比（MFCC+LR、MFCC+MLP、Mel+CNN），并附 `git diff` 证明 `data/` 与 `results/` 未改。

### 关键设计决策（需要先拍板）

- 帧级缓存体积：mel 帧序列比 40 维向量大 1–2 个数量级。建议只缓存**训练/验证/测试用到的窗口**，并保留 `save_frames` 开关；不做全库预计算。
- 短音频与补零：CNN 必须使用 mask，池化只在有效帧上做；否则补零会变成"时长特征"。
- 归一化归属：逐窗标准化属于模型前端，训练集统计量必须冻结并写入 run 包；不得在特征缓存阶段用全库统计。

### 验收

- 新增 mel 与 CNN 后，`data/`、`results/` 目录 `git diff` 为空（或仅有注册表一行）；
- 三组结果在同一 split 下可比，`run.json` 的 `features` 段能区分 `mfcc_*` / `mel_*`；
- 一个"特征布局不匹配"的配置在 plan/dry-run 阶段被拒绝，而不是训练到一半崩；
- 现有 76 项测试全绿，新增 mel 缓存与 mask 池化测试。

### 风险

- CPU 训练 CNN 很慢 → 先做小样本与单 epoch 冒烟，正式矩阵另行评估；
- 帧级缓存 I/O 成为瓶颈 → 优先做 decode/view 缓存（见 E5），避免重复解码。

---

## E3：统计、多分类与跨库基准

### 目标

把 E1/E2 的 run 变成**可以对外讲的基准**：结论带不确定性、失败可见、口径统一。

### 交付

1. **多分类支持（C ≥ 2）**
   - `ClassificationEvaluator` 统一按 `class_order` 计算：accuracy、per-class precision/recall/F1、macro/weighted、OVR macro AUROC、混淆矩阵 C×C；
   - 缺类别的指标返回 `null + reason`，不进排名；
   - 二分类路径保持不变（回归测试锁定）。
2. **阈值治理**
   - 默认 0.5；若用验证集选阈值，必须单独保存 `threshold.json` 并标 `selection` 阶段；
   - 测试集禁止调阈值。
3. **统计规范**
   - 人员/录音分组 bootstrap 固定重抽样次数与 seed；
   - 多 seed + 换划分的方差分解（seed 内 vs 划分间）；
   - 每个主指标附 CI，不以单点排名。
4. **跨库汇总**
   - `benchmark_table.csv`：行 = 任务 × 模型 × seed，列 = 指标 + 成本；
   - 只允许同 `task_hash + split_hash + evaluation_unit` 直接排名；
   - 失败/阻塞任务进入覆盖率列，不消失。
5. **证据等级表**：沿用 E1 的 A/B/C 分级，C 级库不进入"可引用结论"。

### 验收

- 一个 C=3 合成任务不用改模型头与 Evaluator 即可跑通；
- 单类测试集 AUROC 返回 NA + 原因，不崩溃、不出榜；
- 从保存的预测重算指标与原始 `metrics.json` 一致（容差内）；
- 汇总表中失败任务可见，且分母注明。

### 风险

- 多分类会牵动指标契约 → 一次性改完并加回归，避免中途半套；
- 跨库比较容易被误读 → 报告层面固定"按任务分组"，不做总平均。

---

## E4：任务与协议扩展

### 目标

让"非朗读、非二分类、非独立同分布"的真实临床任务进入同一框架，而不复制训练/评测代码。

### 候选内容（按需要做）

1. **多发声任务**：持续元音、DDK、对话/访谈。同一数据集内按 `elicitation` 拆分任务；同一人跨任务仍同 `group_id`。
2. **纵向 / 人-会话**：`evaluation_id = person-session`，split 与 CI 仍按人分组；标签随会话变化时的处理规则要显式声明。
3. **事件级任务**：咳嗽/呼吸扩展轨道（Coswara、COUGHVID 等），评价单位是事件而不是病人；不与主轨道混榜。
4. **官方 split 复现**：若数据集自带 train/test，优先复现并登记来源；无官方 split 时使用本地预注册协议。
5. **多标签 / 严重度**：若引入分级标签，需要新的任务类型与指标 schema，不硬塞进二分类。

### 关键设计

- 任务快照显式化：`task_hash`、`class_order`、`positive_class`、`evaluation_unit`、样本清单一起落盘；
- `evaluation_id` 与 `group_id` 分离：前者决定指标聚合口径，后者决定防泄漏；
- 新任务类型必须新注册 evaluator，禁止在现有二分类函数里加 `if`。

### 验收

- 同一数据集的两个 elicitation 能生成两个 task，互不干扰；
- 一个纵向数据集按人分组、按会话评价，CI 仍按人 bootstrap；
- 事件级任务的分母/覆盖率与病人级任务严格分开。

### 风险

- 数据集元数据质量决定上限 → 每接一个任务先做身份审计，未确认就 `blocked`；
- 任务类型扩张会让 schema 膨胀 → 只有出现第二个同类需求时才抽象。

---

## E5：平台化与长期维护

### 目标

让"加库、加模型、跑套件、断点续跑、对外发布"成为常规操作，而不是每次手工拼命令。

### 交付

1. **Suite 与 plan**
   - `configs/suites/*.yaml`：任务 × 模型 × seed 的声明；
   - `plan`：先输出"将跑多少个 job、哪些 blocked/incompatible、预计缓存命中"，再执行；
   - 每个 job 状态：`planned / running / succeeded / failed / blocked / skipped`。
2. **缓存分层**
   - 解码缓存（内容哈希 + 解码版本）→ 视图缓存（view_hash）→ 特征缓存（feature_key）→ 拟合变换（split + 特征）；
   - 明确失效边界：改 feature 不重解码，改 view 不重解码，改 split 只重拟合。
3. **错误码与退出策略**
   - 稳定错误码（标签未确认、身份不明、视图为空、模型不兼容、预测缺行等）；
   - suite 失败不吞，退出码与汇总一致。
4. **网页结果页**
   - leaderboard、run 详情、多 seed 箱线图、覆盖率与失败清单；
   - 网页数字与 `results_summary` 同源。
5. **回归与发布**
   - 小型合成库冒烟（每次改代码都跑），锁定结果契约；
   - `results_summary` 白名单归档 + git tag；
   - 实验登记表 `docs/experiments/registry.csv` 自动追加。

### 验收

- 一条命令跑完一套 suite，中断可续，失败可见；
- 同一实验重复执行结果一致（固定 seed、单线程）；
- 新增一个库/模型时，不修改 `results/contract.py` 既有字段；
- 网页能回答"这个数字来自哪个 run、哪版配置、哪个 commit"。

### 风险

- 平台化容易过度设计 → 只做当前实验块真正用到的能力；
- 网页与 CLI 双写逻辑 → 统一走子进程/同一函数，禁止复制业务逻辑。

---

## 每块结束的固定动作

1. 更新本文件的状态与"下一块范围"；
2. 更新 [00_status_review.md](00_status_review.md) 的分数与差距表；
3. 追加 `docs/construction_log.md`（做了什么、为什么、验证命令）；
4. `python -m pytest -q` 全绿；
5. 推送并在 README 的"当前状态"里写清现在走到哪一块。
