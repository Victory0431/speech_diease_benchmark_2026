# 平台现状评审：对照 v0.1 规范，我们漏了什么

日期：2026-11-15｜范围：D1–D15（施工批次）｜结论对象：`sddbench` 当前代码与产物

> 这份文档回答三个问题：**架子搭到什么程度、和最初的设计差在哪、第一实验块该干什么。**
> 后续规划见 [01_experiment_block_1_benchmark.md](01_experiment_block_1_benchmark.md) 与
> [02_experiment_blocks_2_plus.md](02_experiment_blocks_2_plus.md)。

## 0. 三份基准文档与本文的关系

| 文档 | 作用 | 仓库位置 |
| --- | --- | --- |
| 产品与代码架构规范 v0.1（HTML） | 最初的目标契约：五模块、数据类型、指标政策、错误码、P0–P6 | `docs/roadmap/reference/product_architecture_spec_v0.1.html` |
| 仓库结构与分阶段施工说明 v1 | 施工批次划分与验收口径 | `docs/roadmap/reference/construction_plan_v1.md`（同 `docs/construction_plan.md`） |
| 数据层补充设计与施工方案（UAspeech 先行） | 数据层细节的第一次设计 | `docs/data_layer_plan.md`（已标注为历史记录） |

规范是"目标态"，施工说明是"怎么排期"，本文是"**实际到哪了**"。三者不一致时，以代码和产物为准，本文负责记录差异。

## 1. 一句话结论与评分

**架子已经搭起来了，而且分层方向是对的；但它现在是一个"能跑通闭环的框架"，还不是一个"能支撑论文的基准"。**

| 视角 | 分数 | 说明 |
| --- | --- | --- |
| 作为可扩展框架 | **7.5 / 10** | 层边界清楚，新增普通分类库已能做到零代码；但任务身份、批量执行、多分类三处仍是硬约束 |
| 作为已完成的基准 | **3.5 / 10** | 只有 UAspeech 一条完整基线；KCL 只到特征验证；没有任何跨库对比表 |
| 综合 | **6.6 / 10** | 框架跑在结果前面，这是好事；下一块的任务是把结果补上，而不是继续加抽象 |

分维度打分（权重为本文自定，用于横向比较，不是客观真理）：

| 维度 | 权重 | 得分 | 理由 |
| --- | --- | --- | --- |
| 架构分层与扩展点 | 20% | 8.0 | data / features / modeling / results / reporting / web 边界清楚；特征层与数据层已解耦 |
| 数据层（扫描、准入、划分、接库） | 20% | 7.5 | 扫描与统计扎实；`label_folders` 让普通库零代码接入；但身份不是内容寻址，标签证据链偏弱 |
| 特征层 | 10% | 6.0 | 注册表与身份已补齐；实际只有 MFCC 一种，缓存形状只支持定长向量 |
| 模型层 | 15% | 5.0 | 只有 MLP；没有 LR 基线、没有能力声明、没有第二类模型可验证"插件"是否真的成立 |
| 结果层与统计 | 15% | 7.0 | 二分类指标齐全、run 包完整、多 seed 与换划分敏感性已有；多分类与阈值配置是缺口 |
| 实验组织与自动化 | 10% | 5.5 | 有 sweep；无 suite、无 plan 阶段、无失败分级，跑 3 库 × 3 seed 需要手工拼命令 |
| 文档与可维护性 | 5% | 8.0 | architecture / onboarding / construction_log 三件套齐；旧文档已标历史 |
| 科学结论与基准覆盖 | 5% | 3.0 | 1 个库 1 条基线；UAspeech 标签真值仍有疑点，尚不足以对外引用 |

## 2. 与 v0.1 规范的逐项对照

| 规范章节 | 要求 | 现状 | 状态 |
| --- | --- | --- | --- |
| §04 数据结构 | `SourceRecord` / `TaskSample` / `TaskSnapshot` 类型化契约 + `task_hash` | DataFrame + `schema.py` 列常量；无 `task_hash` | 部分 |
| §05 数据集插件 | 每库一个 Adapter，注册表显式登记 | `uaspeech` / `parkinson_kcl` / `label_folders` | 部分（EATD 未接） |
| §06 任务编译与划分 | 三步：编译 → 审计 → 划分 | `prepare`（含准入）+ `split`，无独立 `task build/audit` | 部分 |
| §07 音频视图 | Hz/带宽/观察时长/位置/多窗/tail 阈值 | 支持 native/重采样、单段、head/center/tail/fixed_random、短音频补零；**无带宽、无多窗+stride、无 tail_min** | 部分 |
| §08 模型层 | Runner + 插件 + capabilities | 单一 MLP，`model.name != mlp` 直接报错 | 缺口大 |
| §09 配置组合/扫参 | Planner、suite、分层网格 | `experiment` + `sweep`（grid），无 planner/suite | 部分 |
| §10 调用顺序与缓存 | train-valid / evaluate 分离；decode/view/feature 三类缓存 | 一次 run 内闭环；只有特征缓存，解码每次重做 | 部分 |
| §11 指标 | 二分类 + 多分类；NA 政策；阈值记录 | 二分类完整（Acc/BalAcc/Sens/Spec/PPV/NPV/F1/AUROC/AP/LogLoss/Brier/CM）；**多分类缺失** | 部分 |
| §12 run 包 | 固定文件集、可重算 | `run.json`/`metrics.json`/预测/图/报告齐全；无 `reproduce.sh`、无 `status.json`+`events.jsonl` | 大部分 |
| §13 CLI | `plan` / `run --phase` / `evaluate` / `suite` | `run --dry-run` 近似 plan；其余未实现 | 部分 |
| §14 错误码 | `E_LABEL_UNRESOLVED` 等稳定错误码 | 退出码 + 中文/英文文本 | 缺口 |
| §15 验收 P0–P6 | P0 扫描、P1 合成端到端、P2 MDVR、P3 EATD、P4 run 包、P5 模型/扫参、P6 正式套件 | P0/P2/P4 基本达成；P1 部分；P3 未做；P5 部分（sweep 有，CNN/SSL 无）；P6 未做 | 部分 |
| §16 架构验收 | 第二个 Adapter 不得修改 Trainer/Evaluator | 当前代码未因 KCL 改动训练器，但**没有自动化回归测试**证明这一点 | 弱 |
| §17 实验块 | 平台验收 → 受控实验 | 本文档补齐 | 本次 |

## 3. 被忽略或低估的问题（按影响排序）

### P1 任务身份不是内容寻址（最该先修）

`prepared_id` / `split_id` 是时间戳作业号（`20261109T193404+0800_uaspeech_prepare`），实验配置里手工写死。
后果：

- 改了 `configs/datasets/<dataset>_v1.yaml` 的标签映射或筛选规则，旧 `prepared_id` 仍然"看起来有效"；
- 多数据集批量跑时，配置与产物靠人记忆对应，容易张冠李戴；
- 论文复现时，读者无法从 ID 判断用了哪版任务定义。

规范里的 `task_hash` 就是为这个设计的。修法很小：在 `admission.json` / `run.json` 里记录任务配置哈希，`run` 时校验；`prepared_id` 前缀改成 `<task_hash8>_<stamp>` 即可，不必推翻现有目录结构。

### P2 没有 suite / plan 层，第一块就会卡住

现在只能"一条 run 一个命令"或"一个 sweep 一个网格"。第一实验块是 **N 个库 × M 个 seed**，还要区分"计划（哪些能跑、哪些被阻塞）"和"执行"。缺这一层的直接表现：命令散落在 README，失败后不知道哪些库没跑成。

### P3 多分类完全没有实现

`modeling/metrics.py` 只有 `binary_metrics`，混淆矩阵固定 2×2，`per_class_rows` 固定两行。`class_order` 虽然可以写三个类，但指标会静默按二分类算。只要接入一个三分类数据集（比如认知障碍分级），结果就是错的——而且不会报错，这是最危险的一类问题。

### P4 视图只支持"单段定长"，核心音频实验做不了

规范 §07 的"观察 16 秒、4 秒窗、步长 4 秒"目前不支持：`view_mode` 只接受 `single_segment`，`window_seconds` 默认等于 `observe_seconds`。也就是说，"听多久""听哪一段""窗长多少"这三类最基础的敏感性实验，现在只能做一个参数（总时长），不能做多窗聚合。带宽（8k/16k 受控降带宽）同样没有。

### P5 阈值配置是"静默无效"的

实验 YAML 里的 `evaluation.binary_threshold: 0.5` 没有被任何代码读取，`binary_metrics` 内部硬编码 0.5。今天恰好配置也是 0.5，所以看不出来；一旦有人改成 0.6，`metrics.json` 里的 `probability_threshold` 仍会写 0.5 还是 0.6，但计算不会变。这违反"配置必须生效或明确报错"的原则。

### P6 没有解码/视图缓存

每次特征作业都重新解码原始音频。UAspeech 有 14 万文件，改一次 `n_mfcc` 就要重新解码一遍，成本不可接受。规范 §10 把缓存分成 decode/view/feature 三类，现在只做了第三类。

### P7 失败与阻塞没有分级语义

现在只有退出码 3 和一段文字。批量跑 20 个库时，需要的是"每个 job 的 `succeeded / failed / blocked / skipped` 和原因"，并且汇总时必须显示"3 个失败、2 个阻塞"，而不是让失败从表里消失。

### P8 UAspeech 的标签真值仍未回源

规范 §02 与 R5 明确指出：UAspeech 原始论文里 F02 属于构音障碍说话人，而 fresh 目录里 `Healthy/F02` 与 `Disorder/CF02` 并存，旧整理映射至少存在冲突。当前基准按目录名取标签，`label_source` 记为 `directory:*`。这意味着：

- 现有 UAspeech 数字只能称为"**技术基线**"，不能称为"UAspeech 官方基准"；
- 第一实验块必须在报告里显式写"标签依据 = 目录名，未经官方元数据核验"；
- 如果要做可引用结论，需要一次标签回源（读官方说话人表）——这不是代码问题，是数据证据问题。

### P9 没有实验登记表

哪个库、哪个 commit、哪套配置、哪次 run、结论是什么，目前散在 `construction_log.md` 和 `results_summary/` 里。库一多，同一条基线和它对应的配置就会失联。建议在第一块内加一张 `docs/experiments/registry.csv`（run_id / task / split / model / 日期 / 结论 / 链接），由 `publish` 自动追加。

## 4. 已经做得比规范更好的地方

- **普通分类库的接入比规范更简单。** 规范要求每库写 Adapter + TaskSpec + 夹具 + 数据卡；`label_folders` 把最常见的"父目录下若干分类子目录"变成纯配置，`prepare` 自动确认标签，不需要人工复核表。
- **特征身份比规范更明确。** 规范只在实验配置里写 `frontend.id`；现在 `feature_type` 是必填字段，缓存目录、索引、`run.json`、网页生成文件名四处一致，加 `mel` 时不会和 MFCC 混淆。
- **多 seed 与换划分敏感性已经做了。** 规范里 Demo 阶段允许"不跑 CI"，D9 已经产出 `aggregates` 与 `split_sensitivity`，这属于提前完成。
- **网页控制台。** 规范建议 Demo 阶段不要做 GUI，但需求明确要求可调参；现在参数注册表、catalog、运行与看图已打通，是后续"实验管理"的现成载体。

## 5. 风险登记

| 风险 | 类型 | 触发条件 | 处理时机 |
| --- | --- | --- | --- |
| 配置与产物错配（无 task_hash） | 技术债 | 改任务 yaml 后复用旧 prepared_id | 第一块 W1，必做 |
| 多分类静默算错 | 正确性 | 接入 C>2 数据集 | 第一块内明确"只收二分类"，E3 补实现 |
| 阈值配置不生效 | 正确性 | 有人改 threshold | 第一块 W1，顺手修 |
| 标签真值不可引用 | 科学 | 对外发布 UAspeech 数字 | 第一块报告显式声明；E3 前完成回源 |
| 重复解码导致成本失控 | 性能 | 换特征/加模型 | E2 前做 decode/view 缓存 |
| 失败从汇总里消失 | 可复现性 | 批量跑多库 | 第一块 W2 用脚本兜底，E5 平台化 |

## 6. 编号约定（避免后面继续混乱）

- **D1–D15**：施工批次，代表"代码骨架推进到哪一步"。
- **E1–E5**：实验块，代表"科学问题推进到哪一步"。一个实验块可以包含多个 D 批次。
- **W1–W8**：实验块内部的工作项编号。

当前状态：`D15 完成`，位于 **E1 的起点**。
