# 第一实验块（E1）：把"能跑"变成"能比的基准"

日期：2026-11-15｜前置：D15 完成｜状态：**未开始**

> 目标不是再加抽象，而是用现有框架**跑出一张可信、可复现、可比较的基准表**，
> 并用这张表反过来验收"层与层之间是否真的低耦合"。

## 1. 这一块要回答的科学与工程问题

1. 同一套数据层/特征层/模型层，能不能同时容纳 3 个以上真实数据集，且**不改上层代码**？
2. 同一数据集、同一划分下，结果对训练 seed 有多敏感？换一批人/换一个划分，结论是否还成立？
3. 现有的 MFCC + MLP 基线，在各库上的合理区间是多少？（不是"最优"，是"可复现的起点"）
4. 哪些库的标签/身份证据还不足以进入正式榜？必须被明确标出来，而不是混进平均。

## 2. 范围

### 做

- 接入 3 个真实数据集（详见 §4 W3）：
  - `uaspeech`：已有完整基线，本块只做**重跑 + 标签口径声明**；
  - `parkinson_kcl`：已有统计与特征验证，补齐**正式划分与基线**；
  - 新增 1 个库：优先 `label_folders` 形态（目录名即标签），用来验证"零代码接入"这条主线。
- 每个库：1 套固定任务定义 + 1 套固定划分 + 3 个训练 seed。
- 模型：`mfcc + mlp`。**可选加 `mfcc + lr`**（如果成本允许，见 W4）。
- 结果：每库一条基线、一张跨库汇总表（按任务/单位分组，不合并排名）、一份 benchmark 报告。
- 网页：所有 run 与图可在控制台看到。

### 不做（明确排除，避免范围膨胀）

- 不做 mel / CNN / SSL（属于 E2）。
- 不做多分类（C>2）、多标签、纵向任务（属于 E3/E4）；**本块只收二分类库**。
- 不做跨库合并训练或迁移学习。
- 不重新标注 UAspeech，不做官方说话人表回源（只在报告中声明证据等级）。
- 不做多 GPU、后台任务队列。

## 3. 第一实验块的"完成定义"（Definition of Done）

- ≥3 个数据集，每个 ≥3 个 seed 的 run，全部通过结果契约校验（`succeeded`，预测覆盖 100%）。
- `results_summary/leaderboard.csv` 中至少有 3 个数据集、按 `evaluation_unit` 分组展示。
- 新增的那个数据库：**只新增 `configs/datasets/*.yaml`（+ 必要时数据卡），未修改 `modeling/`、`results/`**；用 `git diff --stat` 作为证据贴进报告。
- 任一 seed 的 run 可以从 `predictions_*.csv` 重算出 `metrics.json`（容差内一致）。
- 一份 `docs/benchmark/E1_report.md`（或网页 + Markdown），写清每个数字的：task / split / view / feature_key / commit / 标签证据等级。
- 一条命令能重跑整块（脚本或 suite 雏形），失败/阻塞的库在汇总中可见。

## 4. 工作项与验收

### W1 任务身份与配置生效（先做，其他都依赖它）

**问题**：`prepared_id` 是时间戳，实验 YAML 手钉 ID；`evaluation.binary_threshold` 没被读取。

**动作**
1. `prepare` 时计算任务定义哈希（标签映射 + 筛选 + grouping + 版本），写入 `admission.json` 与 `run.json`；
2. `run` 时校验实验配置指向的 `task` 与 manifest 记录一致，不一致直接失败并提示重跑 `prepare`；
3. `prepared_id` 前缀加 `<task_hash8>`（保留时间戳，目录结构不变）；
4. 让 `binary_threshold` 真正进入指标计算与 `metrics.json`，或在未实现前**拒绝**该字段。

**验收**
- 改一行 `label_map` 后，旧的 `prepared_id` 不能再被 `run` 接受；
- 把阈值改成 0.6，`metrics.json` 的混淆矩阵随之变化；
- 现有测试全绿，新增 2 条针对上述行为的测试。

### W2 批量执行的最小外壳（不做平台，只做脚本）

**问题**：N 库 × M seed 靠手敲命令，不可维护。

**动作**：在 `scripts/` 下加一个薄脚本（或 `sddbench` 子命令）：
读取一份 `configs/suites/e1_baseline.yaml`（库清单 + seed 列表 + 实验模板），
顺序执行或续跑，逐 job 写 `runs.csv`（status / run_id / 原因）。

**验收**
- 中途 kill 后重跑，已完成 job 跳过，未完成 job 继续；
- 某个库 blocked（如标签未确认）时，其他库继续跑，最后汇总显示 `blocked=1`；
- 不引入新依赖。

### W3 数据集接入（每库一个提交）

| 库 | 形态 | 需要做的事 | 验收 |
| --- | --- | --- | --- |
| `uaspeech` | rich（已有） | 重跑新特征缓存 + 3 seeds；报告写明标签来源 | 3 个 run succeeded，`feature_key` 一致 |
| `parkinson_kcl` | rich（已有） | 确认 ReadText 任务、人员分组、生成 prepared/split | 有 1 条正式基线，人数与 README 口径一致 |
| 第三个库 | `label_folders` | 只写 `configs/datasets/*.yaml` + 数据卡 | 无 Python 改动；`prepare` 自动确认；`evaluation.unit=recording` |

**共同验收**：每个库的数据卡写清来源、标签依据、人数/文件数、已知风险。

### W4 基线矩阵

- 每库：seed 0/1/2 各一条 run，同一 `split_id`（划分 seed 固定，只变训练 seed）。
- 可选：`mfcc + lr`（逻辑回归）作为第二模型，用来证明"模型可替换"而不是"只有一个模型"。
  - 若做，只允许新增模型实现与注册，不改数据层/结果层；
  - 若不做，必须在报告中说明"单模型榜不构成模型比较"。

**验收**：每库 ≥3 run；同库同 split 下，模型/seed 可横向对比。

### W5 汇总、区间与敏感性

- `aggregate`：每库每模型输出 mean / std / 95% CI（bootstrap 按人/录音分组）。
- `split_sensitivity`：至少对 1 个库做"换划分 seed"的对比（UAspeech 已有，扩展到新库）。
- `leaderboard`：按 `task_id + evaluation_unit` 分组，禁止跨单位平均。

**验收**：汇总表里每个数字都能追溯到 run_id；CI 与箱线图一致。

### W6 标签与证据等级（科学边界）

每个库在报告中标注证据等级：

- A：官方元数据/论文可确认；
- B：目录名/文件名可确认，但未回源；
- C：存在已知冲突或未核验。

UAspeech 当前只能标 **B（且已知 F02 冲突）**。KCL 依据 ReadText/HC/PD 目录，标 B。新库按实际情况标。

**验收**：报告首页有这张等级表；C 级库不进入"可引用结论"部分。

### W7 文档

- `docs/benchmark/E1_report.md`：目标、数据、协议、结果表、图、失败/阻塞清单、证据等级、复现命令。
- 更新 `docs/datasets/<new>.md` 数据卡。
- 更新 `README.md` 的"当前状态"与 [00_status_review.md](00_status_review.md) 的分数。
- 更新 `docs/experiments/registry.csv`（W1 若已建立）。

**验收**：一个没有上下文的 agent 只读 `docs/dataset_onboarding.md` + 新库路径，能独立完成接入（用第三库实测一次）。

### W8 网页验收

- 控制台能看到 3 个库的 prepared / split / runs / figures / leaderboard；
- 页面数字与 `results_summary` 一致。

**验收**：浏览器逐项核对，截图或验收记录写入 `docs/acceptance_log.md`。

## 5. 建议执行顺序

```text
W1 任务身份 ──> W3 接库（KCL 先，第三库后）──> W4 矩阵 ──> W5 汇总
      │                    │
      └──> W2 批量外壳 ─────┘
                                  └──> W6 证据等级 ──> W7 文档 ──> W8 网页
```

W1 和 W2 是"一次性投入、后面每块都受益"的基础设施，先做。W3 每接一个库就提交一次，保持可回滚。

## 6. 已知风险与应对

| 风险 | 应对 |
| --- | --- |
| UAspeech 标签真值存疑 | 标 B 级；报告写"技术基线"；不与其他论文数字直接对比 |
| KCL 人数少（21 HC / 16 PD） | CI 会很宽；只作工程闭环，不作强结论；报告中写明 n |
| 第三个库选不到合适的 | 退化为只用 UAspeech + KCL，并把"零代码接入"用合成目录在测试中验收（已有 `test_label_folders.py`） |
| 特征重算时间长 | 先小样本 smoke，再全量；decode 缓存留到 E2 |
| 多分类需求提前出现 | 不在此块硬塞；记入 E3，先保证二分类正确 |

## 7. 本块结束后应该能回答

- 框架是否真的做到"加库不改上层"？（有 git diff 证据）
- 每库基线的合理区间与波动有多大？
- 哪些库可以进入正式榜，哪些还缺证据？
- 下一块最该投入的是特征/模型扩展（E2），还是统计与跨库比较（E3）？
