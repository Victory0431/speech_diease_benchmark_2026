# UAspeech 数据集档案（本地版本）

状态：**技术接入；`technical_only / label_conflict`；`benchmark_eligible=false`。**

- 机器可读状态：`configs/datasets/uaspeech_v1.yaml` 的 `scientific_status`。
- 历史结果：保留为技术结果，不覆盖、不改写；不得引用为正式疾病识别基准成绩。
- 准入清单仍可生成（`status=approved` 只表示结构可用），但不会进入正式主榜；
  `publish` 只给 `benchmark_eligible=true` 的 run 排名。
- 本文件记录本机 `fresh_datasets/UAspeech` 的实测结构；不照抄文献数字。

## 来源与许可

- 语料：Kim et al., "Dysarthric Speech Database for Universal Access Research",
  Interspeech 2008（[R3]）。多麦克风采集、角色约定以官方说明为准。
- 本地版本来源与许可条款：**待补**（人工核对原始下载记录后填写）。
- 使用边界：原始数据只读；不提交任何真实音频到 Git。

## 本地实测结构（2026-11-09 预检）

```text
UAspeech/
├── Disorder/   69,615 个 .wav，13 名说话人，全部 C 前缀 token（CF02..CM13）
└── Healthy/    73,675 个 .wav，15 名说话人，无 C 前缀（F02..M16）
```

命名：`<speaker_token>_<block_token>_<prompt_token>_<mic_token>.wav`

| token | 实测域（full 扫描核实） | 含义 | 状态 |
| --- | --- | --- | --- |
| speaker | CF02…CM13（13人）/ F02…M16（15人） | 说话人标识；C 前缀在官方约定中为 Control | 候选，待核验 |
| block | B1 / B2 / B3 | 录音块 | 候选，待核验 |
| prompt | C0…C12、D0…D9（数字）、UW#、CW#、LA…LZ（共 255 个） | 词项/语料；LA–LZ 与 D 系列语义待对照官方说明 | 候选，待核验 |
| mic | M2…M8（7 个） | 麦克风通道；同一词项多 mic = 同源录音 | 候选，待核验 |

满配结构（full 扫描核实）：255 词项 × 3 块 × 7 mic = 每人 5,355 个文件。

- Disorder：13 人全部满配（69,615）。
- Healthy：8 人满配；7 人缺录音（如实报告，不补齐）：
  M01=2,805、M04=3,825、M11=M12=M16=4,590、F03=5,184、F04=5,251（共 73,675）。
- mic 覆盖不均：Disorder 各 mic 完全均衡（各 9,945）；Healthy 中 M2 明显缺录
  （7,610，其他 mic 约 9,906–11,436）。**选 M2 会让 Healthy 组少约 1/3 录音，
  选择前必须对照统计报告第 4 节。**

## 已知疑点（必须在 D4 label_review 中解决）

1. **目录组与官方说话人表冲突**：本库所有 C 前缀说话人位于 `Disorder/`，
   所有非 C 前缀位于 `Healthy/`。官方约定 C 前缀 = Control（对照组）；
   更关键的是，原论文说话人表把 **F02 列为构音障碍说话人（Spastic）**，
   本地却把 `Healthy/F02` 当作对照。目录名与官方角色疑似整体倒置。
2. `artifacts/label_review/uaspeech_label_review.csv` 的 `confirmed` 证据仅写
   “用户确认目录即标签”，这**只确认了目录含义，没有回源**，不能关闭第 1 条冲突。
3. 同一词项（speaker+block+prompt）最多 7 个 mic 文件：同源录音，
   不是独立样本；人员归组以 speaker_token 为准。
4. 无任何本地元数据/评分文件；标签只能来自 label_review 的实证核对。
5. 尚未核实每说话人块×词项×mic 的覆盖完整性（D1 stats 报告提供覆盖表）。

### 解除阻断需要什么证据

- 官方发行版的说话人表/元数据，能与本地 `speaker_token` 一一对应；
- 本地下载记录或版本说明，确认 `Disorder/`、`Healthy/` 是否由本地重新整理；
- 若确认目录与官方角色倒置，生成**版本化的新标签映射**（新 task/prepared/split
  版本），不修改原始文件，也不改写旧结果；
- 人工抽听只能作辅助，不能替代元数据证据，也不能据听感整体翻转标签。

## 标签核验程序（D4）

`<workspace>/label_review/uaspeech_label_review.csv`：
`speaker_token, directory_group, confirmed_subject_id, confirmed_label, evidence, review_status`。
适配器只接受 `review_status=confirmed` 的行；缺失该文件时所有人保持 `pending`，
禁止导出 approved 分类清单。若映射确实颠倒，写新映射并记录证据，不搬动/重命名原始文件。

## 变更记录

- 2026-11-09：初始档案，结构数据来自本机 os.listdir 预检；格式/时长/Hz 以 D1 stats full 报告为准。
