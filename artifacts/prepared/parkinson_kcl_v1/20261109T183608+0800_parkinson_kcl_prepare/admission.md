# 准入预检：20261109T183608+0800_parkinson_kcl_prepare

- 数据集：`parkinson_kcl`｜任务：`parkinson_kcl_v1`
- 源扫描：`20261109T183508+0800_parkinson_kcl_full_a83761e1`（full / completed）
- 状态：**blocked_label_review_incomplete**
- 标签复核文件：`/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/artifacts/label_review/parkinson_kcl_label_review.csv`（存在）
- 输出目录：`/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/artifacts/prepared/parkinson_kcl_v1/20261109T183608+0800_parkinson_kcl_prepare`

## 1. 选择规则（来自任务配置）

- mic：`(未指定)`｜每个 origin 只保留一条：`True`
- variant：`original`｜elicitation：`unknown`
- 合格音频：73｜应用规则后：73｜speaker 数：37（其中参与本次清单 37）

## 2. 说话人复核覆盖

| speaker_token | directory_group | n_files | n_origins | mic_tokens | review_status |
| --- | --- | --- | --- | --- | --- |
| ID00 | Healthy | 2 | 2 |  | pending |
| ID01 | Healthy | 2 | 2 |  | pending |
| ID03 | Healthy | 2 | 2 |  | pending |
| ID05 | Healthy | 2 | 2 |  | pending |
| ID08 | Healthy | 2 | 2 |  | pending |
| ID09 | Healthy | 2 | 2 |  | pending |
| ID10 | Healthy | 2 | 2 |  | pending |
| ID11 | Healthy | 2 | 2 |  | pending |
| ID12 | Healthy | 2 | 2 |  | pending |
| ID14 | Healthy | 2 | 2 |  | pending |
| ID15 | Healthy | 2 | 2 |  | pending |
| ID19 | Healthy | 2 | 2 |  | pending |
| ID21 | Healthy | 2 | 2 |  | pending |
| ID22 | Healthy | 2 | 1 |  | pending |
| ID23 | Healthy | 2 | 2 |  | pending |
| ID25 | Healthy | 2 | 2 |  | pending |
| ID26 | Healthy | 2 | 2 |  | pending |
| ID28 | Healthy | 2 | 2 |  | pending |
| ID31 | Healthy | 2 | 2 |  | pending |
| ID35 | Healthy | 2 | 2 |  | pending |
| ID36 | Healthy | 2 | 2 |  | pending |
| ID02 | Parkinson | 2 | 2 |  | pending |
| ID04 | Parkinson | 2 | 2 |  | pending |
| ID06 | Parkinson | 2 | 2 |  | pending |
| ID07 | Parkinson | 2 | 2 |  | pending |
| ID13 | Parkinson | 2 | 2 |  | pending |
| ID16 | Parkinson | 2 | 2 |  | pending |
| ID17 | Parkinson | 2 | 2 |  | pending |
| ID18 | Parkinson | 1 | 1 |  | pending |
| ID20 | Parkinson | 2 | 2 |  | pending |
| ID24 | Parkinson | 2 | 2 |  | pending |
| ID27 | Parkinson | 2 | 2 |  | pending |
| ID29 | Parkinson | 2 | 2 |  | pending |
| ID30 | Parkinson | 2 | 2 |  | pending |
| ID32 | Parkinson | 2 | 2 |  | pending |
| ID33 | Parkinson | 2 | 2 |  | pending |
| ID34 | Parkinson | 2 | 2 |  | pending |

## 3. 阻断项

- `label_review_unconfirmed`：37 speakers are not confirmed: ['ID00@Healthy', 'ID01@Healthy', 'ID03@Healthy', 'ID05@Healthy', 'ID08@Healthy', 'ID09@Healthy', 'ID10@Healthy', 'ID11@Healthy', 'ID12@Healthy', 'ID14@Healthy']
- `multiple_rows_per_origin`：1 origins have >1 selected row after the mic rule; examples: [(('ID22', 'ID22|hc|0_0_0'), 2)]

## 4. 批准清单

_未生成 `manifest.csv`：准入未通过，不能把本轮结果用于训练。_

## 5. 人员隔离规则

- `group_id = <dataset>:<confirmed_subject_id>`；同一人的所有录音必须同组。
- 同一 subject 出现多个 label、同一 origin 归属多个 subject、或同一 subject 绑定多个 speaker token 都会阻断。
- 同一 utterances 的多麦克风文件只保留 `selection.mic` 指定的一条。

## 6. 下一步

- 按 `label_review.csv` 的列填写证据并改为 `review_status=confirmed`；重跑 `sddbench prepare`。
- 未确认前可继续做技术统计/特征，但不得导出为训练清单。
