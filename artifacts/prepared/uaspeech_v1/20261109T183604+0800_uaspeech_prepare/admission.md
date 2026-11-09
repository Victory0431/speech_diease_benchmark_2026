# 准入预检：20261109T183604+0800_uaspeech_prepare

- 数据集：`uaspeech`｜任务：`uaspeech_v1`
- 源扫描：`20261109T171421+0800_uaspeech_full_dd025653`（full / completed）
- 状态：**blocked_label_review_incomplete**
- 标签复核文件：`/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/artifacts/label_review/uaspeech_label_review.csv`（存在）
- 输出目录：`/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/artifacts/prepared/uaspeech_v1/20261109T183604+0800_uaspeech_prepare`

## 1. 选择规则（来自任务配置）

- mic：`(未指定)`｜每个 origin 只保留一条：`True`
- variant：`original`｜elicitation：`unknown`
- 合格音频：143290｜应用规则后：143290｜speaker 数：28（其中参与本次清单 28）

## 2. 说话人复核覆盖

| speaker_token | directory_group | n_files | n_origins | mic_tokens | review_status |
| --- | --- | --- | --- | --- | --- |
| CF02 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CF03 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CF04 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CF05 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM01 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM04 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM05 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM06 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM08 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM09 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM10 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM12 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| CM13 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| F02 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| F03 | Healthy | 5184 | 741 | M2,M3,M4,M5,M6,M7,M8 | pending |
| F04 | Healthy | 5251 | 751 | M2,M3,M4,M5,M6,M7,M8 | pending |
| F05 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| M01 | Healthy | 2805 | 765 | M3,M5,M6,M7 | pending |
| M04 | Healthy | 3825 | 765 | M3,M5,M6,M7,M8 | pending |
| M05 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| M07 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| M08 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| M09 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| M10 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| M11 | Healthy | 4590 | 765 | M3,M4,M5,M6,M7,M8 | pending |
| M12 | Healthy | 4590 | 765 | M3,M4,M5,M6,M7,M8 | pending |
| M14 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | pending |
| M16 | Healthy | 4590 | 765 | M3,M4,M5,M6,M7,M8 | pending |

## 3. 阻断项

- `label_review_unconfirmed`：28 speakers are not confirmed: ['CF02@Disorder', 'CF03@Disorder', 'CF04@Disorder', 'CF05@Disorder', 'CM01@Disorder', 'CM04@Disorder', 'CM05@Disorder', 'CM06@Disorder', 'CM08@Disorder', 'CM09@Disorder']
- `mic_not_selected`：this dataset has multiple microphones per utterance; set selection.mic in the task config (or --mic) before building an approved manifest
- `multiple_rows_per_origin`：21382 origins have >1 selected row after the mic rule; examples: [(('CF02', 'CF02|B1|C10'), 7), (('CF02', 'CF02|B1|C11'), 7), (('CF02', 'CF02|B1|C12'), 7), (('CF02', 'CF02|B1|C13'), 7), (('CF02', 'CF02|B1|C14'), 7), (('CF02', 'CF02|B1|C15'), 7), (('CF02', 'CF02|B1|C16'), 7), (('CF02', 'CF02|B1|C17'), 7), (('CF02', 'CF02|B1|C18'), 7), (('CF02', 'CF02|B1|C19'), 7)]

## 4. 批准清单

_未生成 `manifest.csv`：准入未通过，不能把本轮结果用于训练。_

## 5. 人员隔离规则

- `group_id = <dataset>:<confirmed_subject_id>`；同一人的所有录音必须同组。
- 同一 subject 出现多个 label、同一 origin 归属多个 subject、或同一 subject 绑定多个 speaker token 都会阻断。
- 同一 utterances 的多麦克风文件只保留 `selection.mic` 指定的一条。

## 6. 下一步

- 按 `label_review.csv` 的列填写证据并改为 `review_status=confirmed`；重跑 `sddbench prepare`。
- 未确认前可继续做技术统计/特征，但不得导出为训练清单。
