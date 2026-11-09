# 准入预检：20261109T193404+0800_uaspeech_prepare

- 数据集：`uaspeech`｜任务：`uaspeech_v1`
- 源扫描：`20261109T171421+0800_uaspeech_full_dd025653`（full / completed）
- 状态：**approved**
- 标签复核文件：`/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/artifacts/label_review/uaspeech_label_review.csv`（存在）
- 输出目录：`/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/artifacts/prepared/uaspeech_v1/20261109T193404+0800_uaspeech_prepare`

## 1. 选择规则（来自任务配置）

- mic：`M3`｜每个 origin 只保留一条：`True`
- variant：`original`｜elicitation：`unknown`
- 合格音频：143290｜应用规则后：21126｜speaker 数：28（其中参与本次清单 28）

## 2. 说话人复核覆盖

| speaker_token | directory_group | n_files | n_origins | mic_tokens | review_status |
| --- | --- | --- | --- | --- | --- |
| CF02 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CF03 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CF04 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CF05 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM01 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM04 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM05 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM06 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM08 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM09 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM10 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM12 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| CM13 | Disorder | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| F02 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| F03 | Healthy | 5184 | 741 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| F04 | Healthy | 5251 | 751 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| F05 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| M01 | Healthy | 2805 | 765 | M3,M5,M6,M7 | confirmed |
| M04 | Healthy | 3825 | 765 | M3,M5,M6,M7,M8 | confirmed |
| M05 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| M07 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| M08 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| M09 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| M10 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| M11 | Healthy | 4590 | 765 | M3,M4,M5,M6,M7,M8 | confirmed |
| M12 | Healthy | 4590 | 765 | M3,M4,M5,M6,M7,M8 | confirmed |
| M14 | Healthy | 5355 | 765 | M2,M3,M4,M5,M6,M7,M8 | confirmed |
| M16 | Healthy | 4590 | 765 | M3,M4,M5,M6,M7,M8 | confirmed |

## 3. 阻断项

_无阻断项。_

## 4. 批准清单

- 行数：21126｜subject 数：28｜origin 数：21126
- 标签分布：{'control': 11181, 'disorder': 9945}
- 指纹：`98583ae144aead616b464a1a2ea97444dd974e995652d331729fc3e44772da55`
- 文件：`manifest.csv`、`task.yaml`、`provenance.json`

## 5. 人员隔离规则

- `group_id = <dataset>:<confirmed_subject_id>`；同一人的所有录音必须同组。
- 同一 subject 出现多个 label、同一 origin 归属多个 subject、或同一 subject 绑定多个 speaker token 都会阻断。
- 同一 utterances 的多麦克风文件只保留 `selection.mic` 指定的一条。

## 6. 下一步

- 进入 split：按 `group_id` 做人员隔离划分，再关联特征缓存。
