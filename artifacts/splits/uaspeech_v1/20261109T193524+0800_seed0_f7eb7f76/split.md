# 数据划分：20261109T193524+0800_seed0_f7eb7f76

- 数据集：`uaspeech`｜任务：`uaspeech_v1`
- 源清单：`20261109T193404+0800_uaspeech_prepare`（manifest 指纹 `98583ae144aead61`）
- 随机种子：`0`｜比例：train 0.6 / val 0.2 / test 0.2
- 单位：`group_id`（确认后的 subject），任何一个人只出现在一个 split

## 1. 总量

- 录音：21126｜subject：28

| split | 录音数 | subject 数 | 标签录音分布 | 标签 subject 分布 |
| --- | --- | --- | --- | --- |
| train | 12711 | 17 | {'control': 6591, 'disorder': 6120} | {'control': 9, 'disorder': 8} |
| val | 3825 | 5 | {'control': 2295, 'disorder': 1530} | {'control': 3, 'disorder': 2} |
| test | 4590 | 6 | {'control': 2295, 'disorder': 2295} | {'control': 3, 'disorder': 3} |

## 2. 按类别的 subject 分配

| 标签 | train | val | test |
| --- | --- | --- | --- |
| control | 9 | 3 | 3 |
| disorder | 8 | 2 | 3 |

## 3. 隔离检查

- 已断言：同一 `group_id` 的所有录音只属于一个 split。
- 已断言：split 行数 = manifest 行数，无漏样本、无重复。
- 该划分是确定性的：相同 seed + manifest 会得到相同结果。

## 4. 边界

- 这是**按人划分**，不是按录音随机划分；因此同一人的不同句子不会跨集。
- 训练时标准化统计量只能来自 train，不得使用 val/test（由 `run` 强制）。
- 该 split 不改变原始数据，只写索引文件。
