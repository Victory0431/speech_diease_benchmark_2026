# 实验运行：20261109T230840+0800_uaspeech_mfcc_mlp_baseline_s0_e7bd8a

## 1. 这次运行做了什么

- 任务：`uaspeech_v1`（数据集 `uaspeech`）
- 数据准备：`20261109T193404+0800_uaspeech_prepare`；固定划分：`20261109T230227+0800_seed2_9a586a2d`
- 特征：D3 缓存的 MFCC 统计量，`40 维/窗口`，采样率 `16000 Hz`，缓存 key `mfcc_stats_7821f53b7b6c12bd`
- 模型：`mlp`，隐藏层 `[128]`，dropout `0.2`，输入 `40` → 输出 `2`
- 训练：epochs `80`，batch `64`，lr `0.001`，优化器 `adamw`，seed `0`，权重 `subject_balanced`
- 早停/选模型：验证集 **subject balanced accuracy**，最佳 epoch `3`（`1.000`）
- 运行状态：`completed`；开始 `2026-11-09T23:06:15+08:00`，结束 `2026-11-09T23:08:40+08:00`
- 代码：`0.1.0`，git `6a0da437c3932a18626d6612bc4816ce51d21136`（dirty=True）

## 2. 数据与划分

- 总录音：`21126`；分组（group）：`28`
| split | 录音数 | 分组数 | 标签录音分布 |
| --- | ---: | ---: | --- |
| train | 12711 | 17 | {'control': 6591, 'disorder': 6120} |
| val | 3825 | 5 | {'control': 2295, 'disorder': 1530} |
| test | 4590 | 6 | {'control': 2295, 'disorder': 2295} |

- 划分按 `group_id` 进行：本库的分组单位是确认后的 subject，train/val/test 的分组集合互斥；训练时标准化统计量只用 train 拟合。

## 3. 指标

| split | 单位 | n | accuracy | balanced acc | sensitivity | specificity | F1 | AUROC | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | window | 12711 | 0.945 | 0.945 | 0.947 | 0.943 | 0.943 | 0.987 | 0.987 |
| train | recording | 12711 | 0.945 | 0.945 | 0.947 | 0.943 | 0.943 | 0.987 | 0.987 |
| train | subject | 17 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| val | window | 3825 | 0.925 | 0.917 | 0.877 | 0.957 | 0.903 | 0.976 | 0.971 |
| val | recording | 3825 | 0.925 | 0.917 | 0.877 | 0.957 | 0.903 | 0.976 | 0.971 |
| val | subject | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| test | window | 4590 | 0.825 | 0.825 | 0.779 | 0.872 | 0.817 | 0.913 | 0.913 |
| test | recording | 4590 | 0.825 | 0.825 | 0.779 | 0.872 | 0.817 | 0.913 | 0.913 |
| test | subject | 6 | 1.000 | 1.000 **←主指标** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

正类：`disorder`；概率阈值 0.5；主评价单位：subject。

### Test 混淆矩阵（subject）

| true \ predicted | control | disorder |
| --- | ---: | ---: |
| control | 3 | 0 |
| disorder | 0 | 3 |

## 4. 怎么读这个结果

- **主评价单位**由任务的分组策略决定：有说话人时按人等权平均，没有说话人时按录音；**window 指标**只用于诊断，不能拿来当最终成绩。
- 本基线每条录音只取 1 个观察窗（head 4 s），因此 window 与 recording 指标相同；两者都保留，是为了以后多窗口协议不改报告结构。
- 测试集从未参与早停或阈值选择；`test` 只在最后评估一次。
- 分数高低不是工程验收标准；先看流程是否可追溯、预测是否覆盖完整、指标是否可由预测重算。

## 5. 限制

- val 只有 `5` 个分组，单次结果的方差较大；正式比较应多 seed 并报告不确定性。
- MFCC 统计量丢弃了时间顺序信息，MLP 只能利用全局均值和标准差；这是第一条可复现基线，不代表方法上限。
- 4 s 观察窗对约 42.6% 的录音需要右补零；补零帧不进入 MFCC 统计，但短录音的信息量确实更少。

## 6. 产物索引

| 文件 | 内容 |
| --- | --- |
| `config.yaml` | 完整生效配置（含 CLI 覆盖） |
| `run.json` | 运行身份、环境、数据指纹、指标与状态 |
| `history.csv` | 每个 epoch 的 loss / accuracy / 验证主指标 |
| `best.pt` | 被选中的模型权重、类序、标准化参数 |
| `predictions_windows.csv` | 窗口级概率与预测 |
| `predictions_subjects.csv` | 人员级概率与预测（主指标来源） |
| `metrics.json` / `metrics_per_class.csv` | 统一指标与每类明细 |
| `figures/` | 混淆矩阵、ROC、分数分布、学习曲线 |
| `inputs/` | 本次使用的 manifest / split / task 快照 |

## 7. 最佳 epoch 记录

- epoch `3`：train_loss `0.172`，val_loss `0.191`，val subject balanced accuracy `1.000`
