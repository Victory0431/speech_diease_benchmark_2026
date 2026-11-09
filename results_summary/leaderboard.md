# 跨数据集基准榜（test / subject 为主指标）

- 结果契约版本：`1.0`
- 运行数：`3`
- 排序：`subject_balanced_accuracy` 降序；同分看 `subject_auroc`。
- 这里只含聚合指标与小体积图；权重、逐录音预测、患者级清单不在仓库。

| # | dataset | run_id | test subjects | seed | dropout | bAcc | sens | spec | precision | F1 | AUROC | AP | log loss | Brier | window bAcc |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uaspeech | 20261109T201821+0800_uaspeech_mfcc_mlp_baseline_s0_a522c1 | 6 | 0 | 0.200 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.438 | 0.135 | 0.699 |
| 2 | uaspeech | 20261109T202353+0800_uaspeech_mfcc_mlp_baseline_s0_f884b4 | 6 | 0 | 0.000 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.439 | 0.136 | 0.698 |
| 3 | uaspeech | 20261109T202610+0800_uaspeech_mfcc_mlp_baseline_s0_82e407 | 6 | 0 | 0.400 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.442 | 0.136 | 0.696 |

> 主指标是 test 说话人级 balanced accuracy：同一人的录音先平均，再按人计算。
> 只有个位数测试说话人时，一两人的差异就可能改变排序；跨数据集比较还需要
> 多 seed、置信区间与同一套数据端参数，不能只看单行数字。
