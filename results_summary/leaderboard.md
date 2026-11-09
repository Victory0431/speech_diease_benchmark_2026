# 跨数据集基准榜（test / 主评价单位）

- 结果契约版本：`1.1`
- 运行数：`10`
- 排序：`primary_balanced_accuracy` 降序；同分看 `primary_auroc`。
- `primary_unit` 标出主评价单位：有说话人时为 subject，没有说话人时为 recording；
  不同单位之间不直接比较，只并列展示。
- 这里只含聚合指标与小体积图；权重、逐录音预测、患者级清单不在仓库。

| # | dataset | run_id | primary unit | test groups | seed | dropout | bAcc | sens | spec | precision | F1 | AUROC | AP | log loss | Brier | window bAcc |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | uaspeech | 20261109T230611+0800_uaspeech_mfcc_mlp_baseline_s0_c4aff1 | subject | 6 | 0 | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.261 | 0.057 | 0.828 |
| 2 | uaspeech | 20261109T230840+0800_uaspeech_mfcc_mlp_baseline_s0_e7bd8a | subject | 6 | 0 | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.247 | 0.053 | 0.825 |
| 3 | uaspeech | 20261109T231100+0800_uaspeech_mfcc_mlp_baseline_s0_bd098c | subject | 6 | 0 | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.217 | 0.045 | 0.853 |
| 4 | uaspeech | 20261109T230928+0800_uaspeech_mfcc_mlp_baseline_s3_2cb19f | subject | 6 | 3 | 0.200 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 1.000 | 1.000 | 0.412 | 0.123 | 0.704 |
| 5 | uaspeech | 20261109T201821+0800_uaspeech_mfcc_mlp_baseline_s0_a522c1 | subject | 6 | 0 | 0.200 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.438 | 0.135 | 0.699 |
| 6 | uaspeech | 20261109T202353+0800_uaspeech_mfcc_mlp_baseline_s0_f884b4 | subject | 6 | 0 | 0.000 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.439 | 0.136 | 0.698 |
| 7 | uaspeech | 20261109T202610+0800_uaspeech_mfcc_mlp_baseline_s0_82e407 | subject | 6 | 0 | 0.400 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.442 | 0.136 | 0.696 |
| 8 | uaspeech | 20261109T230455+0800_uaspeech_mfcc_mlp_baseline_s1_4e4616 | subject | 6 | 1 | 0.200 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.452 | 0.141 | 0.698 |
| 9 | uaspeech | 20261109T230710+0800_uaspeech_mfcc_mlp_baseline_s2_3cf558 | subject | 6 | 2 | 0.200 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.433 | 0.132 | 0.694 |
| 10 | uaspeech | 20261109T231142+0800_uaspeech_mfcc_mlp_baseline_s4_58de49 | subject | 6 | 4 | 0.200 | 0.833 | 1.000 | 0.667 | 0.750 | 0.857 | 0.889 | 0.917 | 0.448 | 0.138 | 0.688 |

> 主指标是 test 主评价单位 balanced accuracy：subject 单位先把同一人的录音平均；
> recording 单位表示该库没有说话人信息，指标无法排除同一人跨划分的泄漏。
> 只有个位数测试单元时，一两人的差异就可能改变排序；跨数据集比较还需要
> 多 seed、置信区间与同一套数据端参数，不能只看单行数字。
