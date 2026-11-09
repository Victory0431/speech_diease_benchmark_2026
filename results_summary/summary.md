# 实验结果汇总

- 运行数：`3`
- 排序：test subject balanced accuracy 降序（仅用于浏览，不替代统计检验）。
- 本文件只含聚合指标；逐录音预测、权重与患者级清单留在本机 `artifacts/runs/`。

| # | run_id | hidden | dropout | lr | seed | best epoch | test subj bAcc | sens | spec | AUROC | test win bAcc | status |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 20261109T201821+0800_uaspeech_mfcc_mlp_baseline_s0_a522c1 | [128] | 0.200 | 0.001 | 0 | 6 | 0.833 | 1.000 | 0.667 | 0.889 | 0.699 | completed |
| 2 | 20261109T202353+0800_uaspeech_mfcc_mlp_baseline_s0_f884b4 | [128] | 0.000 | 0.001 | 0 | 4 | 0.833 | 1.000 | 0.667 | 0.889 | 0.698 | completed |
| 3 | 20261109T202610+0800_uaspeech_mfcc_mlp_baseline_s0_82e407 | [128] | 0.400 | 0.001 | 0 | 6 | 0.833 | 1.000 | 0.667 | 0.889 | 0.696 | completed |

> test 只有个位数说话人时，一两人的差异就可能改变排序；把这里的排序当作「下一轮实验的线索」，不要当作最终结论。
