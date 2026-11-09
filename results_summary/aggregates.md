# 多 seed 汇总（跨数据集）

| dataset | experiment | primary unit | split | dropout | runs | seeds | bAcc mean | bAcc 95% CI | AUROC mean |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T230224+0800_seed1_58e9c78e | 0.200 | 1 | 0 | 1.000 | [1.000, 1.000] | 1.000 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T230229+0800_seed3_9a18fbd8 | 0.200 | 1 | 0 | 1.000 | [1.000, 1.000] | 1.000 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T193524+0800_seed0_f7eb7f76 | 0.200 | 5 | 0/1/2/3/4 | 0.833 | [0.833, 0.833] | 0.911 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T193524+0800_seed0_f7eb7f76 | 0.000 | 1 | 0 | 0.833 | [0.833, 0.833] | 0.889 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T193524+0800_seed0_f7eb7f76 | 0.400 | 1 | 0 | 0.833 | [0.833, 0.833] | 0.889 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T230227+0800_seed2_9a586a2d | 0.200 | 1 | 0 | 1.000 | [1.000, 1.000] | 1.000 |

> 只有 dataset、experiment、primary unit、划分与超参签名都相同的运行才会被合并；
> 每个分组的详细结果在同目录 `aggregates/<experiment>__<unit>__<签名>/`。
