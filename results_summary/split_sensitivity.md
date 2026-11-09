# 划分敏感性（同一配置，换划分 seed）

| dataset | experiment | unit | split | runs | seeds | bAcc mean | across-split mean | range |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T193524+0800_seed0_f7eb7f76 | 5 | 0/1/2/3/4 | 0.833 | 0.896 | 0.167 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T230224+0800_seed1_58e9c78e | 1 | 0 | 1.000 | 0.896 | 0.167 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T230227+0800_seed2_9a586a2d | 1 | 0 | 1.000 | 0.896 | 0.167 |
| uaspeech | uaspeech_mfcc_mlp_baseline | subject | 20261109T230229+0800_seed3_9a18fbd8 | 1 | 0 | 1.000 | 0.896 | 0.167 |

> 同一配置换划分后主指标的位移，是“6 个测试单元够不够”的直接证据；
> 位移大于训练 seed 波动时，优先扩充数据集/换更稳的评价协议，而不是调超参。
