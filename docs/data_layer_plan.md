# 数据层补充设计与施工方案：先统计，再选参数，再产生MFCC

日期：2026-11-09｜版本：v1.0｜供服务器端Codex按步骤实施。

配套文档：《语音疾病基准平台_仓库结构与分阶段施工说明_v1.md》。本文件补充数据层细节；关于实际路径、首个数据集、统计入口、固定随机裁剪、原采样率探索的安排，以本文件为准。模型、指标和结果追溯沿用宏观文档。

**本轮任务：先让UAspeech的统计、音频处理和MFCC真实跑通，不急着训练。数据路径由用户提供；本设计未连接服务器，不能声称已读到音频或统计到其时长、格式和类别。**

## 1. 已确定的真实路径与目录安排

```yaml
# configs/paths.local.yaml：真实服务器路径，不提交远程仓库
construction_root: /mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109
workspace_root: /mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/artifacts
dataset_roots:
  uaspeech: /mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/UAspeech
```

数据集集合根目录：

```text
/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets
```

首个单文件检查入口：

```text
/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/UAspeech/Disorder/CF02_B1_C1_M2.wav
```

为避免“施工目录”和“实验工作区”混淆，本补充统一安排：

```text
speech_disease_benchmark_workspace20261109/   # 用户给定施工根
├── repo/                                   # 新Git仓库；宏观文档的代码放这里
└── artifacts/                              # 统计、清单、划分、缓存、实验结果；不纳入Git
```

如果repo已有代码或已初始化，先检查并沿用，不能重新初始化覆盖。原始fresh_datasets只读，不在其内部写缓存、标签或转换后音频。服务器端缺路径权限时明确报错，不能改去其他同名目录。

### 1.1 首批数据集怎么选

1. **UAspeech：本轮必须实现。** 先单文件，再全库统计，再做未带训练标签的MFCC技术验证。
2. 第二库暂不指定一个未经确认的路径。完成UAspeech后，对fresh_datasets做一级目录清单；如果确有MDVR/KCL或EATD，确认名称与元数据后选一个接第二适配器。没有则报告候选，不能新下载或用相似名强行替代。

本次路径来自你已经整理过的fresh_datasets。它可以作为技术接入源，但“文件在Disorder里”不能自动证明病理标签正确。我们此前已发现旧整理映射有疑点；此处**允许无标签统计和特征提取，禁止未核验就进入正式分类训练**。

## 2. 最关键的解释：Adapter不直接把文件变成MFCC

你对“数据层处理好，下游统一”的理解正确。但数据层内部应分工，否则每个Adapter都复制一遍MP3解码、裁剪和MFCC，后面新增规则反而要改所有数据集。

| 步骤 | 谁负责 | 输入 | 输出 | 此时改变原音频吗 |
| --- | --- | --- | --- | --- |
| ① 发现文件 | 通用scanner | 数据根 | 文件清单，含音频/压缩包/未知文件 | 否 |
| ② 解释这个库 | 本库Adapter | 文件清单＋原始元数据 | 人员、来源、候选标签、任务/麦克风关系 | 否 |
| ③ 测量音频 | 通用decoder＋audit＋stats | 文件清单/标准记录 | 实测格式、Hz、时长、质量、统计报告 | 只解码读取，不改原件 |
| ④ 人工定方案 | 你＋配置文件 | 统计报告与研究目的 | 裁剪长度、位置、目标Hz、MFCC配置 | 否 |
| ⑤ 准备任务与划分 | prepare＋split | 已核验标签/人员和筛选规则 | 固定manifest与split | 否 |
| ⑥ 生成观察窗口 | audio＋windows | 清单＋参数 | 有效声音、补零/mask、窗口索引 | 仅内存/派生缓存 |
| ⑦ 计算特征 | features/mfcc.py | 有效声音及实际处理Hz | MFCC帧/40维统计向量＋索引 | 不改原件 |
| ⑧ 训练 | 下游统一代码 | 标准特征和固定split | 模型/预测/结果包 | 不再了解本库目录 |

**格式处理归通用decoder；身份、标签归Adapter；裁剪和重采样归通用音频处理；MFCC归特征层。** 新增MP3支持只改decoder相关文件；新增P95统计只改stats；新增数据集只加适配与配置。它们都属于数据接入体系，不需要让模型代码知道格式和目录。

顺序有两个实际分支：

- 无标签技术验证：①②③④⑥⑦，得到可检查的特征，不生成分类成绩。
- 正式训练：必须经过⑤，保证人员/来源与标签正确、划分固定。不能把无标签技术验证伪装成已通过数据准入。

## 3. 在宏观仓库上只增补这些文件

不建立另一套项目，不换CLI，不重复实现宏观文档已有模块。

```text
repo/
├── configs/
│   ├── paths.local.yaml
│   ├── datasets/uaspeech_v1.yaml
│   ├── preprocessing/
│   │   ├── uaspeech_native_smoke.yaml
│   │   └── speech_common16k_candidate.yaml
│   └── features/mfcc20_stats.yaml
├── src/sddbench/
│   ├── data/
│   │   ├── scan.py                          # 新增
│   │   ├── decoders.py                      # 新增
│   │   ├── stats.py                         # 新增
│   │   ├── schema.py                        # 沿用并扩展
│   │   ├── audit.py                         # 沿用
│   │   ├── prepare.py                       # 沿用
│   │   ├── audio.py                         # 沿用
│   │   ├── windows.py                       # 沿用
│   │   └── adapters/uaspeech.py             # 新增并注册
│   ├── features/
│   │   ├── mfcc.py                          # 沿用
│   │   └── cache.py                         # 沿用
│   └── reporting/dataset_report.py          # 新增：Markdown＋统计图
├── tests/
│   ├── test_scan_decode.py
│   ├── test_dataset_stats.py
│   ├── test_crop_padding.py
│   └── test_feature_cache.py
└── docs/datasets/uaspeech.md
```

| 文件 | 施工者应具体实现什么 |
| --- | --- |
| `scan.py` | 递归枚举、排序、相对路径、扩展名、文件大小；区分资源旁文件/压缩包/未知；默认不跟随目录符号链接 |
| `decoders.py` | `probe`查真实流信息，`decode`输出统一float32与实际sr；格式选择和失败状态在这里集中管理 |
| `adapters/uaspeech.py` | 从实际命名与元数据提取人员、任务、麦克风和候选标签；不做MFCC、不猜裁剪长度 |
| `audit.py` | 解码有效性、哈希、声学质量告警、来源和标签冲突检查，输出逐文件事实 |
| `stats.py` | 从逐文件事实聚合分位数、分布、分组表与长度候选表；不再重复解码 |
| `dataset_report.py` | 读取stats结果生成报告/图；不另算一套分位数 |
| `audio.py` | 选通道、截取有效区间、重采样；保留有效长度和来源坐标 |
| `windows.py` | 按前/中/后/固定随机生成确定索引；统一短音频和尾片规则 |
| `features/mfcc.py` | 一份固定公式，接收实际sr，不依据dataset_id分支 |
| `features/cache.py` | 指纹、分片保存、元信息、完成标记、复用与失效 |

最小接口即可，不建复杂插件平台：

```python
scan_files(root, scan_config) -> FileInventory
adapter.collect(root, task_config) -> AdapterResult
probe(path, decoder_config) -> AudioInfo
decode(path, decoder_config) -> DecodedAudio   # float32[C,N], sr, backend, warnings
collect_statistics(file_table, group_columns, stats_config) -> dict
build_view_index(records, preprocessing_config) -> ViewTable
extract_mfcc(decoded_view, feature_config) -> FeatureItem
```

具体实现可用dataclass与DataFrame；列名与schema必须统一，禁止仅靠tuple位置传递。`DecodedAudio`的sr是数据真实Hz，不是随手填16000。

## 4. 接入统计：到底统计哪些信息

### 4.1 两档扫描，清楚标注精度

| 模式 | 做什么 | 用途与限制 |
| --- | --- | --- |
| `quick` | 枚举＋头信息/ffprobe；不完整解码所有文件 | 快速认识大库；时长是header/probe估计，不能声称全部可解码 |
| `full` | 全部候选音频完整解码、实际采样点计数、质量摘要、内容哈希 | 本轮UAspeech验收使用；可流式处理，不把全库放内存 |

report必须写 `scan_mode`、是否完整覆盖、扫描时间、工具版本、目录指纹、失败数和未处理数。支持续扫时按内容/文件状态检查，临时中断写partial，不当完成报告。

### 4.2 必需统计表

| 类别 | 统计字段 | 决策用途 |
| --- | --- | --- |
| 文件覆盖 | 总发现数、候选音频数、成功解码数、失败数、未知数、非音频/压缩包/旁文件数 | 避免漏格式或把伪.wav算有效音频 |
| 文件格式 | 扩展名计数＋真实容器/codec/subtype计数，扩展名与内容不符数量 | 决定decoder；不能只写“都是WAV” |
| 音频规格 | 采样率逐值计数与比例、主采样率及占比、通道数、位深/编码精度（适用时） | 判断是否原Hz可直接用，是否需要统一采样率/通道 |
| 时长 | 最小、P10、P25、P50、**P60、P80、P90**、P95、P99、最大、均值、标准差、总小时 | 理解长短分布，比较候选裁剪长度 |
| 时长分布 | 直方图＋累积分布图；按类别/人员/任务/麦克风的分组摘要 | 避免同一个全库P90掩盖类别差异 |
| 人员/标签 | 已确认人数、候选人数、未知ID数，每类人数/录音数，每人录音分布，标签冲突 | 判定能否可靠分组训练 |
| 来源版本 | 精确文件重复簇、原录音/处理版/增强版、疑似同源多麦克风组 | 防止把同一次录音当多个独立样本 |
| 基础信号质量 | 空音频、NaN/Inf、全零、峰值、RMS、均值/DC、近满幅样本比例、极短录音 | 技术异常检查；不自动判疾病或批量删除低音量样本 |
| 数据异常 | 路径不存在、元数据缺行、ID解析失败、标签与元数据冲突 | 明确谁可统计、谁可提特征、谁可训练 |

格式字段要区分：WAV是容器，PCM16/浮点等是编码；MP3通常不报告一个有意义的“原始PCM位深”；未知/不适用填null，不编造16bit。

时长分位数定义：对**完整成功解码的原始音频时长**按录音等权，使用明确固定的quantile算法（如NumPy `method="linear"`）。短音频、损坏音频、选择前/选择后分别列数；一条文件就是一条记录时，不用多声道数乘长时长。quick的估计与full实测不可混为同一表。

另外输出“每人中位时长”的分位数作为补充（仅已确认人员），防止录音特别多的人主导全库分位数。别把这一表当成录音等权分位数。

质量摘要默认逐声道计算，再分别报告；选通道后的可用样本统计另列。近满幅阈值、低能量阈值等写入scan配置与报告，不能用未记录的经验常数自动排除样本。

### 4.3 哪些质量检查先不做

第一轮不做自动疾病判断、自动听感质量分、强制去静音、自动降噪、语音识别转写或假装精确SNR。低RMS、长停顿可能与研究目标有关，只告警。可选“低能量帧比例”使用配置阈值和算法名，明确不是语音活动的真值，也不是SNR。

全零、非有限数据、无法解码可按明确技术规则排除；削波/低音量先人工查。近满幅只是指标，浮点幅度超过1不一定等于ADC削波，告警名称应准确。

### 4.4 给出长度候选表，而不是自动决定10秒

统计阶段可展示候选L=[1,2,4,6,10]秒，并允许自定义。每个L报告：

- 原录音短于L的条数/比例（需要补零）。
- 长于L的条数/比例（单段策略会截断）。
- 单段策略保留的总有效秒数、总截去秒数及保留比例。
- 总补零秒数和补零比例；同时报告“每条补零比例的均值”，两者不要混用。
- 按候选标签/任务/人员组分解；标签未确认时标为directory_group，不写confirmed_class。

计算公式（录音时长为d_i，目标长度L）：有效秒数=sum(min(d_i,L))；补零秒数=sum(max(0,L-d_i))；截去秒数=sum(max(0,d_i-L))；总补零比例=补零秒数/(n×L)。只对有效且已选通道的可用录音计算。

**P90=3秒时，选10秒可能只是多数补零；P90=20秒时，选10秒可能丢掉大量声音。** 报告给事实和候选表，由你决定。不能在没扫描实际UAspeech前宣称10秒合理。

正式研究中用训练部分的时长分布确定/选择参数，避免反复依据测试集表现挑长度。全库描述统计可用于资产审计，但若利用全库分布制定协议应透明说明；最好锁定按人划分后只用train做数据驱动选参。Demo全库探索可先做，但不能因此叫独立确认实验。

## 5. UAspeech适配：先把标签和麦克风关系讲清

示例文件：`Disorder/CF02_B1_C1_M2.wav`。

- 路径中的Disorder只存为 `directory_group`/`label_raw`，不是已批准训练标签。
- 文件首段CF02先作为完整的 `speaker_token` 保存，不能去掉C变成F02；其含义要与本地元数据、原始说明对照。
- B1、C1、M2先保存解析token；只有含义验证后才分别赋予block、词项、mic等正式字段。解析失败明确列出，不随机补ID。
- 若确认M2表示麦克风，单次相同词项的不同麦克风不是多个独立人，也可能不是独立录音来源。保留mic字段，同源音频归同一origin；首个任务优先固定一个双方都有的mic，减少通道重复。
- 单条示例在M2不代表全库只用M2。选择mic前先统计各候选类别/人员的mic覆盖率；缺麦克风的一组不能悄悄被筛光。

本地新增 `label_review.csv`，列：`speaker_token, directory_group, confirmed_subject_id, confirmed_label, evidence, review_status`。适配器读取已确认映射；未确认者可统计/技术提特征，但不得导出为approved分类manifest。若现有标签颠倒，生成新映射和报告，不搬动原文件或整体一键翻转。

官方文献描述UASpeech有多麦克风采集，因此需要显式处理mic来源。[R3] 人数和麦克风具体分布必须以这份本地数据版本实测为准，不照抄论文数。许可与来源记录沿用原文档要求，不将真实音频作为Git测试fixture。

## 6. 不同音频格式：怎样统一到后续流程

第一版：`soundfile/libsndfile`作为首选解码器；不支持或失败时，按明确策略使用`ffprobe/ffmpeg`后备。[R1][R2] 实际MP3支持与部署版本有关，安装后必须跑能力测试；不能说安装soundfile就一定覆盖全部格式。

| 输入情况 | 处理 |
| --- | --- |
| WAV/FLAC等可被首选库正确解码 | 保留真实sr/channels，输出float32，不强制先落盘WAV |
| MP3/M4A/OGG/WebM等 | probe真实流；首选支持则用，否则ffmpeg；记录实际backend/codec/版本 |
| .wav实际损坏或不是音频 | 标decode_failed或extension_mismatch，不仅按后缀算成功 |
| 视频/容器多个音频流 | 明确stream index；未指定不能自动选“第一个看起来像的” |
| ZIP/TAR等 | 记archive；不自动原地解压，不算空库；本阶段按需另提可审核解压方案 |
| DAT/HEA或裸PCM等专用格式 | 标requires_special_decoder；没有配套元数据不能靠猜采样率读取 |

`probe`不改变音频，full模式解码默认也不重采样、不折叠声道、不归一化音量。ffmpeg调用使用参数数组、不经shell拼接路径，限制本地文件、超时和输出；同一源文件固定backend，不能每次随机fallback。后备失败输出错误原因和处理状态。

后续音频层只接收：`waveform float32[C,N] + sample_rate + decoder metadata`。MLP/CNN不需要知道原文件是MP3还是WAV。解码器升级可能改变数值，需更新缓存key和运行环境记录。

不能把有损文件转成WAV后称为无损原始音频；也不能通过重新封装“消除”设备/编码偏差。保留原始codec统计用于解释潜在混杂。

## 7. 原始Hz要不要保持：分探索与正式主协议

### 7.1 当前技术验证可以不重采样

本轮提供 `sample_rate_policy: native`。如果实际选中样本均为相同sr，直接在该Hz算MFCC，少一个处理变量，完全合理。

但“多数是16k就不管剩下8k/44.1k”不是一个完整规则。即使20个MFCC都变成40维，频带和时间分析仍可能不同。

### 7.2 两种明确模式

| 模式 | 行为 | 适用 |
| --- | --- | --- |
| native | 保持每条真实Hz；按毫秒推导帧长/步长/FFT；记录每条实际参数 | 统计、单库技术探索，或事先定义的原Hz实验 |
| resample | 全部按固定算法重采样到target_sample_rate | 最终共同主协议候选，初始考虑16k，但需看统计再确认 |

native遇到混合Hz：默认 `allow_mixed_native: false`，特征plan报出分布并停止批量提取；你可明确选择统一重采样，或开启mixed-native探索。**“停止”的是待定特征批处理，不是统计。** 混合模式需按Hz建对应前端并逐条保存参数；禁止拿16000参数去解释44100波形。

frame=25ms、hop=10ms：`win_length=round(sr*0.025)`，`hop_length=round(sr*0.010)`；`n_fft=next_power_of_two(win_length)`。具体取整方式写定（非负值half-up），hop/win至少1。16k时为400/160/512；8k为200/80/256；44.1k时FFT应随win增大，不会仍固定512。

频率上限有两种声明：native探索可用每条Nyquist，明示频带不一致；做严格共同前端时固定共同f_max且不高于所有输入Nyquist，或统一采样接口并单列原8k有效带宽限制。8k升到16k不恢复高频。最终主参数应固定，不能每个库自动选自己的最优Hz后当作只比较模型。

### 7.3 两份预处理配置的具体内容

以下2秒只用于D3小样技术验证，**不是已选定UAspeech主长度**。D1报告后由你改为认可的长度，变更留记录。

```yaml
# configs/preprocessing/uaspeech_native_smoke.yaml
schema_version: "1.0"
profile: technical_only
sample_rate_policy: native
allow_mixed_native: false
target_sample_rate: null
channel_policy: select
channel_index: 0                     # D1确认声道后使用，不代表已经核实
view_mode: single_segment
observe_seconds: 2.0
window_seconds: 2.0
position: head
crop_seed: 0
short_policy: right_zero_pad
amplitude_normalization: none
```

第二份 `speech_common16k_candidate.yaml` 复制上述字段，改为：`sample_rate_policy: resample`、`target_sample_rate: 16000`；profile仍为technical_only，长度仍是待验收候选。重采样统一使用锁定版本torchaudio的`sinc_interp_hann`，展开保存filter宽度、rolloff等全部实际参数；本轮不以一个未展开的默认字符串声称完整可复现。两份配置不要同时再保留宏观旧字段`audio.target_sample_rate`作为独立真相：由config.py归一成一套最终audio配置，再存入job/run。

以上channel_index是解码后通道索引；文件名M2若为独立麦克风文件，与该文件内部channel_index不是同一个概念。两者分别记录，不要把M2解释为“取第二个声道”。

## 8. 裁剪、补零、随机位置：精确定义到能写代码

### 8.1 区分观察长度与模型窗口

`observe_seconds`是每条录音取多少秒；`window_seconds`是送入一次特征/模型的窗口大小。第一轮为了简单，设二者相同，单段模式每条录音输出一个窗口。之后才测试“整条录音分成多个窗”。

```yaml
view_mode: single_segment
observe_seconds: 10.0          # 这里只是演示配置，不是UAspeech已选主长度
window_seconds: 10.0
position: head                # head / center / tail / fixed_random
short_policy: right_zero_pad
```

不足10秒：保留全部真实声音，右侧补零至目标波形长度，不循环、不复制、不拉伸，也不跨不同录音拼接。

### 8.2 在原始采样点坐标上选择

令原录音采样点数N、采样率s、目标长度样本数K=round_half_up(L*s)，D=max(N-K,0)。

| 位置 | 起始采样点start |
| --- | --- |
| head | 0 |
| center | floor(D/2) |
| tail | D |
| fixed_random | 在整数[0,D]中均匀抽取一次 |

end=min(start+K,N)。N<K时所有位置都取[0,N]；补零不能伪造“尾部只剩零”的截取。原采样坐标和秒数都保存，避免多次浮点转换漂移。

fixed_random种子由 `global_seed + 稳定sample_id + crop_algorithm_version` 的SHA256派生固定64位整数，再用锁定版本的明确RNG（如NumPy PCG64）抽样。不能用Python进程随机化hash，不能依赖文件扫描顺序。保存实际start/end；相同配置重跑和换文件顺序后相同。改变seed允许改变裁剪，但不保证每条都一定变化。

第一版不做每epoch重新随机截取。以后若要在线随机增强，单独模式、训练端种子与epoch绑定；验证/测试仍固定，缓存政策也需另定。不要把fixed_random实现成每次调用都抽一次。

### 8.3 处理顺序与补零mask

1. 解码，确认实际sr和通道；按配置选择通道。
2. 在原始采样点上确定start/end，截出有效声音。
3. 必要时重采样有效声音，记录算法、预期/实际点数；对重采样长度取整作明确校验。
4. 生成目标长度的波形与 `valid_num_samples`，不足部分右补零；记录padding ratio。
5. **计算MFCC统计量时只使用有效声音对应的合法分析帧，不把凑长度的零段算入均值/标准差。**

短音频补零是张量容纳策略，不代表让MFCC统计“学习几秒假静音”。最容易实现的做法：保留未补零的有效数组计算MFCC，另保留固定长度wave与mask供未来波形/CNN使用。MFCC帧需要batch对齐时再在特征层补零，同时传frame_mask。

对于极短到不够一个分析帧的有效声音，第一版返回 `too_short_for_frontend`，排除并报告，不擅自改变帧长。整库按统一阈值处理；不能每个库写不同的秘密补救。窗边界的STFT padding与凑10秒补零是两件事，配置和记录应区分。

`windows.csv`至少保存：sample_id、window_id、原sr、原start/end样本与秒、处理后sr、有效点数、目标点数、补零点数、position、seed、声道策略、解码backend、view_hash。

### 8.4 以后多窗模式的约束

沿用宏观文档：不重叠stride=window；保留足够长尾窗并mask，小于明确阈值的尾片排除且记录丢失秒数。单段模式中的短录音不直接套“尾片排除”：它是完整录音，按前端最小长度决定可用性。两种模式必须明确区分。

## 9. MFCC中间产物：具体是什么、怎样保存、怎样用于模型

### 9.1 本轮先实现一个明确配方

```yaml
# configs/features/mfcc20_stats.yaml
name: mfcc_stats
implementation: torchaudio
n_mfcc: 20
n_mels: 64
frame_ms: 25.0
hop_ms: 10.0
n_fft_policy: next_power_of_two_window
window: hann_periodic
center: false
power: 2.0
f_min_hz: 0.0
f_max_policy: nyquist           # native探索；主协议需显式锁定
mel_scale: htk
mel_norm: null
log_mels: true
# log的epsilon/加法方式等必须按锁定实现展开保存，不能另造一套却同名
# torchaudio版本变化后需验证语义并更新特征实现版本
log_implementation: torchaudio_native
log_epsilon: from_locked_implementation
dct_type: 2
dct_norm: ortho
include_c0: true
stats: [mean, std]
std_ddof: 0
save_frames: false
```

选择log-Mel MFCC以避开默认dB缩放对输入最大值的额外依赖；torchaudio文档明确提示默认实现切片与整条输入可能得到不同尺度。[R4] 不要求“局部片段”和“整条音频”边界帧完全相同，只需规则固定、参数明确。服务器锁定安装版本后，解析log细节为具体值并写meta；禁止把 `from_locked_implementation` 原样留在最终配置冒充完整参数。

建议直接使用同一版本的torchaudio MFCC，配置完整melkwargs。native模式对每个sr创建/复用对应transform；帧数依该实现的 `center=false` 与n_fft实际支持区间推导，不只按win_length估算。有效声音不足n_fft时判too_short，不悄悄让零段进入统计。

### 9.2 逐条计算过程

```text
一条文件
→ 解码得到原始波形与真实Hz
→ 按view索引截取/选通道/必要重采样
→ 有效声音得到MFCC [20,T_valid]
→ 每个系数在有效帧上算mean和std
→ 按[mean_0…mean_19,std_0…std_19]拼成[40]
→ 写入数值分片，保留window_id与元信息
```

模型训练时：从split确定train/val/test窗口 → 读取对应40维向量 → 用train拟合Scaler → 所有集合使用同一Scaler → MLP输入40维。改n_mfcc=40则输出80维，模型自动推导输入维度。

保存40维统计向量是因为本轮MLP只用它；没必要默认保存所有帧。以后要改成其他帧统计，可开启save_frames或重算；不能在不保存帧的前提下承诺任意新统计都无需重算。缓存里没有经过跨样本拟合的Scaler。

### 9.3 一个可直接检查的特征目录

```text
artifacts/cache/features/mfcc_stats/<feature_key>/
├── meta.json                 # source/view/front-end指纹、全部参数、版本、完成状态
├── index.csv                 # window_id → shard文件、row_index、有效帧数
└── shard_000.npz              # window_ids[N]、x[N,40]；float32，无pickle对象
```

`index.csv`不复制训练标签作唯一依据；训练始终与固定manifest/split按ID关联。检查feature覆盖率和唯一性。大库分片避免几十万个小文件；NPZ加载需考虑内存，未来确有瓶颈再改存储格式，不先引入数据库。

save_frames=true时允许另存帧分片与offsets/mask说明，不能把变长帧偷偷塞成object数组要求allow_pickle。小批检查导出不超过明确数量的MFCC图，PNG不作训练输入。

一次提特征的记录还要单独保存：所用inventory/prep_id、view配置、feature配置、开始/结束时间、代码commit/源码快照、实际成功/失败样本、输出key与位置。它与训练run不是同一种产物，但同样可追溯。

## 10. 中间产物和统计报告存在哪里

```text
artifacts/
├── inventory/uaspeech/<scan_id>/
│   ├── files.csv               # 每个发现文件的类型、probe/full状态和基础事实
│   ├── statistics.json         # 结构化分布/分位数/范围/覆盖率
│   ├── group_statistics.csv    # directory_group/confirmed_label/任务/mic等分组
│   ├── length_candidates.csv   # 不同L的补零/截断代价
│   ├── issues.csv
│   ├── report.md
│   └── figures/                # 时长直方图/CDF、Hz/格式分布等
├── prepared/uaspeech_v1/...    # 核验标签后才生成approved任务清单，沿用宏观结构
├── splits/uaspeech_v1/...      # 固定人员/来源分组
├── feature_jobs/<job_id>/
│   ├── job.json
│   ├── config.yaml
│   ├── windows.csv
│   ├── issues.csv
│   ├── inspection.md
│   ├── figures/
│   └── cache/                  # 本轮run模式的局部特征
├── cache/features/...          # shared特征缓存
└── runs/...                    # 训练结果，沿用宏观结构
```

为了同时支持“一个小样先看看”和“全库预计算”，本补充明确：数据层独立提取命令用 `feature_jobs/<job_id>`，可在没有训练run时运行。cache.mode=run在这种命令里指当前feature_job局部cache；由训练run触发时仍指该run的cache。mode=shared才放公共cache。

一次性十几种MFCC参数用局部cache；同一前端反复扫模型/seed用shared。过期特征可显式清理，但inventory、prepare、split、feature_job配置/窗口索引和正式run不可作为缓存顺手删除。

## 11. 后续补统计字段，为什么不会影响模型层

`statistics.json`建议结构：

```json
{
  "schema_version": "1.0",
  "scan_id": "由程序生成",
  "status": "partial_or_completed",
  "scope": {"mode": "full", "dataset_id": "uaspeech"},
  "counts": {},
  "formats": {},
  "sample_rates": {},
  "duration_seconds": {},
  "groups": {},
  "quality": {},
  "extensions": {}
}
```

示例未填实测数字，不是运行输出。新增P98或新格式占比，往stats/extension加字段，展示按schema支持，旧模型根本不读取statistics.json。可选附加列不改变manifest必需列语义；下游应明确选取需要的字段，而不是假设CSV列顺序或禁止所有附加列。

分开三个指纹：`source_fingerprint`识别实际选中原件；`view_hash`识别Hz/通道/裁剪等处理；`feature_hash`识别MFCC公式与实现。新增报告图表/分位数不应让MFCC全失效；改变MFCC或裁剪才失效。不能简单以整个Git commit作为唯一特征key，否则改README也重算；记录Git用于追溯，缓存key使用受影响数据/处理代码的版本或摘要。

如果新增统计只是观察，改stats即可；如果发现格式原来漏扫，新增原件使任务集合变化，就应生成新清单与相应缓存，不能打着“兼容”旗号静默改变历史实验。分层保障可控变化，不是承诺任何变更都不影响结果。

## 12. 命令与页面：用户最终怎样操作

以下是施工目标命令，尚未实现；所有路径从本地配置绑定。`SOURCE`、`SCAN_ID`等为实际输出标识替换位，不得自动选latest。

```bash
cd /mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109/repo

# A. 指定示例只检查，不把它误认为全数据集
sddbench inspect-file --dataset uaspeech --relative-path Disorder/CF02_B1_C1_M2.wav --paths configs/paths.local.yaml

# B. 全库完整统计，不需要训练标签全部批准
sddbench stats --dataset uaspeech --mode full --paths configs/paths.local.yaml

# C. 看已保存统计；无需再次解码
sddbench dataset-report --dataset uaspeech --scan SCAN_ID --paths configs/paths.local.yaml

# D. 技术提特征：先少量确定性样本；source引用已扫描对象，不叫approved训练数据
sddbench extract-features --dataset uaspeech --source SCAN_ID --preprocess configs/preprocessing/uaspeech_native_smoke.yaml --features configs/features/mfcc20_stats.yaml --limit 12 --cache-mode run --paths configs/paths.local.yaml

# E. 经人工选参数后先预检，再全量提取；移除limit必须明确
sddbench extract-features --dataset uaspeech --source SOURCE_ID --preprocess configs/preprocessing/speech_common16k_candidate.yaml --features configs/features/mfcc20_stats.yaml --dry-run --paths configs/paths.local.yaml
```

`--source`须识别明确类型scan或prepared：前者只产生 `technical_only=true` 的特征job；后者绑定已准入清单，才可给训练使用。抽12条要按已知目录组/人员分布确定性选取，并输出精确样本列表，不只取排序前12条同一个人。

所有命令打印：dataset、模式、输入根、扫描/样本范围、输出绝对路径、ID、成功/失败数与下一步。缺少配置、标签冲突、解码错误有不同状态；不为了得到绿色success吞掉问题。

未来页面增加“数据集统计”页，复用同一后端函数：选择数据集 → 显示已有scan列表 → 选择quick/full后点统计 → 显示报告与长度候选表 → 人工保存参数配置 → 小样特征预览。不能一选数据集就无提示全库重扫。页面统计可实现本机受限后台job与进度，第一版CLI即可；不扩展成任意命令执行器。仍不在这里启动训练。

## 13. 本次施工拆成四个停点

宏观阶段0可先完成；本补充分解其阶段1、2的数据部分。每个停点提交实际文件清单、命令输出、产物绝对路径、问题和人工检查项，**停止等验收，不一次跨完四步**。

### D1：单文件与全库统计

实现路径绑定、scan、decoder、UAspeech候选解析、audit、stats、报告。单文件先检查用户给出的CF02示例，再full全库；不做裁剪、MFCC或训练。

验收：

- 能展示示例真实容器、codec、Hz、通道、时长与解码backend；打不开则如实报告。
- 全库数量口径自洽，所有发现文件有状态；格式大小写、伪.wav、压缩包和未识别格式不漏报。
- 有P60/P80/P90、主Hz占比、格式分布、长度候选代价；分组明确是目录组还是已确认标签。
- issues列出标签/ID/mic疑问；不把CF02直接当已确认病人。

**你看完统计后再决定长度、Hz和选用mic。这是第一次人工决策点。**

### D2：只验证裁剪/补零与索引

实现head/center/tail/fixed_random、native/resample、统一窗口索引和mask。暂不MFCC。

验收：

- 20秒示例取10秒：head起0、center起5、tail起10；随机位置在合法范围。
- 3秒示例取10秒：有效3秒、补零7秒；四个位置不丢原声音。
- 相同seed重跑/更换文件枚举顺序索引一致；不同seed结果可追溯。
- native不改变实际Hz；resample按规则输出正确长度，原文件哈希不变。
- 多声道按策略处理，未知声道语义不自动平均；view记录来源坐标与有效长度。

用合成fixture验证精确坐标，并对少量真实音频导出可审听的派生片段到feature_job检查目录，原始数据不变。真实派生音频不提交Git。

### D3：MFCC小样、缓存与模型输入验收

实现固定配方、40维统计、meta/index/shard和feature_job记录；先12条技术验证。标签未核实不阻止这一步，但输出technical_only。

验收：

- x形状[N,40]，所有值有限，ID与窗口一一对应；n_mfcc改40则80维。
- 凑长度补零不改变同一有效声音的MFCC统计；真正改截取内容允许改变特征。
- 混合native默认明确报待决策；毫秒到采样点、n_fft随Hz正确推导。
- 相同源/视图/特征命中cache；改n_mfcc或位置失效；只改统计图/P95不失效。
- 删除特征cache可用记录重算；数值在锁定环境声明容差内一致。

**通过后才考虑全量提取。** 若多数短词被补成10秒，回到D1候选表讨论主长度，不为固定10秒硬上。

### D4：标签/来源复核与第二库接入

单独完成label_review、mic/origin、人员隔离规则；产出approved manifest。不是在D3提完特征后顺手按Disorder/Healthy生成标签。准入通过后才调用宏观文档的split与模型训练。

再接第二库验证可扩展：新增Adapter与配置，同一decoder/statistics/view/MFCC代码应复用。无合适第二库可先报告候选，不阻塞已经完成的UAspeech技术功能，也不降低准入要求。

## 14. 可直接发送给Codex的本轮任务

```text
请阅读宏观施工文档和《数据层补充设计与施工方案_UAspeech先行_v1.md》。
本轮仅做补充文档D1：单文件检查与UAspeech全库统计。
宏观与本补充冲突时，本补充的真实路径、首个数据集和数据层安排优先。

施工根：
/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109
代码：施工根/repo
运行产物：施工根/artifacts
只读数据根：
/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/UAspeech
先检查：Disorder/CF02_B1_C1_M2.wav

如果宏观阶段0未完成，先完成D1所需最小安装/CLI/路径基础，再做D1。
不要训练、不要切片/重采样落盘、不要全量提MFCC，不自动进入D2。
不要修改、移动、翻转标签或删除原始文件。
不要因Disorder目录名就认定CF02有疾病；标签/人员/mic作为待核查事实报告。
实现通用scan/decoder/stats，Adapter只处理本库元数据与命名关系。
全库报告必须有真实格式、采样率分布、P60/P80/P90及长度候选表，
明确quick/full与覆盖率；非音频、压缩包和失败要有去向。

完成后提交：修改文件及用途、实际执行命令、报告绝对路径、统计摘要、
阻断问题、我需要人工核查/选择的事项以及建议验收步骤。
到此停止，等待验收；不自动commit/push。
```

## 15. 参考依据与采用边界

这些来源用于确认具体工具行为和语料采集结构，不替代本地实测，也不规定本研究的最优长度/Hz。

- [R1] [python-soundfile官方文档](https://python-soundfile.readthedocs.io/en/latest/)：读取格式、真实sr/通道和解码能力。部署时记录实际soundfile/libsndfile版本并测试。
- [R2] [ffprobe官方文档](https://ffmpeg.org/ffprobe.html)：容器、流与机器可读probe信息。
- [R3] [Kim等，Dysarthric Speech Database for Universal Access Research，Interspeech 2008](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.pdf)：多麦克风采集与语料结构；本地人员和标签仍须元数据核验。
- [R4] [torchaudio MFCC官方文档](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MFCC.html)：参数与默认dB缩放注意事项；实现锁定实际发布版，不安装nightly来追随网页。

**本次先验收“真实统计”，再确定裁剪与Hz；再验收“窗口与MFCC”，最后才进入模型。统计项可以扩展，数据/特征接口保持稳定，模型层无需为新增统计或音频格式另写分支。**
