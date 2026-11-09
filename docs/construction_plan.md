# 语音疾病基准平台：仓库结构与分阶段施工说明

版本：v1.0｜日期：2026-11-09｜用途：新建 Git 仓库、服务器端 Codex 逐阶段实施、人工验收与论文实验追溯。

**本说明替代上一版 HTML 的工程施工部分。它是一份待实现的设计，不代表代码已经存在，也不代表数据标签已通过核验。**

## 1. 先确定我们到底要建什么

建一个命令行驱动的分类实验平台：为不同数据集编写适配器，把它们转换成同一种清单；后面共用音频处理、特征、训练、评价和报告代码。前端先做读取结果的展示页。

阅读顺序：先看第3、4节确定文件位置，再看第12节逐阶段施工；第5—10节是实现时查阅的规则。

第一版只研究人类语音，包括说话声、朗读、访谈、持续元音等发声任务。咳嗽、呼吸和听诊音以后扩展，不占用本次施工时间。**每个数据集的任务分别训练模型，不把不同疾病库混成一套“健康/患病”标签，也不要求一个模型同时诊断所有疾病。**

先接入两个候选：

| 顺序 | 本地目录线索 | 平台名称 | 第一个任务 | 开始训练的前提 |
| --- | --- | --- | --- | --- |
| 1 | `Parkinson_KCL_2017`，实际路径由你填写 | `mdvr_kcl` | 只用 ReadText，HC/PD 二分类 | 核对原始说明、HC/PD含义、人员ID和重复文件 |
| 2 | `EATD`，实际路径由你填写 | `eatd` | 抑郁相关二分类；录音任务按确认后的规则选取 | 核对评分/原始标签、二分类规则、人员ID，只选一种可追溯版本 |

这两个名称是适配候选，不是已完成身份认证。第一阶段若发现 MDVR 标签或人员关系不可靠，应停在数据验收处，不能为了跑通而按文件随机划分。EATD 的 positive/negative/neutral 可能描述采集的表达任务，不能直接当抑郁类别；具体以元数据和原始说明核定。

**第一轮闭环：MDVR → MFCC统计特征 → MLP → 标准结果目录。第二轮：只增加 EATD 适配与任务配置，验证下游不用改。** CNN、多参数批量实验和展示界面随后加入。

## 2. 先分清两个根目录

代码仓库与大体积实验工作区分开。下面都是示例路径，不能让 Codex 原样当作真实服务器路径。

| 路径 | 放什么 | Git 是否管理 |
| --- | --- | --- |
| `/path/to/speech-disease-benchmark/` | Python代码、配置模板、测试、施工与实验说明 | 是 |
| `/path/to/speechbench-workspace/` | 清单、划分、缓存、完整实验结果、模型权重 | 否；独立备份 |
| 你现有的原始数据目录 | 已下载的原始数据集 | 否；只读使用 |

不搬动、不重命名、不覆盖你的原始数据。不要求把音频拷进新仓库。换服务器时只修改本地路径绑定；数据清单使用相对于各数据根的路径。

## 3. Git 仓库目录：按这个位置放文件

以下是第一版完成后的目标结构。**按阶段创建实际需要的文件，不要第一天生成一堆空壳。** `__init__.py` 等常规包文件省略。后面的表逐项解释用途。

```text
speech-disease-benchmark/
├── README.md
├── pyproject.toml
├── requirements.lock.txt
├── .gitignore
├── configs/
│   ├── paths.example.yaml
│   ├── paths.local.yaml                    # 本机文件，不提交
│   ├── datasets/
│   │   ├── mdvr_kcl_read_v1.yaml
│   │   └── eatd_depression_v1.yaml
│   ├── experiments/
│   │   ├── mdvr_mfcc_mlp.yaml
│   │   ├── eatd_mfcc_mlp.yaml
│   │   └── mdvr_logmel_cnn.yaml
│   └── sweeps/
│       └── mdvr_dropout.yaml
├── src/sddbench/
│   ├── cli.py
│   ├── config.py
│   ├── run.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── prepare.py
│   │   ├── audit.py
│   │   ├── split.py
│   │   ├── audio.py
│   │   ├── windows.py
│   │   └── adapters/
│   │       ├── base.py
│   │       ├── registry.py
│   │       ├── mdvr_kcl.py
│   │       └── eatd.py
│   ├── features/
│   │   ├── registry.py
│   │   ├── mfcc.py
│   │   ├── logmel.py
│   │   └── cache.py
│   ├── models/
│   │   ├── registry.py
│   │   ├── mlp.py
│   │   └── cnn.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── aggregate.py
│   │   └── metrics.py
│   └── reporting/
│       ├── export.py
│       ├── summarize.py
│       └── dashboard.py
├── tests/
│   ├── fixtures/                          # 人工合成小音频/小清单，无真实患者录音
│   ├── test_adapters.py
│   ├── test_split.py
│   ├── test_audio_features.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── docs/
│   ├── construction_plan.md               # 本说明
│   ├── acceptance_log.md                  # 每阶段人工验收
│   ├── experiment_log.md                  # 正式实验问题与结论索引
│   └── datasets/
│       ├── mdvr_kcl.md
│       └── eatd.md
└── results_summary/
    ├── README.md
    └── ...                               # 人工选中的去标识化小结果，按run_id保存
```

### 3.1 顶层、配置与文档

| 文件/目录 | 具体职责 |
| --- | --- |
| `README.md` | 安装、填写路径、运行一个任务、新增数据集、查找结果；只写已经可用的命令 |
| `pyproject.toml` | 包与CLI入口、依赖声明、测试设置；统一命令名为 `sddbench` |
| `requirements.lock.txt` | 实际验证环境的精确依赖版本；注明Python与PyTorch/CUDA安装来源，禁止只写一组浮动依赖就声称可复现 |
| `.gitignore` | 排除本地路径、音频、权重、缓存、患者级结果、密钥和日志；不排除应提交的代码与实验配置 |
| `configs/paths.example.yaml` | 可提交的占位路径模板 |
| `configs/paths.local.yaml` | 原始数据和工作区的真实绝对路径；服务器自行填写 |
| `configs/datasets/*.yaml` | 一个数据任务的固定定义：适配器、数据根、录音任务/版本选择、标签映射、类别顺序、评价单位 |
| `configs/experiments/*.yaml` | 一次实验的可调参数：划分ID、Hz、时长、位置、特征、模型、训练、缓存策略 |
| `configs/sweeps/*.yaml` | 批量覆盖哪些参数、有哪些取值；每个组合仍生成独立run目录 |
| `docs/datasets/*.md` | 来源、许可引用信息、人员规则、标签依据、已排除内容、人工抽查记录；不记真实姓名 |
| `docs/acceptance_log.md` | 哪一阶段、验收了什么、通过/未通过、问题和对应Git提交 |
| `docs/experiment_log.md` | 实验目的、实验组ID、run_id、结果位置、是否纳入论文、简短结论 |
| `results_summary/` | 有意选入Git的小型汇总：配置、总体指标、图片、运行说明；不默认上传逐人预测 |

先用普通 YAML + Python 类型验证即可，不需要引入复杂的配置组合系统、数据库、任务队列或训练框架。建议使用 PyTorch、torchaudio、scikit-learn、pandas、PyYAML、matplotlib、pytest；具体兼容版本由服务器实际安装与验证后锁定。第一版选择一种MFCC/梅尔实现并固定，不能混用两个库的默认参数。

### 3.2 数据层：只有这里了解各个数据集长什么样

| 文件 | 具体职责 |
| --- | --- |
| `data/adapters/base.py` | 定义最小适配器接口：输入数据根和任务配置，输出标准记录、排除项和问题；不训练、不划分、不算特征 |
| `data/adapters/registry.py` | 明确字典：`mdvr_kcl → MDVRAdapter`、`eatd → EATDAdapter`；新增库只加一个注册项 |
| `data/adapters/mdvr_kcl.py` | 识别ReadText、HC/PD、人员ID和原始录音；不读取旧整理目录来猜标签 |
| `data/adapters/eatd.py` | 关联人员、录音和评分/标签；识别原始与处理版本；不把每人的多条录音当独立人 |
| `data/schema.py` | 定义清单必需列与合法值；检查ID、标签、路径、人员归组是否满足要求 |
| `data/prepare.py` | 调适配器、调用审计、生成不可覆盖的准备版本；是prepare命令的入口 |
| `data/audit.py` | 解码、采样率/时长/通道统计、文件哈希、重复与标签冲突检查、记录排除原因 |
| `data/split.py` | 从合格清单产生按人分组的train/val/test；保存划分；检查同人和同源录音不跨集合 |
| `data/audio.py` | 按清单定位原件，读取、选通道、裁剪、重采样；没有“如果是EATD就……”的分支 |
| `data/windows.py` | 把观察时段变成固定窗口索引；记录每个窗口对应原件的起止秒数和有效长度 |

### 3.3 特征层：音频处理参数在这里统一生效

| 文件 | 具体职责 |
| --- | --- |
| `features/mfcc.py` | 从窗口得到MFCC帧，按有效帧计算均值和标准差，输出固定向量 |
| `features/logmel.py` | 输出数值log-Mel时频矩阵和有效帧mask，供CNN使用；不生成PNG作为训练数据 |
| `features/registry.py` | 按 `mfcc_stats` / `logmel` 选特征提取器；检查与模型是否兼容 |
| `features/cache.py` | 决定复用还是计算、生成缓存键、写完整性标记；不负责改变实验参数 |

**特征缓存和模型训练后的标准化不是同一回事。** MFCC等不拟合总体统计的特征可提前缓存；StandardScaler只在本次训练集上拟合，存进本次run，验证/测试只使用该变换。不能先对全库标准化再划分。

### 3.4 模型、训练、评价与展示

| 文件 | 具体职责 |
| --- | --- |
| `models/mlp.py` | 只有网络结构：输入维度、隐藏层、dropout、输出类别数；不知道数据目录 |
| `models/cnn.py` | 只有CNN结构：输入通道、卷积宽度、池化、dropout、分类头；按有效帧处理补零 |
| `models/registry.py` | 按配置选模型，推导合法输入尺寸；输出维度统一为类别数C |
| `training/dataset.py` | 把标准窗口/特征组成batch，保留window_id、sample_id和有效长度 |
| `training/trainer.py` | 一个通用PyTorch训练循环；优化器、早停、checkpoint、学习曲线；不写数据集分支 |
| `training/predict.py` | 加载选定checkpoint，在完整目标集合输出按类别排序的概率 |
| `evaluation/aggregate.py` | 窗口概率聚合成录音概率，再聚合成人员概率；没有各库私有公式 |
| `evaluation/metrics.py` | 统一计算分类指标、缺失值原因和曲线；独立于网络 |
| `reporting/export.py` | 给一个run生成指标表、混淆矩阵、学习曲线与人可读报告 |
| `reporting/summarize.py` | 扫描完整结果目录，汇成总表；可重新生成，不是唯一结果来源 |
| `reporting/dashboard.py` | 用Streamlit做只读展示：过滤数据集/模型/实验组、查看配置、指标、图；初版不从网页启动训练 |
| `config.py` | 读配置、解析本地路径、校验未知字段、输出完整配置；禁止拼错字段后静默忽略 |
| `run.py` | 串起统一流程，并创建/更新本次结果目录；失败也保留配置与错误记录 |
| `cli.py` | 参数解析与命令分发；不把实际算法全写在这里 |

## 4. 服务器工作区目录：中间数据和结果都放哪里

```text
speechbench-workspace/
├── prepared/
│   ├── mdvr_kcl_read_v1/
│   │   └── prep_<fingerprint>/
│   │       ├── manifest.csv
│   │       ├── excluded.csv
│   │       ├── issues.csv
│   │       ├── audit.json
│   │       ├── task.yaml
│   │       └── provenance.json
│   └── eatd_depression_v1/...
├── splits/
│   └── mdvr_kcl_read_v1/
│       └── demo_group_s0_<fingerprint>/
│           ├── split.csv
│           └── split_meta.json
├── cache/
│   ├── features/
│   │   ├── mfcc_stats/<cache_key>/
│   │   │   ├── index.csv
│   │   │   ├── shard_000.npz
│   │   │   └── meta.json
│   │   └── logmel/<cache_key>/...
│   └── tmp/                               # 写入尚未完成的临时文件
├── runs/
│   └── 20261109T143000+0800_mdvr_mlp_s0_a13f92/
│       ├── run.json
│       ├── config.yaml
│       ├── environment.txt
│       ├── requirements.lock.txt
│       ├── command.txt
│       ├── inputs/
│       │   ├── manifest.csv
│       │   ├── split.csv
│       │   ├── task.yaml
│       │   ├── audit.json
│       │   ├── provenance.json
│       │   └── split_meta.json
│       ├── windows.csv
│       ├── transforms.joblib
│       ├── checkpoints/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── history.csv
│       ├── logs.txt
│       ├── predictions_windows.csv
│       ├── predictions_recordings.csv
│       ├── predictions_subjects.csv
│       ├── metrics.json
│       ├── metrics_per_class.csv
│       ├── figures/
│       │   ├── confusion_matrix.png
│       │   ├── learning_curve.png
│       │   ├── roc.png
│       │   └── pr.png
│       ├── report.md
│       └── cache/                         # 仅cache.mode=run时存在
├── sweeps/
│   └── <sweep_id>/
│       ├── sweep.yaml
│       └── jobs.csv
└── summaries/
    ├── runs.csv
    └── comparison.csv
```

这里的 `<fingerprint>` 等是程序生成的指纹占位符，不是要求人工命名。运行文件夹的时间包含时区；程序还应附加短随机后缀以避免同秒并发冲突。以上时间仅是格式示例。

关键区别：

- `prepared/`：这批原件怎样变成任务样本，是小型、应保留的数据依据，不是音频副本。
- `splits/`：谁在训练、验证、测试，固定后重复使用，不随模型参数改变。
- `cache/`：计算加速用品，可以删除；删掉之后只应变慢，不应改变任务或计算口径。
- `runs/`：不可混淆的实验事实，原则上保留并备份，不能当缓存清理。
- `summaries/`：从runs汇出来的索引，可删除重建；不能只剩这张总表。

## 5. 新数据集怎样接进来：只认一张标准清单

### 5.1 `manifest.csv` 每行是什么

第一版每行表示一条被选中的原始录音。适配器先解释来源，统一审计器补齐音频属性，最后写出以下字段。**类别标签来自已核定的任务规则，而不是把每个子文件夹自动当一类。**

| 字段 | 含义与规则 |
| --- | --- |
| `sample_id` | 该任务内唯一稳定录音ID；不能用本次扫描序号 |
| `relative_path` | 相对本库数据根的路径；不写服务器绝对路径 |
| `subject_id` | 有依据的匿名人员ID；不能从每个文件名硬造一个“人员ID” |
| `group_id` | 划分隔离组；同人、多次录音、同源版本必须归到同组 |
| `origin_id` | 同一次原始录音的版本/切片共同来源；无法确定时记问题，不假装已排除重复 |
| `label_raw` | 原始组别/评分；保留可审计值 |
| `label` | 任务规定的规范标签，例如 `control`、`parkinson` |
| `variant` | `original` / `processed` / `augmented` / `unknown` |
| `elicitation` | 朗读、访谈、元音等采集任务 |
| `sample_rate` | 解码得到的原始采样率 |
| `duration_sec` | 解码得到的原始时长 |
| `channels` | 原始通道数 |
| `sha256` | 选中原文件的内容哈希；准备时计算一次并保存 |
| `official_split` | 有官方划分时保留；没有就留空 |
| `label_source` | 原始元数据文件/行号或说明依据，不能只写“AI判断” |

数据集/任务/版本等公共信息放在旁边的 `task.yaml` 与 `provenance.json`，不在每一行重复长文本。第一版只接受**整条录音可以作为合格目标人输入**的任务；需要复杂说话人分离的访谈先不准入，之后再扩展区间字段和数据读取器。

如果数据的真实性质决定“标签随会话变化”，人员依旧是划分单位，但评价单位要改为人员-会话。第一版先完成同一人在任务内标签一致的分类任务，不能把相互矛盾的标签平均掉。

### 5.2 适配器的最小接口

```python
class DatasetAdapter:
    def collect(self, root: Path, task_config: dict) -> AdapterResult:
        """返回 records、excluded、issues；不修改root中的任何文件。"""
```

`records` 是待审计的标准记录列表。`excluded` 必须含路径和原因；`issues` 含问题、严重级别、涉及记录和待核查依据。统一 `prepare.py` 完成类型检查、解码审计、哈希与落盘。不要要求每个适配器各写一套审计。

MDVR适配器只负责回答：哪个目录是ReadText、HC/PD代表什么、哪个ID属于同一个人、选哪个录音版本。EATD适配器只负责回答：哪条录音属于谁、原始评分/标签在哪里、哪些是处理副本。**二者都不决定16000Hz，也不决定MLP宽度。**

### 5.3 添加第三个同类数据集时，应只增加什么

1. `data/adapters/new_dataset.py`：识别原始结构。
2. `adapters/registry.py`：增加一条注册。
3. `configs/datasets/new_task_v1.yaml`：标签、筛选与评价定义。
4. `docs/datasets/new_dataset.md` 和适配器抽查/测试记录。
5. 本机 `paths.local.yaml`：增加数据根。
6. 复制一个实验配置，改任务与已验收划分ID。

随后运行prepare、split、run，训练器、特征公式、指标代码都不改。这是“只加Adapter”的准确含义：**增加数据解释代码与声明配置，不另写训练和评估脚本。** 完全新格式或新任务类型可能需要扩展通用能力，不能承诺任何数据天然兼容。

## 6. 数据验收与划分：先守住这几个底线

`audit.json` 至少统计：发现文件数、有效音频数、排除数、未识别数；每类人员数与录音数；采样率、时长、通道分布；重复簇；标签与人员关系问题。档案文件不应只因后缀不支持就被报告为“空库”。

准入条件：标签含义明确；人员有可靠归组或有“每人一条原始录音”的明确证据；可解码；来源版本明确；重复/标签冲突等阻断问题已解决。不能把没有ID理解成“肯定一人一条”。旧整理目录可辅助对照，不能覆盖原始元数据。

`split.csv` 固定三列：`sample_id, group_id, split`。每个样本恰好属于train、val或test之一。官方划分可靠时优先保留并检查人员/来源交叉；没有官方协议时，Demo使用分组且尽量分层的60%/20%/20%留出划分，seed=0。类别人数不够时明确报错，不偷偷退回按录音随机划分。

同人有不同版本、麦克风或录音任务，也必须在同组；跨人员共享同源内容的异常情况需要扩大隔离组或排除。**先按来源与人员归组和划分，再切窗；不是切完窗随机分。**

Demo的单次留出成绩只用于工程闭环。正式论文再为满足人数条件的任务锁定外层分组交叉验证；每折的验证/调参只使用外层训练人员。框架保留 `split_id` 即可容纳以后多个折，不必现在就把嵌套调参全部实现。

## 7. 一份实验配置：所有可调参数放在同一个地方

### 7.1 本机路径绑定

`configs/paths.example.yaml`：

```yaml
workspace_root: /REPLACE_ME/speechbench-workspace
dataset_roots:
  mdvr_kcl: /REPLACE_ME/original_dataset/Parkinson_KCL_2017
  eatd: /REPLACE_ME/original_dataset/EATD
```

复制为 `paths.local.yaml` 后填真实路径，不提交。数据根必须显式绑定，不能扫描全服务器寻找“名字最像”的目录。

### 7.2 数据任务定义

`configs/datasets/mdvr_kcl_read_v1.yaml` 的目标格式：

```yaml
task_id: mdvr_kcl_read_v1
dataset_id: mdvr_kcl
adapter: mdvr_kcl
task_version: 1
admission: pending_review                 # 人工数据验收后改为approved
selection:
  elicitation: read_text
  variant: original
labels:
  class_order: [control, parkinson]
  positive_class: parkinson
  mapping: {HC: control, PD: parkinson}   # 必须核验来源后使用
evaluation_unit: subject
```

EATD先建待核验配置，把评分来源、阈值或原始类别映射留成阻断字段；缺少依据时prepare只能输出待核查报告，不能生成“已准入任务”。不要套用网上某个同名版本的阈值。

### 7.3 实验参数定义

`configs/experiments/mdvr_mfcc_mlp.yaml`：

```yaml
experiment_name: mdvr_mfcc_mlp_baseline
experiment_group: demo_v1
purpose: 验证MDVR到统一结果的完整流程
stage: demo                              # demo / exploratory / confirmatory
task_config: configs/datasets/mdvr_kcl_read_v1.yaml
prepared_id: REPLACE_WITH_APPROVED_PREP_ID
split_id: REPLACE_WITH_APPROVED_SPLIT_ID
seed: 0

audio:
  target_sample_rate: 16000
  channel_policy: select
  channel_index: 0                       # 需核查所选通道确为目标声音
  observe_seconds: all                   # 听整条合格录音；也可设16
  position: center                       # all时不生效，但仍保存
  window_seconds: 4.0
  stride_seconds: 4.0                    # 初版固定等于window，避免重叠计时
  min_tail_seconds: 0.25                 # 极短尾片排除并报告，不悄悄丢弃
  amplitude_normalization: none

features:
  name: mfcc_stats
  n_mfcc: 20
  n_mels: 64
  frame_ms: 25.0
  hop_ms: 10.0
  n_fft: 512
  f_min_hz: 0.0
  f_max_hz: null                         # 解析为目标采样率/2并保存
  delta: false
  stats: [mean, std]                    # 初版固定；输入40维
  implementation: torchaudio            # 具体版本和其余默认值展开保存

model:
  name: mlp
  hidden_dims: [128]
  dropout: 0.2

training:
  epochs: 80
  batch_size: 32
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  selection_metric: subject_balanced_accuracy
  selection_direction: max
  feature_standardization: train_only
  loss_weighting: equal_subject

metrics:
  primary: balanced_accuracy
  binary_threshold: 0.5
  confidence_interval: none             # demo；正式关键结果再开人员bootstrap

cache:
  mode: shared                          # off / run / shared
  pinned: false

runtime:
  device: cuda
  num_workers: 4
```

这些数值是启动配置，不是声称最优。解析器应展开全部生效值并保存，不能只存这份简短模板。位置固定随机采样等新选项以后明确实现再开放；第一版支持head、center、tail即可。

为了复现，还要固定：重采样方法、MFCC的Mel刻度/归一化、功率与对数方式、DCT、`center`/padding、标准差`ddof`、概率聚合方式、模型激活、初始化与优化器betas/epsilon等。实现者可以采用所选库明确默认值，但必须把解析后的值写入 `config.yaml`；改变默认值视为实验变化。

正式运行前校验所选原件与准备清单的内容哈希；仅靠文件名或修改时间不能证明数据未变。数据已变就重新prepare，不覆盖旧清单。

实验配置锁住 `prepared_id` 和 `split_id`，不使用“latest”自动指向新数据。改变标签/样本纳入规则产生新的准备版本；只改dropout不重新划分人。

## 8. 音频、特征、模型尺寸怎样衔接

统一的数据流是：**清单录音 → 固定划分 → 观察时段/窗口 → 特征 → 模型 → 窗口概率 → 录音/人员概率 → 指标。**

### 8.1 音频参数

原始8k和16k都可以读取，初始接口统一16k。8k上采样到16k只统一输入接口，不会恢复高频内容，报告要保留原始采样率分布。以后做频率实验时区分“接口Hz变化”与“有效带宽变化”。第一版不增加VAD、增强和降噪，避免一开始就引入更多变量。

比如20秒录音，observe=16、position=center、window=4：实际听[2,18]，分成4个4秒窗口。短于窗口的录音只保留有效声音，必要补零并传递有效长度；特征统计排除补零帧。不同模型使用同一窗口索引，不能一方听全部一方只听前4秒却当作同等输入。

### 8.2 先实现MFCC统计＋MLP

MFCC帧矩阵为 `[20, T]`，有效帧上逐系数计算均值和标准差，得到40维向量。MLP为 `40 → 128 → C`，隐藏层ReLU与Dropout；输出logits，用交叉熵训练。输入维度从特征自动推导，类别数从 `class_order` 推导。改 `n_mfcc=40` 后输入应自动变成80维，不许在模型里硬编码40。

StandardScaler用训练集特征拟合并保存 `transforms.joblib`，只应用于后续集合。每条训练样本的权重按人员/录音/窗口层级归一，避免长录音和录音多的人支配训练：每人总权重相同，同人录音等权，同录音窗口按有效时长分配。训练循环对这些全训练集权重归一后的加权损失求无偏均匀batch估计；不能再按每个batch权重和重归一而改变目标。此规则先固定，不同时叠加类别重加权；如要研究类别权重，单列实验。

### 8.3 再实现log-Mel＋CNN

log-Mel输入为 `[B, 1, 64, T]`。建议启动结构：三组 `Conv2d(3×3,padding=1) → ReLU → Pool`，通道16、32、64；池化只在合法维度内进行，保留有效帧mask，最后按有效位置池化为固定64维，再 `Dropout → Linear(64,C)`。每层时间长度/mask随池化更新；排除补零对卷积边界和池化统计的污染，并用短音频测试验证。不要以为普通全局平均自动忽略补零。

第一版log-Mel可明确固定绝对功率加epsilon取log的转换，归一化只对有效位置计算；如使用训练集统计，同样按训练拟合、随run保存。MLP与CNN的前端本来不同；**统一的是清单、划分、观察预算、训练入口、评价和记录，并不是强迫所有模型吃同一形状的张量。**

两条基线可以比较整个系统的表现；如果要单独归因于网络结构，就必须再固定相同前端设计专门的对照实验。不要把“MFCC＋MLP对log-Mel＋CNN”解释为只改变网络。

## 9. 缓存政策：不把十几种MFCC都永久存下来

第一版默认只缓存数值特征，不另存一整套重采样音频，也不保存每条频谱PNG。审计可导出少量人工检查图，不作为训练依赖。

| 模式 | 存放位置 | 用于什么 | 生命周期 |
| --- | --- | --- | --- |
| `off` | 不持久化，仅有当前运行所需内存/临时batch | 一次性、小规模探索 | 用完释放；相同配置重跑需重算 |
| `run` | `runs/<run_id>/cache/` | 不打算复用的特征参数探索；同一run多epoch仍可复用 | 运行后可显式清理该run缓存，结果仍保留 |
| `shared` | `cache/features/<feature>/<cache_key>/` | 同一特征配方重复训练多个模型/seed | 手动审阅后清理；正式基线可标pinned |

**十几种MFCC参数且每种只跑一次：用run模式；同一种特征扫十几个dropout/宽度：用shared模式。** 不必持久化并不等于每个epoch重算；run模式就是把单次训练复用与长期共享分开。

缓存key至少包括：选中源文件内容与样本集合指纹、通道/裁剪窗口、目标Hz、前端全部参数、实现版本/特征代码版本。修改n_mfcc、位置、Hz或前端代码应失效；只改MLP层数不失效。缓存中保留 `meta.json`、窗口索引和完成标记，写完后原子发布；半截缓存不能当成功结果读取。

不缓存StandardScaler/PCA等训练拟合结果到跨划分共享区。第一版不做自动LRU驱逐；先做 `cache list`、`cache clean --dry-run`，真正删除必须明确范围并排除pinned。删除目录限定在所绑定工作区的cache范围，校验解析后的路径、不跟随逃逸符号链接；绝不能清理原始数据或runs的正式产物。

每个run保存缓存key、命中情况和特征参数。即使缓存已删，只要原件、清单、代码与环境还在，仍能重建。特征矩阵通常不是必须永久归档的论文证据；原始依据、划分、配置、代码版本、预测和指标才是。

## 10. 一次实验的完整记录：从此不再出现“这个结果是哪次的”

### 10.1 运行标识与配置

创建run时立即分配唯一 `run_id`，例如 `20261109T143000+0800_mdvr_mlp_s0_a13f92_<random>`。**实验参数相同重跑仍是新run，不覆盖旧目录。** 另存 `config_hash` 用于识别相同科学配置，时间与绝对路径不参与科学配置指纹。

`run.json` 必须含以下信息：

| 类型 | 必须记录 |
| --- | --- |
| 身份 | run_id、experiment_name、experiment_group、purpose、stage |
| 时间 | 带时区的开始/结束ISO时间、运行耗时 |
| 状态 | created/running/completed/failed/interrupted，失败步骤与错误摘要 |
| 代码 | Git commit、工作树是否dirty；dirty时保存已跟踪文件diff和执行源码快照，避免未跟踪代码无法追溯 |
| 数据 | dataset_id、task_id、准备版本、清单/划分/任务定义指纹、样本数与人员数 |
| 参数 | config_hash、seed、所有相关随机源种子、确定性设置 |
| 环境 | Python/依赖、OS、CPU/GPU、驱动、CUDA/cuDNN信息与安装来源 |
| 关联 | sweep_id、父run（如重试）、是否为论文正式结果 |

`config.yaml` 保存**最终完整配置**，包括CLI覆盖后的值；再保存原始配置来源、执行命令和本机路径绑定快照（后者属于私有运行产物）。复制清单和划分到 `inputs/`，而不是只留下指向会变文件的链接。正式结果要求干净Git提交；Demo允许dirty但必须记录可恢复源码。精确相同数值在不同GPU/库版本上不总能保证，环境记录与确定性设置要如实呈现。

### 10.2 训练、预测、评价

`best.pt` 保存按验证集主指标选中的模型、类序、模型配置、标准化引用；`last.pt` 保存最后训练状态，包括模型/优化器/epoch/随机状态。第一版若暂不支持精确断点续训，就明确标“不支持”；失败后创建新run重跑，不能加载部分状态却称精确续训。

`history.csv` 每epoch一行，至少有epoch、train_loss、val_loss、val_primary、learning_rate、耗时。验证指标先按正式评价单位聚合；不能训练时用窗口accuracy选模型，最终却以人员balanced accuracy作主成绩而不说明。

预测文件保存概率，不只保存预测标签：

```text
predictions_windows.csv:
  split, window_id, sample_id, subject_id, y_true,
  prob_control, prob_parkinson, valid_seconds

predictions_recordings.csv:
  split, sample_id, subject_id, y_true, y_pred, prob_control, prob_parkinson

predictions_subjects.csv:
  split, subject_id, y_true, y_pred, prob_control, prob_parkinson, n_recordings
```

上面概率列只是二分类示例，实际依固定类别顺序生成C列；类名先校验为安全键，原始名称保存在task。保存被选checkpoint对val与test的预测，并明确split；不要保存来自不同epoch的val概率却未注明。测试不参与调参、早停、阈值选择。

窗口按有效时长加权平均概率得到录音概率；同人多录音等权平均得到人员概率。二分类阈值默认0.5，多分类argmax。只对任务允许且标签一致的评价对象聚合。所有方法完整覆盖同一测试对象；漏行/重复行/非法概率必须失败，不能悄悄删掉难例。

### 10.3 统一指标

| 必须输出 | 约定 |
| --- | --- |
| Accuracy、Balanced accuracy | 分别报告；主指标先固定balanced accuracy |
| Precision、Recall、F1 | 每类、macro、weighted；二分类另报预声明正类 |
| Sensitivity、Specificity | 二分类正类；多分类逐类one-vs-rest |
| ROC-AUC | 使用概率；二分类指定正类，多分类OVR macro与逐类 |
| Average precision和PR曲线 | 命名明确为AP，不能与PR曲线梯形面积混为一谈 |
| Confusion matrix | 固定类别顺序，原始计数与按真实类归一版本 |
| Log loss、二分类Brier | 概率质量；保存epsilon与计算口径 |
| Support与覆盖率 | 各类人数/录音数/窗口数，实际与应有数量 |
| 成本 | 参数量、训练/特征/推理时间、设备；MLP/CNN均记录 |

人员和录音的指标放不同键，主单位由task固定，不能混在同一列。分母为0、测试缺类、AUC不可定义时写 `null` 并附原因；不硬填0，不生成假ROC图。PRF的零分母处理可固定zero_division=0，但另标undefined类别。缺预期类别的主指标不得进入完整任务排名。

`metrics.json` 至少包含run_id、task_id、class_order、evaluation_unit、positive_class、primary_metric、各指标、support和warnings。`metrics_per_class.csv` 每类一行。**它们必须能由保存的预测文件与配置重新计算，不依赖再次训练。**

`report.md` 自动写：本次做了什么、数据与划分、有效配置、最佳epoch、指标、图、限制和产物索引。成功状态只能在预测覆盖、指标导出等检查全部通过后写completed；运行失败仍留配置、日志和failed状态，汇总时显示失败，不能消失。

### 10.4 论文与Git怎样对应

- 正式实验前先提交代码与配置，记录干净commit，再运行；不要运行完改代码却仍把结果归到旧版本。
- `docs/experiment_log.md` 用一行链接每个实验组、配置、run_id列表和存储位置，注明哪些用于论文。
- 完整runs在服务器备份；选入Git的每次结果保存在 `results_summary/<run_id>/`，至少含去标识化config、metrics、report与图，并保留inputs指纹和代码commit。
- 不默认把患者级清单、逐人预测、真实路径和原始数据推到远程仓库。正式小结果导出先检查许可与可识别信息；无法公开的复现材料在受控备份中保存，并说明获取条件。
- Git同步代码进展不等于备份实验数据。准备清单、固定划分、完整run必须另有持久备份，否则代码还在也无法查回历史结果。

实验索引建议直接使用下表模板，不必再开发一个记录系统：

| 实验组 | 研究问题 | 配置文件 | run_id列表/清单 | 代码commit | 完整产物位置 | 是否入论文 | 结论/限制 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| demo_v1 | 验证流水线 | configs/experiments/mdvr_mfcc_mlp.yaml | 运行后填写 | 运行后填写 | 工作区runs下对应目录 | 否 | 工程闭环，不用于正式结论 |

验收日志每条至少记录：阶段、日期、验收人、执行命令、通过项、未通过项、证据路径、最终通过结论、代码提交号。验收未通过时不要把下一阶段标为已启动。

## 11. 统一运行顺序与命令

下列是需要实现的目标命令，**不是声称已经可执行**。每个阶段只实现相应子命令，README同步标出可用范围。

```bash
# 一次性数据准备，输出prep_id和审计报告
sddbench prepare --task configs/datasets/mdvr_kcl_read_v1.yaml --paths configs/paths.local.yaml

# 人工认可数据后生成候选划分；REPLACE_ME换为真实已验收ID
sddbench split --task mdvr_kcl_read_v1 --prepared REPLACE_ME --protocol demo_group --seed 0 --paths configs/paths.local.yaml

# 完整参数、样本数和输出位置预检；只读，不训练
sddbench run --config configs/experiments/mdvr_mfcc_mlp.yaml --paths configs/paths.local.yaml --dry-run

# 一次正式执行；CLI打印唯一run_id和绝对输出目录
sddbench run --config configs/experiments/mdvr_mfcc_mlp.yaml --paths configs/paths.local.yaml

# 由预测重算指标与报告，不重新训练
sddbench report --run /ABSOLUTE_PATH_TO_ONE_RUN

# 结果总表与只读展示
sddbench summarize --paths configs/paths.local.yaml
sddbench dashboard --paths configs/paths.local.yaml

# 完成基础闭环后再实现批量与缓存管理
sddbench sweep --config configs/sweeps/mdvr_dropout.yaml --paths configs/paths.local.yaml --dry-run
sddbench sweep --config configs/sweeps/mdvr_dropout.yaml --paths configs/paths.local.yaml
sddbench cache list --paths configs/paths.local.yaml
sddbench cache clean --paths configs/paths.local.yaml --key REPLACE_ME --dry-run
```

`run.py` 的固定执行顺序：读并校验配置 → 绑定已验收数据/划分 → 创建run并存完整配置 → 生成窗口索引 → 读/算特征 → 只用train拟合变换 → 统一训练与val选checkpoint → 对选定checkpoint生成val/test概率 → 按固定规则聚合 → 指标与报告 → completed。

测试阶段虽可以由同一命令自动执行，但训练代码不能拿到test标签/指标来选参数。探索实验如果反复看过同一test，它就不再是最后的独立确认依据；正式研究需要预先锁定协议或保留未用于选择的外层测试。

## 12. 分阶段施工：每一步都由你验收后再继续

**Codex每次只做当前阶段；完成后停下，交付清单和验收说明，不自动跨阶段。** 所有真实数据路径不明确时先列出缺项；允许做不依赖路径的骨架和合成测试，不能自行猜目录。下面的提交说明是建议，不代表自动获准push；你验收后再让它提交/推送。

### 阶段0：建仓库与填写路径

**实现**：顶层文件、最小包与CLI、配置读取、paths模板、工作区安全路径检查、README、验收日志。只创建需要的data/features等目录，不写假模型与假成绩。

**你提供**：新仓库路径、工作区路径、MDVR/EATD原始根目录；确认当前仓库远程是否私有。

**验收**：能安装、`sddbench --help`能运行；占位路径不能误运行；工作区初始化不碰原始数据；`git status`无真实数据和路径文件。

**交付**：修改文件清单、实际命令、输出、已知问题。通过后记录阶段0与提交号。建议提交：`chore: initialize benchmark repository`。

### 阶段1：只接通MDVR数据，不训练

**实现**：schema、MDVR适配器、prepare和audit。优先ReadText；列出所有被忽略/排除的原目录项和理由，不把整个原始库挤进任务。

**你验收**：

- 人工抽查至少每类5条（不足则全部），核对路径、能听到的内容、标签来源和人员ID；另外核对至少两条同人/版本关系（若存在）。
- 查看每类人数、录音数、时长、Hz；与原目录和说明对上。
- 查看全部阻断问题；不能只看扫描“成功”。
- 解码失败、重复文件、未知ID有明确归宿；没有为了消除警告直接删除原件。

**通过标准**：任务定义已确认，manifest有效，阻断问题清零并注明处理依据。若人数与此前HTML不一致，解释原因，不强行凑37。建议提交：`feat: add audited MDVR dataset adapter`。

### 阶段2：固定划分＋音频窗口＋MFCC；仍不训练

**实现**：split、audio、windows、mfcc、cache基础；至少支持run/shared缓存和off直算。输出一份人工检查摘要：样本、划分、窗口数、MFCC维度、缓存命中。

**你验收**：

- 抽查split三部分，确认人员与来源互斥，类别覆盖满足协议；认可后固定split_id。
- 对几条长短音频检查原时长、选段起止、窗口数、尾片和有效帧。
- MFCC20确实输出40维；改为MFCC40后变为80维，修改Hz/位置使缓存失效。
- 相同配置复算数值在声明容差内一致；补零不进入统计；更改MLP参数不会触发特征重算。
- 对短录音、坏文件、重复来源、缺类别有真实验证，不只写几个与实现一模一样的测试。

**通过标准**：已能得到可信的模型输入；尚不需要任何分类成绩。建议提交：`feat: add grouped splits and reusable MFCC pipeline`。

### 阶段3：MLP完整闭环与结果追溯

**实现**：MLP、通用trainer/predict、聚合、统一指标、report和run记录。首先小规模短训练确认代码，再运行真实Demo；小规模结果必须标demo。

**你验收**：

- 每次运行有独立目录，配置完整，run_id、时间、Git版本、数据和划分可查。
- 查看验证选择逻辑、只在train拟合的Scaler和人员权重；test不参与选择。
- 所有应预测对象完整覆盖；预测列序与类别对应，概率合法。
- 从保存预测重算metrics，与原文件一致；另一人只看run目录能说清“何时、什么数据、哪套划分、什么配置、什么代码”。
- 同配置重跑不覆盖，故意中断/失败留下状态；至少验证一次“特征缓存清掉仍可重算”的流程。

**通过标准**：一个数据集已经形成可复现的实验事实。成绩高低不是工程验收标准。建议提交：`feat: complete reproducible MLP experiment pipeline`。

### 阶段4：接入EATD，证明可扩展

**实现**：只加EATD适配器、注册、任务与实验配置、文档和相应测试；所有下游模块复用。先完成人工标签规则核对，再允许训练。

**你验收**：

- 正确区分人员目录与疾病标签，正确关联评分/标签。
- 原件与 `_out` 等处理版本不同时当独立输入；缺失录音有记录，不强行补齐预期数量。
- 同人的多段录音在同一集合，最后按声明单位评价。
- EATD能用同一个run命令得到同样结构的结果目录、指标与报告。
- 查看Git diff：trainer、MLP和metrics不应出现按EATD名称分支。若确需修通用缺陷，单列原因并回归验证MDVR。

**通过标准**：两个真实数据集使用同一条后续流水线。建议提交：`feat: add EATD through standard adapter interface`。

### 阶段5：CNN、批量实验和只读展示

**实现**：log-Mel与CNN；sweep顺序任务清单；summarize；最后Streamlit展示。先串行跑，不急着多GPU并行。

**你验收**：

- MLP/CNN复用相同数据和划分，观察预算一致；前端差异清楚记录。
- 小扫参如dropout=[0,0.2]、seed=[0,1]产生4个独立run，jobs.csv完整链接成功和失败。
- 网页按实验组、任务、模型、seed筛选，点开可看到配置、指标、图与目录；failed也可见。
- 不同split/task/评价单位不能无提示混成一个“总排名”。
- 不同参数缓存按规则复用/失效，扫参不会无限默认保留所有特征。

**通过标准**：具备两数据集、两基线、可扫参、可追溯、可浏览的平台。建议分两次提交：CNN与sweep一次，展示一次。

### 阶段6：锁定正式基准，再扩展更多数据集

先选已通过审计的任务，不为凑二三十个库降低标签/人员要求。固定数据准备、评价单位、划分协议、前端与训练预算，形成baseline版本。再按Hz、观察时长、位置、特征维数、宽度、dropout做有目的的实验，不一次把所有轴全组合。

每轮写明：要回答什么、只改哪个因素、保持什么、哪些run属于这一组、以什么指标和不确定性报告。正式比较使用相同测试人员/折，报告多seed结果与关键比较的人员级或分组置信区间；不能将各折/seed当独立患者扩大样本量。跨任务优先逐任务报告及胜负/名次概览；需要总指标时预先定义权重与共同任务集合，不能简单池化不同疾病标签算一个AUC。

平台本身的可交付成果是：可扩展数据适配、固定协议、统一记录与可复现基线。论文的实证贡献来自随后真实结果：哪些设置敏感、何种结论跨数据集稳定、何处不成立。**代码规模和界面完整不自动等于研究创新。**

## 13. 前端第一版具体展示什么

只读仪表盘布局固定为：

1. 顶部筛选：task、experiment_group、模型、seed、完成状态。
2. 实验总表：run_id、时间、commit、数据/划分ID、前端、Hz、观察长度、模型参数、主指标、AUC、状态。
3. 点击一个run：显示完整config、每类support、混淆矩阵、ROC/PR、学习曲线和运行说明。
4. 选择多个run：先检查task/split/评价单位兼容，再比较参数与成绩；不兼容则分组展示并说明原因。

仪表盘读取runs/metrics/config；不重新算另一套指标，不从文件夹名字反推参数。关闭网页不影响训练；关闭训练不丢已有结果。第一版不做登录、在线编辑数据、网页启动GPU任务和模型服务部署。

## 14. 交给服务器Codex的首条施工指令

把本说明放到新仓库 `docs/construction_plan.md`，下面填完真实路径，只发这一阶段。以后让它读取同一文档执行下一阶段，不要每轮换一套命名。

```text
你是此仓库的工程实现者。请先阅读 docs/construction_plan.md。

代码仓库绝对路径：<填写>
实验工作区绝对路径：<填写；不要放原始数据里>
MDVR原始数据根目录：<填写>
EATD原始数据根目录：<填写；本阶段不处理>
本次授权阶段：仅阶段0。

要求：
1. 先检查现有仓库与路径，不覆盖已有工作，不改动任何原始数据。
2. 按文档确定的目录和命名施工；只创建本阶段用得到的文件。
3. 只实现安装、CLI骨架、配置与路径验证、文档和Git排除规则。
4. 未实现的命令不得假装运行成功，不生成假清单、假模型或假成绩。
5. 需要我决定的真实数据含义或路径问题列清单，不自行猜测。
6. 完成后提供：实际修改文件及用途、执行命令与结果、未解决事项、
   我应执行的人工验收步骤、下一阶段需要的信息。
7. 到此停止，等待我验收；不进入阶段1，不自动commit或push。

若文档内部有冲突，指出具体位置并给出最小修正建议，不擅自重构全部方案。
```

下一条只需改为“本次阶段1；阶段0验收已通过，提交号为……”，并补充已核实的数据说明。每次人工验收后把结论写入acceptance_log，再由你明确要求提交/推送。这样每一步都有可回退的版本，也不会在数据尚未可信时堆出一整套无法验收的平台。

## 15. 完成第一版的最终判据

- 新增第二个数据集，不另写训练/评价脚本。
- 同一次任务更换MLP/CNN，复用固定样本、划分和评价口径。
- 改Hz、时长、位置或模型参数，只改配置；结果准确记录变化。
- 需要时复用特征，不需要时能清掉缓存且保留复现依据。
- 每个结果可追溯到原件清单、标签依据、固定划分、完整参数、代码与环境。
- 从已保存预测可重算所有已实现指标，从固定数据/代码/配置可重跑训练。
- 你验收一个阶段后再进入下一个阶段；Git进度和服务器实验备份各有记录。

**现在先做阶段0，然后做阶段1的数据验收。先拿到一个可信清单，再拿到一个可信结果；不要同时启动所有数据集和所有参数实验。**
