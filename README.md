# speech_diease_benchmark_2026

语音疾病（构音障碍）识别基准平台。目标：在同一套数据流水线、同一套评测协议下，
比较不同模型在"从语音判断说话人是否患病"这件事上的表现，并能公平地扩展到多个数据集。

> 本仓库当前完成了**数据层（D1–D5）+ 模型层（D6–D7）+ 结果层（D8）**：
> 把数据集摸清楚、量清楚、留证据，把"看哪一段声音"固化成可复现的窗口索引，
> 在有效声音上算出可缓存的 MFCC 统计量，接入第二个数据集、
> 建立"标签未确认就拒绝出训练清单"的准入门禁，按说话人隔离地划出 train/val/test，
> 用缓存的 MFCC 跑出**第一条可复现的 MLP 基线**（test 6 人，balanced accuracy 0.833，AUROC 0.889），
> 支持用一行命令扫参数，并把每次运行按**统一结果契约**归档进仓库、自动生成跨数据集榜单。
> 尚待完成：多 seed 置信区间、把第二个数据集纳入训练、自动生成汇总页。

---

## 一、先说人话：这次到底干了什么

假设你要做一场"语音看病"的比赛，第一步不是找模型，而是先搞清楚**手里的录音到底是什么**：
一共多少条？都是什么格式？多长？谁录的？标签靠不靠谱？如果这一步没做扎实，
后面模型再好，结论也是建在沙子上的。

这次做的事就是这个"盘点"：

1. **搭了一个能长久的骨架**：一个命令行工具 `sddbench`，把"读数据"和"跑模型"彻底分开。
   以后加新数据集，只需要写一个几十行的"数据集说明文件"（适配器），
   扫描、解码、统计、特征、训练这些通用代码一行都不用改。这就是文档里说的
   "数据层对模型层透明"。
2. **把 UAspeech 全库 14 万多条录音逐条验了一遍**：不是看文件名猜，而是真的打开解码，
   记录每条的真实格式、采样率、通道数、时长、哈希、音量、是否有静音/坏文件。
   结果是 **143,290 条全部可正常解码，0 条损坏**。
3. **生成了一份能写进论文的统计报告**：时长分位数、格式分布、每个说话人多少条、
   每个麦克风覆盖多少条，还给出了"如果截断到 1/2/4/6/10 秒，会丢多少、要补多少零"的对照表，
   方便后面秉着数据说话去定观察窗长度，而不是拍脑袋。
4. **把标签和麦克风问题按"工程事实"处理完了**：
   - 用户确认本地 `Healthy/`、`Disorder/` 目录就是健康/疾病标签，标签复核全部改为 confirmed；
   - UAspeech 每条话在 7 个麦克风下各有一份，全部收进来会让同一句话出现 7 次。
     所以准入时每个 `speaker|block|prompt` 只保留一条，并显式选定 **M3**（覆盖全部 28 人；
     M2 在 Healthy 组缺约 1/3 录音）。这是防泄漏的工程选择，不是研究结论。
5. **接入了第二个数据集 Parkinson KCL 2017**：73 条长录音、44.1 kHz/PCM_24、37 位说话人，
   用同一套扫描/统计/窗口/MFCC 代码跑通，证明"加数据集只写适配器"不是口号。
   同时新增**准入门禁**：只有 `label_review.csv` 里逐人确认过身份和标签，才会生成
   `manifest.csv`；KCL 仍未确认，所以它的真实运行被明确阻断，而不是自动猜标签。
6. **搭起模型层并跑出第一条基线**：直接读取 D3 缓存的 40 维 MFCC，
   训练 `40 → 128 → 2` 的 MLP；按人隔离划分，按人计算主指标，
   并把配置、权重、逐窗口/逐人预测、指标、图、报告打包成一个可追溯的结果目录。
7. **把结果层单独拿出来，定成一份所有数据集共用的契约**：每次运行都必须报
   accuracy、balanced accuracy、sensitivity、specificity、precision、F1、AUROC、
   average precision、log loss、Brier 和混淆矩阵，并且按 **窗口 / 录音 / 说话人** 三种粒度
   在 train/val/test 上都各报一遍；主干指标固定为 test 说话人级。
   小体积的部分（指标、逐类指标、报告、配置、4 张图、清洗后的运行元信息）自动归档到
   `results_summary/<数据集>/<运行ID>/`，再汇总成一张 `leaderboard.csv/md`。以后接新库，
   只要按同一契约产出结果，就能直接进同一张榜。

**没有上传的东西**：原始患者录音一条都没传（隐私 + 体积）；
这次扫描产生的逐文件清单 `files.csv`（73MB）和断点续扫日志 `progress.jsonl`（119MB）
也按规则排除，只上传了 KB 级的"总览统计"。模型权重 `best.pt`、逐窗口/逐人预测、
特征缓存和患者级 `manifest.csv` 同样留在本机；目前入库的结果归档合计 **592 KB**。

---

## 二、本阶段的技术交付

| 项目 | 内容 |
| --- | --- |
| 命令 | `inspect-file`（单文件核查）、`stats`（quick/full 统计，可断点续扫）、`dataset-report`（重建报告）、`extract-features`（窗口索引 + MFCC 统计 + 分片缓存）、`prepare`（标签复核 + 人员隔离清单，未确认则阻断）、`split`（按说话人隔离的分层划分）、`run`（冻结特征 MLP 训练 + 结果包）、`sweep`（参数网格）、`summarize`（同库内跨运行汇总）、`publish`（按结果契约归档 + 跨数据集榜单） |
| 数据层模块 | `scan`（枚举分类）、`decoders`（probe+解码，soundfile 优先、ffprobe 兜底）、`audit`（并行审计+质量摘要）、`stats`（聚合）、`adapters/uaspeech` + `adapters/parkinson_kcl`（命名与归属解释）、`audio`（通道/裁剪/重采样）、`windows`（位置规则+`view_hash`）、`prepare`（确定性分层抽样）、`admission`（复核/隔离/批准清单）、`split`（人员隔离分层划分）、`features/mfcc`（固定配方）、`features/cache`（分片/续算/复用）、`modeling/*`（特征读取、指标、MLP、实验配置、训练、报告、扫描/汇总）、`results/*`（结果契约、发布白名单、跨数据集 leaderboard） |
| 产物 | `files.csv`、`statistics.json`、`meta.json`、`issues.csv`、`length_candidates*.csv`、`report.md`、`figures/*.png`；D2/D3 追加 `windows.csv`、`job.json`、`inspection.md`、可审听预览、`cache/features/mfcc_stats/<feature_key>/`（`meta.json` + `index.csv` + `shard_*.npz`）；D4 追加 `label_review_status.csv`、`admission.json`、`admission.md`、通过时才有 `manifest.csv`；D5 追加 `split.csv`、`split.json`、`split.md`；D6 追加 `runs/<run_id>/`（`run.json`、`metrics.json`、`predictions_*`、`best.pt`、`report.md`、`figures/`）；D8 追加 `results_summary/<dataset>/<run_id>/`（契约指标、逐类指标、报告、配置、清洗后的 `run_meta.json`、`figures/*.png`）与顶层 `leaderboard.csv/md` |
| 测试 | 62 项合成数据测试全绿（不含任何真实患者音频） |
| 设计约束 | 适配器不碰解码；统计只报事实不自动选参数；标签未核验前不产出分类清单 |

### UAspeech 全库统计结果（full 模式，逐条解码）

| 维度 | 事实 |
| --- | --- |
| 文件 | 143,290 条：Disorder 69,615（13 人）/ Healthy 73,675（15 人） |
| 格式 | 100% WAV / PCM_16 / 16 kHz / 单声道；扩展名与实际容器 0 处不符 |
| 解码 | 成功 143,290，失败 0；命名解析失败 0 |
| 时长 | P50 2.06 s，P60 2.24 s，P80 2.93 s，P90 3.88 s，P95 5.09 s，最长 56.9 s，合计约 100 小时 |
| 结构 | 255 词项 × 3 块 × 7 麦克风 = 每人 5,355 条（满配）；Healthy 有 7 人缺录音（M01 最少 2,805 条） |
| 音量 | 峰值中位数 1.0（语料疑似整体做过归一化），RMS 中位数 0.096 |

观察窗候选（截断/补零成本，供人工决策，不自动选定）：

| 观察长度 L | 保留语音比例 | 需补零比例 |
| --- | --- | --- |
| 1 s | 39.8% | 0.0% |
| 2 s | 73.4% | 7.8% |
| 4 s | 91.4% | 42.6% |
| 6 s | 95.8% | 59.9% |
| 10 s | 98.4% | 75.3% |

### Parkinson KCL 2017 全库统计（第二个数据集，full 模式）

| 维度 | 事实 |
| --- | --- |
| 文件 | 73 条：Healthy 42（21 人）/ Parkinson 31（16 人），共 37 位说话人 |
| 格式 | 100% WAV / PCM_24 / 44.1 kHz / 单声道；解码成功 73，失败 0，命名解析失败 0 |
| 时长 | P50 141.3 s，P90 178.0 s，最长 220.6 s，合计约 2.87 小时 |
| 结构 | 文件名 `ID<nn>_?hc/pd_...`；`hc`/`pd` 仅作为候选标签，仍需人工复核 |
| 技术验证 | 6 条 native 44.1 kHz / head 4 s：`win=1103, hop=441, n_fft=2048`，MFCC 40 维、0 失败 |
| 待处理 | `ID22_hc_0_0_0` 与 `ID22hc_0_0_0` 归一化后 origin 冲突（时长/哈希不同），准入被阻断 |

### D6 第一条 MLP 基线（UAspeech / M3 / head 4 s / seed 0）

| 项 | 结果 |
| --- | --- |
| 输入 | 40 维 MFCC 统计量（20 维 mean + std），21,126 条 / 28 人 |
| 模型 | `40 → 128 → 2`，ReLU，dropout 0.2 |
| 划分 | train 17 人 / val 5 人 / test 6 人（人员互斥） |
| 主指标（test，按人） | balanced accuracy **0.833**（5/6 人判对） |
| 其他（test，按人） | sensitivity 1.000、specificity 0.667、AUROC 0.889 |
| 参考（test，按录音） | balanced accuracy 0.699、AUROC 0.762 |
| 运行 | 139 s；代码 `1246735`（dirty=false）；最佳 epoch 6 |

> 这是**流程基线**，不是调优结果：test 只有 6 位说话人，换 seed 可能变成 4/6 或 6/6。
> 它的价值是证明"数据 → 特征 → 模型 → 指标 → 结果包"已经闭环、可复现、可调参。

### 必须人工核查的发现

1. **标签疑点（最要紧）**：本库所有 `C` 前缀说话人（CF02…CM13，13 人）都在 `Disorder/`，
   所有非 `C` 前缀（F02…M16，15 人）都在 `Healthy/`。而 UASpeech 官方约定中 `C` 前缀是 Control（对照/健康）。
   如果本地整理时把目录映射反了，那么"目录名当标签"就是错的。**在核对官方语料说明之前，不出任何分类标签清单。**
2. **麦克风覆盖不均**：Healthy 组 M2 麦克风只有 7,610 条，其他麦克风约 9.9k–11.4k 条；
   Disorder 组各麦克风均衡（各 9,945 条）。若首任务选 M2，Healthy 组会莫名少约 1/3 录音。
3. 其他：`LA–LZ`、`D0–D9` 词项 token 语义待对照官方说明；个别组合出现 20–57 s 离群时长，建议抽查。

---

## 三、快速开始

```bash
pip install -e .[test]

cp configs/paths.example.yaml configs/paths.local.yaml   # 填写本机真实路径（该文件不入库）
python -m pytest tests/ -q

# 单文件核查
sddbench inspect-file --dataset uaspeech \
    --relative-path Disorder/CF02_B1_C1_M2.wav --paths configs/paths.local.yaml

# 全库统计（full：逐条解码 + 哈希 + 质量摘要；中断后加 --scan-id 可续扫）
sddbench stats --dataset uaspeech --mode full --paths configs/paths.local.yaml --workers 32

# 从已保存结果重建报告与图（不再解码）
sddbench dataset-report --dataset uaspeech --scan <SCAN_ID> --paths configs/paths.local.yaml

# D2：用已保存的 full 扫描生成窗口索引与预检报告（limit 为确定性分层抽样）
sddbench extract-features --dataset uaspeech --source <SCAN_ID> \
    --preprocess configs/preprocessing/uaspeech_native_smoke.yaml \
    --limit 12 --previews 6 --dry-run --paths configs/paths.local.yaml

# D3：在同样的窗口上计算 MFCC 统计并写入分片缓存（--cache-mode shared 可跨作业复用）
sddbench extract-features --dataset uaspeech --source <SCAN_ID> \
    --preprocess configs/preprocessing/uaspeech_native_smoke.yaml \
    --features configs/features/mfcc20_stats.yaml \
    --limit 12 --cache-mode shared --paths configs/paths.local.yaml

# D4：生成待复核模板并尝试准入（未确认时返回 3，且不会写 manifest.csv）
sddbench prepare --dataset uaspeech --source <SCAN_ID> \
    --write-review-template --paths configs/paths.local.yaml

# D4/D5：人工确认标签与麦克风后再准入；M3 是第一条技术基线的默认选择
sddbench prepare --dataset uaspeech --source <SCAN_ID> --mic M3 \
    --paths configs/paths.local.yaml

# D5：按说话人隔离、按标签分层的 60/20/20 划分
sddbench split --dataset uaspeech --prepared <PREPARED_ID> --seed 0 \
    --paths configs/paths.local.yaml

# D6：先空跑校验配置/划分/特征缓存，再训练并写结果包
sddbench run --config configs/experiments/uaspeech_mfcc_mlp_baseline.yaml \
    --dry-run --paths configs/paths.local.yaml
sddbench run --config configs/experiments/uaspeech_mfcc_mlp_baseline.yaml \
    --paths configs/paths.local.yaml
# 改一个参数做对比实验（新 run 目录，不覆盖旧结果）
sddbench run --config configs/experiments/uaspeech_mfcc_mlp_baseline.yaml \
    --set model.dropout=0.0 --set training.seed=1 --paths configs/paths.local.yaml

# D7：按网格扫参（每个组合一个 run），再把所有 run 汇成对比表
sddbench sweep --config configs/experiments/uaspeech_mfcc_mlp_baseline.yaml \
    --grid configs/sweeps/uaspeech_mlp_dropout.yaml --paths configs/paths.local.yaml
sddbench summarize --runs artifacts/runs --out results_summary --repo-root .

# D8：按结果契约归档（默认 <repo>/results_summary），并重建跨数据集榜单
sddbench publish --runs artifacts/runs --out results_summary --repo-root .

# 第二个数据集：统计 + 4 s 技术验证
sddbench stats --dataset parkinson_kcl --mode full --paths configs/paths.local.yaml
sddbench extract-features --dataset parkinson_kcl --source <KCL_SCAN_ID> \
    --preprocess configs/preprocessing/parkinson_kcl_native_smoke.yaml \
    --features configs/features/mfcc20_stats.yaml \
    --limit 6 --cache-mode shared --paths configs/paths.local.yaml
```

结果写入 `<workspace_root>/inventory/uaspeech/<scan_id>/`。

---

## 四、仓库结构

```text
.
├── src/sddbench/
│   ├── cli.py                 # 命令入口
│   ├── config.py              # 路径绑定与工作区安全（写操作不允许越出 workspace）
│   ├── data/
│   │   ├── schema.py          # 列定义唯一来源
│   │   ├── scan.py            # 文件枚举与分类（只读）
│   │   ├── decoders.py        # probe / decode（soundfile → ffprobe 兜底）
│   │   ├── audit.py           # 并行审计，progress.jsonl 支持断点续扫
│   │   ├── stats.py           # 聚合统计（只报告事实）
│   │   ├── audio.py           # 通道选择/裁剪/重采样/补零（不碰原文件）
│   │   ├── windows.py         # 位置规则、随机种子、view_hash、窗口表
│   │   ├── prepare.py         # 确定性分层抽样 + 准入前的选择规则
│   │   ├── admission.py       # 标签复核、人员隔离、批准清单
│   │   ├── split.py           # 人员隔离的确定性分层划分
│   │   └── adapters/          # 数据集适配器（只解释命名/归属，不碰音频）
│   ├── features/              # MFCC 固定配方 + 分片缓存
│   ├── modeling/              # 特征读取、指标、MLP、实验配置、训练、扫描/汇总、报告
│   ├── results/               # 结果契约、发布白名单、跨数据集 leaderboard
│   └── reporting/             # markdown 报告 + 图 + 作业预检
├── configs/
│   ├── paths.example.yaml     # 路径模板（真实 paths.local.yaml 不入库）
│   ├── datasets/uaspeech_v1.yaml
│   ├── datasets/parkinson_kcl_v1.yaml
│   ├── preprocessing/         # native 小样 / 16 kHz 候选配置
│   ├── features/              # mfcc20_stats.yaml
│   ├── experiments/           # 基线实验配置（数据 ID、划分 ID、模型、训练）
│   └── sweeps/                # 参数网格示例
├── docs/
│   ├── construction_plan.md   # 宏观架构与分阶段施工说明
│   ├── data_layer_plan.md     # 数据层补充设计（UAspeech 先行）
│   ├── datasets/uaspeech.md   # UAspeech 数据集档案与已知疑点
│   ├── datasets/parkinson_kcl.md
│   ├── construction_log.md    # 每部分施工记录（命令、输出、决策）
│   └── acceptance_log.md      # 验收日志
├── artifacts/
│   ├── inventory/...          # 扫描产物：只保留总览统计（见下）
│   ├── prepared/<task>/...    # 准入证据（manifest.csv 本机再生，不入库）
│   ├── splits/<task>/...      # 划分结果 split.csv/json/md
│   └── runs/<run_id>/...      # 模型结果包（权重/预测/指标/图/报告，本机）
├── results_summary/           # 入库的结果归档：<dataset>/<run_id>/ + leaderboard.csv/md
│                              # 只含聚合指标/报告/配置/小图，无权重与逐录音数据
└── tests/                     # 合成数据 fixture 测试
```

### 什么进 Git，什么不进

| 类型 | 是否提交 | 说明 |
| --- | --- | --- |
| 代码、配置模板、文档、测试 | ✅ | 主体内容 |
| `artifacts/**/meta.json`、`statistics.json`、`report.md`、`issues.csv`、`group_statistics.csv`、`length_candidates*.csv`、`figures/*.png` | ✅ | 总览统计，KB 级，可复现结论 |
| `artifacts/label_review/*.csv`、`artifacts/prepared/*/*/admission.md`、`admission.json`、`label_review_status.csv`、`provenance.json` | ✅ | D4 准入证据，小体积，不含音频 |
| `artifacts/splits/*/*/split.json`、`split.md` | ✅ | D5 划分协议与汇总，KB 级 |
| `artifacts/splits/*/*/split.csv` | ❌ | 逐录音的人员/标签归属，1.2MB 且属患者级清单；本机可再生 |
| `artifacts/**/manifest.csv` | ❌ | 可能很大；通过准入后本机可再生，不随仓库分发 |
| `artifacts/**/files.csv` | ❌ | 逐文件清单，73MB，本机可再生 |
| `artifacts/**/progress.jsonl` | ❌ | 断点续扫日志，119MB |
| 原始音频（`*.wav` 等所有音频/压缩包） | ❌ | 患者隐私 + 体积 |
| `configs/paths.local.yaml` | ❌ | 含本机绝对路径 |
| 缓存/权重（`*.npz`、`*.pt`、`*.npy` 等） | ❌ | 体积大；本机可再生 |
| `artifacts/runs/**` | ❌ | 含权重、逐窗口预测与患者级输入快照；只把指标摘要写进文档/汇总 |
| `results_summary/<dataset>/<run_id>/` 的 `metrics.json`、`metrics_per_class.csv`、`report.md`、`config.yaml`、`run_meta.json`、`figures/*.png` | ✅ | D8 归档：小体积、已清洗（无绝对路径/患者级清单），可重算 leaderboard |
| `results_summary/leaderboard.csv`、`leaderboard.md`、`summary.csv`、`summary.md` | ✅ | 聚合指标，KB 级，可由各 run 的 `metrics.json` 重算 |

`.gitignore` 对 `artifacts/` 采用**白名单**：默认全部忽略，只逐个放行上面那份总览清单，
再加一条"大文件兜底"规则，避免以后误提交。

> 与设计文档的一点差异：为让统计总览进入版本库，仓库根即工作区根（`docs/`、`src/`
> 与 `artifacts/` 平级），而不是原来的 `repo/` 子目录 + 仓库外工作区。
> 数据流水线的分层与"数据层对模型层透明"的约定不变。

---

## 五、当前边界（重要）

- **已实现**：`inspect-file`、`stats`（quick/full）、`dataset-report`、
  `extract-features`（窗口索引 + MFCC 统计 + 缓存，支持 `--mic`）、
  `prepare`（标签复核/人员隔离/批准清单）、`split`（按人员隔离的分层划分）、
  `run`（冻结特征 MLP 训练与结果包）、`sweep`（参数网格）、`summarize`（同库内跨运行汇总）、
  `publish`（结果契约校验 + 小体积归档 + 跨数据集 leaderboard）。
- **未实现**：`cache`、`dashboard`；多 seed 置信区间、第二个数据集的训练接入尚未开始。
- **结果契约 v1.0**：train/val/test × window/recording/subject，
  指标含 accuracy、balanced accuracy、sensitivity、specificity、precision、F1、
  AUROC、average precision、log loss、Brier 与混淆矩阵；主指标固定为 test/说话人级。
  缺项会直接报错，不允许静默产出"半套结果"。
- **准入现状**：UAspeech 已按人工确认的目录标签与 **M3** 麦克风通过准入，
  得到 21,126 行 manifest（28 人）；KCL 仍是 `pending`，真实运行被阻断。
  UAspeech 选 M3 是工程基线（M2 在 Healthy 组缺录音），不是研究结论。
- **不做的事**：不自动选观察长度、不自动选采样率、不自动选麦克风、不把目录名当标签、
  不删除任何"疑似异常"样本。这些都需要人工依据 `report.md` 的决策点拍板。
- **状态**：阶段0、D1–D8 均"待人工验收"（见 `docs/acceptance_log.md`）。
  施工细节与命令证据见 `docs/construction_log.md`。

## 六、下一步

1. 人工验收 D1–D8：对照 `artifacts/inventory/<dataset>/<scan_id>/report.md`、
   `artifacts/feature_jobs/<job_id>/inspection.md`、
   `artifacts/prepared/<task_id>/<job_id>/admission.md`、
   `artifacts/splits/<task_id>/<split_id>/split.md`、
   `artifacts/runs/<run_id>/report.md`、缓存 `index.csv` 与
   `results_summary/leaderboard.md` 复核数字与决策点。
2. 多 seed 与不确定性（D9）：同一配置跑 3–5 个 seed，按人员重采样估计置信区间，
   再决定哪些超参差异值得继续追。
3. 接入更多数据集（D10）：每个库只需适配器 + 标签复核 + prepared/split ID，
   再复制一份实验配置；KCL 的标签复核与 ID22 命名问题在接入时综合处理。
4. 汇总页（D11）：把 `results_summary` 渲染成一页可浏览的 HTML/表格。
