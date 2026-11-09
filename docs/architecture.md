# 项目说明书：语音疾病识别基准平台

> 这份文档回答三个问题：**这个项目有哪些模块、每个模块能做什么、参数和产物在哪里。**
> 它不是施工日志（见 `docs/construction_log.md`），也不是验收记录（见 `docs/acceptance_log.md`），
> 而是给“第一次接手的人”和“网页控制台”共用的一份模块地图。

## 1. 一句话定位

把“一个语音数据集”变成“一条可复现的疾病/健康二分类基线”：原始音频 → 扫描统计 →
窗口与特征缓存 → 标签准入 → 按人隔离的划分 → MLP 训练 → 统一指标与结果包 → 跨数据集榜单。

平台刻意把**数据事实**、**人的决策**、**模型实验**分开：

- 代码只负责“可重复的计算”；
- 标签复核、麦克风选择、窗口长度、是否按人隔离，都必须由人显式写进配置；
- 任何一步缺输入就报错或阻断，不猜、不伪造。

## 2. 分层与目录

```
configs/                  所有可调参数的入口（YAML）
  paths.example.yaml      路径绑定模板（真实路径写在 paths.local.yaml，不入库）
  datasets/               数据任务定义：adapter、grouping、标签映射、麦克风选择
  preprocessing/          观察窗语义：采样率、通道、时长、位置、补零策略
  features/               MFCC 配方：n_mfcc、n_mels、帧长、统计量
  experiments/            一次实验：读哪个 prepared/split、模型结构、训练超参
  sweeps/                 参数网格

src/sddbench/
  config.py               路径解析、workspace 子目录约定
  data/                   D1–D5：扫描、审计、统计、准入、划分
    adapters/             数据集适配器（uaspeech / parkinson_kcl）
  features/               D3：MFCC 计算与分片缓存
  modeling/               D6–D7：特征装载、MLP、训练、指标、报告、sweep
  results/                D8–D9：结果契约、归档与榜单、多 seed 聚合
  reporting/              各类 Markdown 报告渲染
  cli.py                  所有命令行子命令

docs/                     方案、施工日志、验收日志、数据集说明、本说明书
artifacts/                本机工作区：中间数据与 run 包（大文件不入库）
results_summary/          入库的小体积结果归档与榜单
tests/                    合成数据测试，不碰真实患者音频
```

## 3. 完整流水线

| 步骤 | 命令 | 输入 | 产物 | 人工决策 |
| --- | --- | --- | --- | --- |
| 0 路径绑定 | 编辑 `configs/paths.local.yaml` | 数据根、workspace 根 | 无 | 数据根必须显式 |
| 1 扫描与统计 | `stats --mode full` | 原始数据根 | `inventory/<dataset>/<scan_id>/` | 无（纯事实） |
| 2 观察窗 | `extract-features --dry-run` | 扫描 + 预处理配置 | `feature_jobs/<job_id>/windows.csv` | 采样率、通道、4 s、head/tail |
| 3 特征缓存 | `extract-features --features ...` | 窗口索引 + MFCC 配置 | `cache/features/mfcc_stats/<sr>/` | n_mfcc、n_mels、帧长、统计量 |
| 4 标签准入 | `prepare` | 扫描 + 标签复核 CSV + `grouping` | `prepared/<task_id>/<job_id>/manifest.csv` | 标签、麦克风、是否按人隔离 |
| 5 划分 | `split --seed N` | manifest | `splits/<task_id>/<split_id>/split.csv` | 比例、seed、隔离单位 |
| 6 训练 | `run --config ...` | 缓存 + 划分 + 实验配置 | `runs/<run_id>/` | 模型、超参、评价单位 |
| 7 网格/汇总 | `sweep` / `summarize` | 实验配置 + sweep 配置 | 多个 run + `summary.csv` | 扫哪些参数 |
| 8 归档/榜单 | `publish` | 本地 run 包 | `results_summary/leaderboard.*` | 无（按契约白名单） |
| 9 多 seed 聚合 | `aggregate` | 本地 run 包 | `results_summary/aggregates.*` | bootstrap 次数 |
| 10 网页控制台 | `serve`（D12） | 上面所有产物 | 浏览器页面 | 全部可视化 |

**关键点：第 2、3 步的配置改了，必须重新生成缓存；第 6 步只读缓存，不重新算 MFCC。**
网页控制台会明确标出“改这一项需要重算特征”。

## 4. 模块清单

### 4.1 基础设施

| 模块 | 代码文件 | 职责 | 关键入口 |
| --- | --- | --- | --- |
| 路径与配置 | `src/sddbench/config.py` | 读取 `paths.local.yaml`；把 `inventory/prepared/splits/runs/...` 映射到 workspace | `load_paths`、`workspace_dir`、`dataset_root` |
| 列名契约 | `src/sddbench/data/schema.py` | `files.csv` / `manifest.csv` / `issues.csv` 的列名常量 | `FILES_COLUMNS`、`MANIFEST_COLUMNS` |
| 命令行 | `src/sddbench/cli.py` | 所有子命令的参数解析与编排 | `build_parser`、`main` |

### 4.2 数据层（D1–D5）

| 模块 | 代码文件 | 职责 | 产物/关键点 |
| --- | --- | --- | --- |
| 扫描 | `data/scan.py` | 递归列出文件，按扩展名判定音频候选 | `files.csv` 的初始行 |
| 解码与探测 | `data/decoders.py` | `soundfile`/`ffmpeg` 探测与解码、sha256、工具版本 | 采样率、通道、时长、编码 |
| 审计 | `data/audit.py` | quick/full 两种模式；full 会真实解码；支持断点续扫 | `progress.jsonl`、`files.csv` |
| 统计 | `data/stats.py` | 计数、时长分位、按目录/麦克风分组、候选窗口长度 | `statistics.json`、`length_candidates*.csv` |
| 数据集适配器 | `data/adapters/*.py` | **只解释数据**：谁属于谁、候选标签在哪、哪些是派生副本 | `group_id`、`subject_id`、`label` 的候选值 |
| 采样 | `data/prepare.py` | 按类别/目录做确定性分层抽样 | 小规模 smoke 用 |
| 准入 | `data/admission.py` | 人工复核表校验、麦克风/版本选择、origin 去重、`grouping` | `manifest.csv`、`admission.json/md` |
| 划分 | `data/split.py` | 按 `group_id` 分层、组隔离的 60/20/20 | `split.csv`、`split.json`、`split.md` |
| 窗口 | `data/windows.py` | 采样率策略、通道选择、裁剪/补零、窗口索引、指纹 | `windows.csv`、`view_hash` |

**适配器接口**（`data/adapters/base.py`）：`collect(root, task_config, inventory) -> AdapterResult`。
新数据集只需要实现一个适配器并在 `registry.py` 注册，**不需要改扫描、审计、划分代码**。

### 4.3 特征层（D3）

| 模块 | 代码文件 | 职责 |
| --- | --- | --- |
| MFCC 配方 | `features/mfcc.py` | 校验并归一化配置；用 torchaudio 在有效帧上算 MFCC 与 mean/std；记录实现版本 |
| 缓存 | `features/cache.py` | 按 `(特征配置, 采样率, 视图指纹)` 分目录；`meta.json + index.csv + shard_*.npz`；支持增量与碰撞检测 |

- 输入：`windows.csv` 的 `window_id` 与 `view_hash`；
- 输出：每个窗口 40 维（`n_mfcc=20` 的 mean+std）；
- **缓存键**：特征配置哈希 + 采样率 + 预处理指纹。任何一项变化都会生成新缓存目录。

### 4.4 模型层（D6–D7）

| 模块 | 代码文件 | 职责 |
| --- | --- | --- |
| 实验配置 | `modeling/experiment.py` | 默认值、YAML 合并、`--set key=value`、网格展开、配置哈希 |
| 特征装载 | `modeling/features.py` | 读缓存、按 `sample_id` 对齐划分、校验特征维度与指纹 |
| 模型 | `modeling/mlp.py` | `Linear → ReLU → Dropout → Linear`，输入维度自动推导 |
| 指标 | `modeling/metrics.py` | 混淆矩阵、balanced accuracy、sensitivity/specificity、AUROC、AP、log loss、Brier |
| 训练 | `modeling/trainer.py` | train-only 标准化、subject-balanced 权重、早停、三种粒度预测、结果包 |
| 报告 | `modeling/report.py` | `report.md` + 混淆矩阵/ROC/分数分布/学习曲线 |
| 网格 | `modeling/sweep.py` | 按配置网格逐个 run，`jobs.csv`；跨 run 汇总 |

**三种评价粒度**（每次 run 都同时写）：

1. `window`：每个 4 s 窗口一条预测；
2. `recording`：同一录音的窗口概率取平均；
3. `subject`：同一人的录音概率取平均（有说话人时的主单位）。

`primary_unit` 由 `grouping` 决定：`subject` 库用 `subject`，`file` 库用 `recording`。

### 4.5 结果层（D8–D9）

| 模块 | 代码文件 | 职责 |
| --- | --- | --- |
| 契约 | `results/contract.py` | v1.1：必需指标、单位、`primary_unit`、字段白名单 |
| 归档/榜单 | `results/publish.py` | 只复制白名单文件；按 `primary_unit` 生成跨数据集榜单 |
| 多 seed | `results/aggregate.py` | 配置签名分组、跨 seed 均值/区间、test 分组 bootstrap、换划分敏感性、箱线图 |

### 4.6 报告层

| 模块 | 代码文件 | 产出 |
| --- | --- | --- |
| 数据集报告 | `reporting/dataset_report.py` | `inventory/<dataset>/<scan_id>/report.md` + 图 |
| 作业巡检 | `reporting/job_report.py` | `feature_jobs/<job_id>/inspection.md` |
| 准入门槛 | `reporting/admission_report.py` | `prepared/.../admission.md` |
| 划分报告 | `reporting/split_report.py` | `splits/.../split.md` |

### 4.7 网页控制台（D12）

| 模块 | 代码文件 | 职责 |
| --- | --- | --- |
| 参数注册表 | `web/schema.py` | 所有可调参数的键、类型、默认值、范围、影响范围（窗口/特征/划分/训练）；生成 YAML 前统一校验 |
| 目录服务 | `web/catalog.py` | 只读汇总 `configs/datasets`、inventory、prepared、splits、runs、`results_summary` |
| HTTP 服务 | `web/server.py` | 标准库 `http.server`；GET schema/catalog/jobs、POST experiment/run/split、图床代理 |
| 前端 | `web/static/index.html`、`style.css`、`app.js` | 原生 HTML/CSS/JS，无构建步骤；左右布局 + 五个页签 + 日志浮层 |

接口：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/schema` | 参数 schema |
| GET | `/api/catalog` | 数据集、prepared、split、run、榜单、聚合 |
| POST | `/api/experiment` | 校验参数并写 `configs/experiments/generated/<id>/` |
| POST | `/api/run` | 同上，再以子进程启动 `sddbench run`（可 `dry_run`） |
| POST | `/api/split` | 同步调用 `sddbench split`，返回新 `split_id` |
| GET | `/api/jobs/<id>` | 任务状态与日志尾部 |
| GET | `/media/run/<run_id>/<png>` | 只允许 run 目录 `figures/*.png` |

**写边界**：网页只写 `artifacts/web/` 与 `configs/experiments/generated/`；
不覆盖手工基线、不删除 run、不扫描原始数据根。默认只监听 `127.0.0.1`，无鉴权。

## 5. 参数地图

平台有四层配置，网页控制台会把它们分组展示（表单由 `web/schema.py` 的同一份注册表生成）。

### 5.1 数据任务 `configs/datasets/<dataset>_v1.yaml`

| 参数 | 作用 | 可选值/注意 |
| --- | --- | --- |
| `adapter` | 用哪个适配器解释文件 | `uaspeech`、`parkinson_kcl` |
| `grouping` | 隔离单位 | `subject`（默认）/ `file`（无说话人） |
| `selection.mic` | 只保留哪个麦克风 | UAspeech 必须显式指定 |
| `selection.one_per_origin` | 同一 origin 多麦克风只留一条 | 布尔 |
| `labels.mapping` | 目录/候选标签 → 正式标签 | 必须人工复核 |
| `labels.class_order` / `positive_class` | 类别顺序与正类 | 影响指标方向 |
| `evaluation_unit` | 期望的主评价单位 | `subject` / `recording` |

### 5.2 预处理 `configs/preprocessing/*.yaml`

| 参数 | 作用 | 当前支持 |
| --- | --- | --- |
| `sample_rate_policy` | 原生或重采样 | `native` / `resample` |
| `target_sample_rate` | 重采样目标 | `resample` 时必填 |
| `channel_policy` / `channel_index` | 通道选择 | `select` / `require_mono` |
| `observe_seconds` / `window_seconds` | 观察长度 | 当前必须相等 |
| `position` | 取头部/尾部/随机 | `head` / `tail` / `random`（随机需 `crop_seed`） |
| `short_policy` | 不足长度怎么办 | 仅 `right_zero_pad` |
| `frame_ms` / `hop_ms` | 窗口与步长 | 正数 |
| `amplitude_normalization` | 幅度归一化 | 当前仅 `none` |

### 5.3 特征 `configs/features/*.yaml`

| 参数 | 作用 | 当前约束 |
| --- | --- | --- |
| `n_mfcc` | MFCC 维数 | 正整数，≤ `n_mels` |
| `n_mels` | Mel 滤波器数 | 正整数 |
| `frame_ms` / `hop_ms` | 帧长/帧移 | 正数；`n_fft` 自动取 2 的幂 |
| `f_min_hz` / `f_max_policy` | 频率范围 | `nyquist` / `fixed` |
| `stats` | 统计量 | `[mean, std]` |
| `std_ddof` | 标准差自由度 | 0 |
| `center`、`log_mels`、`n_fft_policy` | 数值实现细节 | 已锁定，改前要评审 |

### 5.4 实验 `configs/experiments/*.yaml`

| 参数 | 作用 | 默认 |
| --- | --- | --- |
| `dataset` / `prepared_id` / `split_id` | 绑定数据与划分 | 必填 |
| `features.config` / `features.preprocess` | 读哪套缓存 | 必填 |
| `model.hidden_dims` | 隐藏层宽度 | `[128]` |
| `model.dropout` / `activation` | 正则与激活 | `0.2` / `relu` |
| `training.seed` | 训练随机种子 | `0` |
| `training.epochs` / `batch_size` | 训练规模 | `100` / `128` |
| `training.learning_rate` / `weight_decay` | 优化器 | `1e-3` / `1e-4` |
| `training.early_stopping_patience` | 早停 | `15` |
| `training.class_weighting` | 权重策略 | `subject_balanced` |
| `evaluation.unit` | 主评价单位 | `subject`（`file` 库必须 `recording`） |

划分的 seed 与比例不在实验配置里，而在 `split` 命令上（`--seed`、`--train-frac`、`--val-frac`、`--test-frac`）；
训练 seed 在实验配置里。两者互相独立，这是 D9 的核心。

## 6. 工作区产物

```
artifacts/
  inventory/<dataset>/<scan_id>/
    meta.json  files.csv  statistics.json  issues.csv  report.md  figures/
  feature_jobs/<job_id>/
    config.yaml  feature_config.yaml  windows.csv  issues.csv  inspection.md
    previews/  cache/features/mfcc_stats/<sample_rate>/
  prepared/<task_id>/<job_id>/
    manifest.csv  admission.json  admission.md  label_review_status.csv
    issues.csv  task.yaml  provenance.json
  splits/<task_id>/<split_id>/
    split.csv  split.json  split.md
  runs/<run_id>/
    config.yaml  run.json  history.csv  metrics.json  metrics_per_class.csv
    best.pt  last.pt
    predictions_windows.csv  predictions_recordings.csv  predictions_subjects.csv
    report.md  figures/
results_summary/
  leaderboard.csv|md          跨数据集榜单
  aggregates.csv|md           多 seed 汇总
  split_sensitivity.csv|md    换划分敏感性
  <dataset>/<run_id>/         白名单归档（无权重、无逐条预测）
  <dataset>/aggregates/       每组的多 seed 明细与图
```

## 7. 扩展点

1. **新数据集**：写一个 adapter → 建 `configs/datasets/<id>_v1.yaml` → 在 `paths.local.yaml` 加数据根 →
   跑 `stats` → 人工填标签复核 → `prepare` → `split` → 复制实验配置。
   没有说话人时把 `grouping` 设为 `file`，主单位自动变 `recording`。
2. **新特征**：在 `features/` 加实现并注册到缓存键；实验配置只换 `features.config`。
3. **新模型**：在 `modeling/` 加模型并让 `model.name` 分发；输入维度仍从缓存推导。
4. **新指标**：在 `modeling/metrics.py` 与 `results/contract.py` 同步加，契约升版本号。

## 8. 已知限制

- 目前只有二分类；多分类/回归需要扩契约版本。
- test 单元个位数时，指标波动很大（D9 已量化：换划分 range 0.167）。
- `grouping=file` 无法排除“同一个人跨划分”，指标偏乐观，报告与榜单已注明。
- UAspeech 的 `C` 前缀标签疑点、KCL 的 ID22 命名冲突仍需人工核对后才能对外引用数字。
- 训练为单线程 CPU；GPU 下的 seed 方差尚未验证。

## 9. 命令速查

```bash
sddbench stats --dataset uaspeech --mode full --paths configs/paths.local.yaml
sddbench extract-features --dataset uaspeech --source <scan_id> \
    --preprocess configs/preprocessing/uaspeech_native_4s_head.yaml \
    --features configs/features/mfcc20_stats.yaml --cache-mode shared --paths ...
sddbench prepare --dataset uaspeech --source <scan_id> --paths ...
sddbench split --dataset uaspeech --prepared <prepared_id> --seed 0 --paths ...
sddbench run --config configs/experiments/uaspeech_mfcc_mlp_baseline.yaml --paths ...
sddbench sweep --config ... --grid configs/sweeps/uaspeech_seed_ci.yaml --paths ...
sddbench summarize --runs artifacts/runs --out results_summary
sddbench publish  --runs artifacts/runs --out results_summary
sddbench aggregate --runs artifacts/runs --out results_summary
sddbench serve --paths configs/paths.local.yaml --port 8765   # D12
```
