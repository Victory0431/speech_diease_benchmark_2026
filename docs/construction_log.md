# 施工日志（按部分推进）

记录每个数据层部分的**目标、实际命令、真实输出、关键决策与偏差**。
与 `docs/acceptance_log.md` 的区别：验收日志面向"是否通过"，
本日志面向"当时到底做了什么、为什么这样做、证据在哪"。

工作区：`/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/2026_new_mission/speech_disease_benchmark_workspace20261109`
远端：`https://github.com/Victory0431/speech_diease_benchmark_2026`

---

## D1 —— 扫描、解码与全库统计（已完成并推送）

- **目标**：在写任何裁剪/特征代码前，先拿到 UAspeech 逐文件的真实事实。
- **实现**：`scan` / `decoders` / `adapters/uaspeech` / `audit` / `stats` /
  `reporting/dataset_report`，CLI：`inspect-file`、`stats`、`dataset-report`。
- **实跑证据**：scan_id `20261109T171421+0800_uaspeech_full_dd025653`；
  143,290 条 WAV/PCM_16/16 kHz/单声道全部解码成功；时长 P50=2.06 s、P90=3.88 s；
  合计约 100 小时。详见 `docs/acceptance_log.md` 与
  `artifacts/inventory/uaspeech/<scan_id>/report.md`。
- **遗留**：目录名与 `C` 前缀的标签疑点、Healthy 组 M2 麦克风缺录，均保持 `pending`。

---

## D2 —— 观察窗口索引（通道选择 / 裁剪 / 重采样 / 补零）

### 目标
把"选哪一段声音"变成可复现、可审计、与 D1 强耦合的索引表；此阶段**不计算 MFCC**。

### 实现
| 文件 | 作用 |
| --- | --- |
| `src/sddbench/data/audio.py` | 通道选择（绝不自动平均）、按原始采样点裁剪、锁定重采样、右补零、毫秒→采样点/`n_fft` 几何 |
| `src/sddbench/data/windows.py` | 位置规则（head/center/tail/fixed_random）、SHA256 派生随机种子、`view_hash`、窗口表构建 |
| `src/sddbench/data/prepare.py` | 跨目录组/说话人的确定性分层抽样（先给每个说话人一条，再循环） |
| `src/sddbench/reporting/job_report.py` | `inspection.md`（不重算统计、不猜值） |
| `configs/preprocessing/uaspeech_native_smoke.yaml` | 技术验证：native Hz、head、2.0 s |
| `configs/preprocessing/speech_common16k_candidate.yaml` | 主协议**候选**：resample 16 kHz、head、4.0 s（未批准） |
| `tests/test_crop_padding.py` | 坐标/补零/确定性/通道/重采样/端到端（含原始文件哈希不变） |

CLI 新增 `extract-features`：
`--source <scan_id>` 必须指向 **completed 的 full 扫描**，逐条重新解码并校验
`files.csv` 里的 `sha256`；不一致直接拒绝该条，避免"统计的是旧文件、特征来自新文件"。

### 关键计算规则（写死，不靠默认值）
- `K = round_half_up(window_seconds × sr)`，`D = max(N − K, 0)`；
  head=0，center=⌊D/2⌋，tail=D，fixed_random∈[0,D]（PCG64）。
- `fixed_random` 种子 = `SHA256(algorithm_version|global_seed|sample_id)` 前 8 字节；
  与文件枚举顺序、Python `hash()` 随机化无关。
- 短音频：所有位置都保留整条，`pad = K − N`，补零数与有效样本数分开记录；
  `padding_ratio = pad / K`。
- 补零只改变张量容纳长度，不循环、不复制、不伪造静音。
- native 策略遇到**本次选中样本**混合 Hz 且 `allow_mixed_native=false`：
  作业状态写 `blocked_mixed_sample_rate`，退出码 3，不偷偷批量提取。
- 重采样锁定 `torchaudio.functional.resample` + `sinc_interp_hann`
  (`lowpass_filter_width=64, rolloff=0.9, beta=None`)；输出长度与
  `round_half_up(N×target/src)` 相差 >1 采样点直接报错，±1 内显式对齐并记录。
- 前端几何按**实际采样率**推导：`win=round(25 ms×sr)`、`hop=round(10 ms×sr)`、
  `n_fft=next_power_of_two(win)`；16 kHz 为 400/160/512。
- `view_hash` 覆盖预处理参数、样本 ID、源文件哈希、解析后的坐标/长度/通道/重采样信息；
  改位置、长度、采样率或源文件都会改变它。

### 实跑（2026-11-09）
1. 12 条技术验证（native / head / 2 s）：
   - job：`artifacts/feature_jobs/20261109T181848+0800_uaspeech_views_head_2s_ecc135d0/`
   - 选中 12 条 = Disorder 6 + Healthy 6，12 个不同说话人；全部 16 kHz；
   - 状态 `planned`，窗口 12 条，派生预览 6 个（`previews/*.wav`，PCM_16）。
2. 4 条 16 kHz 候选（resample / head / 4 s）：
   - job：`artifacts/feature_jobs/20261109T181900+0800_uaspeech_views_head_4s_da65659f/`
   - 源均为 16 kHz，故 `resample_applied=false`（预期：本库无需重采样）；
   - 4 s 目标 64,000 点；短于 4 s 的样本补零数记录在 `windows.csv`。
3. 测试：`python -m pytest tests -q` → **32 passed**。
4. 原始音频只读：端到端测试断言处理前后文件 SHA256 完全一致。

### 决策与偏差
- **小样长度 2.0 s**：仅用于快速技术验证（P50=2.06 s），不是主协议。
- **候选主协议 4.0 s**：覆盖到 P90=3.88 s，保留 91.4% 有效声音、总补零 42.6%；
  仍未获批，正式训练前需人工确认。
- 抽样策略修正过一次：最初实现按组顺序取，12 条全落在 Disorder；
  已改为"组间按说话人槽位交错"，现为 6+6。
- 运行环境：Python 3.13；`torch 2.15.0+cpu` / `torchaudio 2.11.1+cpu`。
  CFG 加入了 `features` 可选依赖；`job.json` 记录工具版本。
- `--features` 目前显式拒绝（退出码 2），不伪造 MFCC 结果；D3 接入。

---

## D3 —— MFCC 统计量与分片缓存（已完成并推送）

### 目标
在 D2 已锁定的观察窗口上计算固定配方 MFCC，并建立可复用、可失效、可断点续算的特征缓存；
此阶段仍只做**技术验证**（`technical_only=true`），不产生训练标签。

### 实现
| 文件 | 作用 |
| --- | --- |
| `src/sddbench/features/mfcc.py` | 固定配方解析/校验、`feature_hash`、`feature_key`、transform 缓存、逐窗 MFCC 统计 |
| `src/sddbench/features/cache.py` | `FeatureCache`（meta/index/shard 原子写）与 `FeatureCacheManager`（按实际 Hz 多缓存懒加载） |
| `src/sddbench/features/__init__.py` | 包初始化 |
| `configs/features/mfcc20_stats.yaml` | 20 维 MFCC + mean/std(ddof=0) → 40 维 |
| `tests/test_feature_cache.py` | 形状/有限性/补零不变性/缓存复用与失效/CLI 端到端/dry-run |

CLI `extract-features` 的 `--features` 从"显式拒绝"改为真实计算；新增
`--cache-mode run|shared`（默认 run，位于 job 目录内；shared 位于
`artifacts/cache/features/<config_name>/<feature_key>/`）、`--no-verify-source-hash`。
单遍解码完成窗口构建与特征提取（`build_view_index(on_view=...)`），不重复读音频。

### 锁定配方（`mfcc20_stats`）
- 25 ms / 10 ms，periodic Hann，`center=false`，`n_fft=next_power_of_two(win)`（16 kHz 为 512）。
- `n_mfcc=20`、`n_mels=64`、`power=2`、`htk`、无 mel 归一化、`f_min=0`、`f_max=nyquist`。
- `log_mels=true`，等价 `log(mel_power + 1e-6)`；DCT-II ortho，含 c0。
- 统计：仅有效声音帧（补零帧不进入），逐系数 `mean` + `std(ddof=0)` → 40 维 float32。
- 帧数公式与 torchaudio 对齐：`n_frames = 1 + (N - n_fft) // hop`；`N < n_fft` 报 `too_short_for_frontend`。
- `feature_hash` 覆盖配方、实现版本（torch/torchaudio 版本、log offset、DCT 约定）、
  以及**实际采样率推导出的**帧长/帧移/`n_fft`；只改统计图或 P95 不会让缓存失效。

### 缓存布局
```text
artifacts/cache/features/mfcc_stats/mfcc_stats_<feature_hash[:16]>/
├── meta.json      # 配方、实现版本、每个采样率的维度、完成标记
├── index.csv      # window_id / sample_id / feature_hash / shard / row
└── shard_000.npz  # x: float32[N, 40]
```
- 分片按 2048 条切分；写盘先 `.tmp.npz` 再 `os.replace`，崩溃不会留下半个分片。
- 存在 `meta.json` + `index.csv` 即视为可复用；删除缓存目录可全量重算。
- 缓存目录在 `.gitignore` 白名单之外，不进入 Git。

### 实跑（2026-11-09）
命令：
```bash
sddbench extract-features --dataset uaspeech \
  --source 20261109T171421+0800_uaspeech_full_dd025653 \
  --preprocess configs/preprocessing/uaspeech_native_smoke.yaml \
  --features configs/features/mfcc20_stats.yaml \
  --limit 12 --cache-mode shared --previews 6 \
  --paths configs/paths.local.yaml
```
真实输出：
- `windows built=12/12`，`features: new=12 cached=0 failed=0`，状态 `completed`；
- job：`artifacts/feature_jobs/20261109T182847+0800_uaspeech_views_head_2s_ecc135d0/`；
- 缓存：`mfcc_stats_efe488280a7dc9a9`，16000 Hz、dim=40、12 条 / 1 个分片；
- `inspection.md` 第 6 节显示 MFCC 配方、维度、命中/新增/失败与缓存路径。
- 复跑同一命令：`new=0 cached=12`，证明缓存命中；改 `n_mfcc=40` 产生新 `feature_key`（80 维）。
- 测试：`python -m pytest tests -q` → **40 passed**。

### 决策与偏差
- 单遍解码（窗口 + 特征同遍完成）优先于"先写窗口表再二次解码"，代价是 D2 的
  纯窗口 dry-run 与 D3 实际计算共用同一入口；`--dry-run` 仍可只看窗口不计算特征。
- `window_id` 由 `scan_id|sample_id|w0|view_hash[:12]` 组成；D2 初版漏了 scan 前缀，
  已修正，保证换源扫描不会误命中旧缓存。
- 缓存键只按"配方 + 实际 Hz"区分；同一配方不同窗口天然由 `window_id` 区分。
- 补零不改变 MFCC 统计（以 3 s vs 4 s 补零到 4 s 的单测验证），但真正改变截取内容会改变特征。
- `feature_hash` 不含数据集名与路径，便于第二库复用同一缓存代码。

---

## D4 —— 标签复核、批准清单与第二数据集（已完成并推送）

### 目标
1. 建立"人工复核 -> 批准清单"的门禁：没有逐人 `confirmed` 标签，绝不产出 `manifest.csv`。
2. 落实麦克风/origin/人员隔离规则，保证后续 split 可以按人隔离。
3. 接入第二个数据集（选定 Parkinson KCL 2017），复用全部通用代码。
4. 保持原始数据只读，不自动翻转标签、不自动选麦克风。

### 第二库选择
检查 `fresh_datasets` 后有两个候选：
- `EATD`：485 条抑郁/健康语音，16 kHz 混合单/双声道；文件名只有情绪词
  (`positive/negative/neutral_out_N`)，**没有说话人 ID**，无法做人员隔离。
- `Parkinson_KCL_2017`：73 条长录音，44.1 kHz/PCM_24/单声道；文件名含 `ID<nn>`
  与 `hc/pd`，可解析 speaker，适合作为第二适配器。

选定 **Parkinson_KCL_2017**（不下载、不改名、不移动任何文件）。

### 实现
| 文件 | 作用 |
| --- | --- |
| `src/sddbench/data/schema.py` | 新增 `LABEL_REVIEW_COLUMNS`、`MANIFEST_COLUMNS` 与 subject/group/label 等列常量 |
| `src/sddbench/data/admission.py` | 复核表加载/模板生成/覆盖检查、mic+origin 选择、人员隔离、manifest 指纹 |
| `src/sddbench/data/adapters/parkinson_kcl.py` | 解析 `ID<nn>_?hc/pd_...`；只产出候选事实与告警 |
| `src/sddbench/data/adapters/registry.py` | 注册 `parkinson_kcl` |
| `src/sddbench/reporting/admission_report.py` | 生成中文 `admission.md` |
| `src/sddbench/cli.py` | 新增 `prepare` 命令；`prepare` 从"未实现"名单移除 |
| `configs/datasets/uaspeech_v1.yaml` | 增加 `selection`（mic 必须显式指定，`one_per_origin=true`） |
| `configs/datasets/parkinson_kcl_v1.yaml` | 第二库任务定义（无 mic 维度） |
| `configs/preprocessing/parkinson_kcl_native_smoke.yaml` | 44.1 kHz / head 4 s 技术验证 |
| `tests/test_admission.py` | 7 个测试：KCL 命名、缺复核、pending、批准路径、mic 阻断、subject 冲突、第二库复用 |
| `docs/datasets/parkinson_kcl.md` | 第二库档案与已知问题 |

### 准入规则（硬性）
1. `label_review.csv` 必须存在，列固定为
   `speaker_token,directory_group,confirmed_subject_id,confirmed_label,evidence,review_status`。
2. 本次选择涉及的每位 speaker 都必须是 `review_status=confirmed`，且 subject/label 非空；
   缺失、pending、confirmed 但字段为空都阻断。
3. 同一 `confirmed_subject_id` 不得出现两个 label，也不得绑定两个 speaker token。
4. UAspeech 有 7 麦克风；必须显式 `selection.mic`（或 `--mic`），否则阻断。
   选定后每个 `speaker|block|prompt` origin 必须只剩一条。
5. KCL 无 mic 维度；`require_mic=false`。
6. 任一 origin 归到两个 subject/label 都阻断。
7. 通过后 `group_id = <dataset>:<subject_id>`，`label_source` 记录复核文件行号；
   `manifest.csv` 才允许写出。

### 实跑（2026-11-09）
第二库全库统计：
```bash
sddbench stats --dataset parkinson_kcl --mode full --workers 4 --paths configs/paths.local.yaml
```
- scan：`20261109T183508+0800_parkinson_kcl_full_a83761e1`
- 真实输出：`decoded 73/73, failed 0, parse_failed 0`，15.0 s
- 73 条 WAV/PCM_24/44100 Hz/单声道；P50 141.3 s，P90 178.0 s，总计 10,318.5 s；
  Healthy 42 条/21 人，Parkinson 31 条/16 人。

第二库特征技术验证：
```bash
sddbench extract-features --dataset parkinson_kcl \
  --source 20261109T183508+0800_parkinson_kcl_full_a83761e1 \
  --preprocess configs/preprocessing/parkinson_kcl_native_smoke.yaml \
  --features configs/features/mfcc20_stats.yaml \
  --limit 6 --cache-mode shared --previews 3 --paths configs/paths.local.yaml
```
- job：`20261109T183546+0800_parkinson_kcl_views_head_4s_539b0061`
- `windows built=6/6`，`features new=6 cached=0 failed=0`，状态 `completed`
- 44.1 kHz 前端几何：`win_length=1103`、`hop_length=441`、`n_fft=2048`；MFCC 40 维
- 共享缓存 key：`mfcc_stats_7c1bdbbeb6594e5c`

UAspeech 准入（生成模板 + 尝试批准）：
```bash
sddbench prepare --dataset uaspeech \
  --source 20261109T171421+0800_uaspeech_full_dd025653 \
  --write-review-template --paths configs/paths.local.yaml
```
- 模板：`artifacts/label_review/uaspeech_label_review.csv`（28 人，全部 `pending`）
- 输出：`artifacts/prepared/uaspeech_v1/20261109T183604+0800_uaspeech_prepare/`
- 真实结果：`status=blocked_label_review_incomplete`，退出码 3，未写 `manifest.csv`
- 阻断：28 人未确认；未选 mic；21,382 个 origin 在未选 mic 时有 7 条重复

KCL 准入：
```bash
sddbench prepare --dataset parkinson_kcl \
  --source 20261109T183508+0800_parkinson_kcl_full_a83761e1 \
  --write-review-template --paths configs/paths.local.yaml
```
- 模板：`artifacts/label_review/parkinson_kcl_label_review.csv`（37 人，全部 `pending`）
- 输出：`artifacts/prepared/parkinson_kcl_v1/20261109T183608+0800_parkinson_kcl_prepare/`
- 真实结果：`status=blocked_label_review_incomplete`，退出码 3，未写 `manifest.csv`
- 额外阻断：`ID22_hc_0_0_0` 与 `ID22hc_0_0_0` 归一化后 origin 相同但时长/哈希不同，
  记为 `multiple_rows_per_origin`，等待人工确认两者关系

测试：`python -m pytest tests -q` → **47 passed**。
批准路径由 `tests/test_admission.py::test_confirmed_review_writes_manifest` 覆盖：
合成复核表全部 confirmed 后，`prepare` 写出 3 行、3 个 subject、唯一 origin 的
`manifest.csv`，并记录 `label_source` 行号。

### 决策与偏差
- 真实数据**不允许**由施工者代填 `confirmed` 标签，因此两个库的准入都停在阻断态；
  这不是未完成，而是 D4 设计要求的"没有证据就不出清单"。
- 第二库选 KCL 而不是 EATD：EATD 缺说话人 ID，无法满足人员隔离；KCL 的 `ID<nn>` 可直接分组。
- UAspeech 的 `selection.mic` 保持 `null`：D1 已发现 Healthy 组 M2 覆盖异常，
  选哪个麦克风是研究决策，不能由代码代选。
- KCL 的 `hc/pd` 只作为候选：目录与文件名词元一致仍不等于人工确认。
- `ID22` 命名冲突不自动去重：两条时长/哈希不同，不能假定是同一条录音。
- `manifest.csv` 不提交 Git（可能很大）；`admission.md/json`、复核模板等小体积证据提交。
- `.gitignore` 新增 KCL 统计白名单与 D4 准入证据白名单，仍不包含任何音频/特征缓存。
