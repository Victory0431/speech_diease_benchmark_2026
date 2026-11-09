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

## D4 —— 标签复核、批准清单与第二数据集（施工中）
