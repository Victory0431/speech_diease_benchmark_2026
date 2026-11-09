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

## D3 —— MFCC 统计量与分片缓存（待施工）

## D4 —— 标签复核、批准清单与第二数据集（待施工）
