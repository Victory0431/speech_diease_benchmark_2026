# 验收日志

每条记录：阶段、日期、验收人、执行命令、通过项、未通过项、证据路径、结论、代码提交号。
**未验收通过的阶段不得标记为已启动后续阶段。**

## 仓库说明（2026-11-09）

- 远端仓库：`https://github.com/Victory0431/speech_diease_benchmark_2026`
- 与设计文档的差异：为让 D1 总览统计进入版本库，仓库根上移到工作区根
  （`src/ tests/ configs/ docs/` 与 `artifacts/` 平级），不再是 `repo/` 子目录 + 仓外工作区。
  数据流水线的分层与"数据层对模型层透明"的约定不变。
- 提交策略：`artifacts/` 白名单，仅 `meta.json`、`statistics.json`、`report.md`、`issues.csv`、
  `group_statistics.csv`、`length_candidates*.csv`、`figures/*.png` 入库；
  `files.csv`（73MB）、`progress.jsonl`（119MB）、原始音频、`configs/paths.local.yaml` 一律排除。

## 阶段0（最小基础）——待验收

- 日期：2026-11-09
- 施工内容：仓库骨架、pyproject/CLI 安装、paths 绑定与工作区安全检查、
  README、设计文档入库、`.gitignore`（含 paths.local.yaml 排除）。
- 已知限制：仅 D1 相关命令可用；其余命令显式拒绝。
- 验收步骤建议：
  1. `pip install -e .[test]` 成功；
  2. `sddbench --help` 可运行，未实现命令返回非零且不假装成功；
  3. `git status` 不含真实数据、音频或 paths.local.yaml；
  4. 原始数据目录未被写入/修改。
- 结论：待人工验收
- 代码提交号：未提交（验收后执行 `chore: initialize benchmark repository`）

## D1（单文件检查与 UAspeech 全库统计）——待验收

- 日期：2026-11-09
- 施工内容：scan / decoders / uaspeech 适配器 / audit / stats / dataset_report、
  CLI（inspect-file、stats、dataset-report）、合成 fixture 测试（22 项全绿）。
- 实际执行：
  - `sddbench inspect-file --dataset uaspeech --relative-path Disorder/CF02_B1_C1_M2.wav --paths configs/paths.local.yaml`
    → WAV / PCM_16 / 16 kHz / 单声道 / 2.046 s / peak 1.0（满幅事实）；
  - `sddbench stats --dataset uaspeech --mode full --paths configs/paths.local.yaml --workers 32`
    → 全库 143,290 文件：发现=候选音频 143,290 + zip 0 + 旁车 0（本库无）；
    probe OK 143,290；解码成功 143,290、失败 0；解析失败 0。
- scan_id：`20261109T171421+0800_uaspeech_full_dd025653`（目录指纹 dd025653…）
- 证据路径：`<workspace>/inventory/uaspeech/20261109T171421+0800_uaspeech_full_dd025653/`
  （meta.json、files.csv、statistics.json、group_statistics.csv、length_candidates.csv、
  issues.csv、report.md、figures/）
- 关键统计摘要（详见 report.md）：
  - 全库统一 16 kHz / 单声道 / WAV(PCM_16)，无扩展名与容器不符；
  - 时长 P50=2.06 s、P60=2.24 s、P80=2.93 s、P90=3.88 s、P95=5.09 s、max≈56.9 s，
    合计约 100 小时；所有词项类别中位时长 1.7–2.3 s，个别组合出现 20–57 s 离群；
  - 满配每人 5,355 = 255 词项 × 3 块 × 7 mic；Disorder 13 人全满配；
    Healthy 7 人缺录音（M01=2,805 最低）；Healthy 中 M2 mic 明显缺录（7,610 vs 其他 ~11k）；
  - peak p50=1.0：语料似乎整体做过归一化/放大，需在 D2 前端设计时注意；
  - length candidates：L=1 保留 39.8%（补零 0%）；L=2 保留 73.4%（补零 7.8%）；
    L=4 保留 91.4%（补零 42.6%）；L=6 保留 95.8%（补零 59.9%）；L=10 保留 98.4%（补零 75.3%）。
- 已知问题（issues.csv）：label_review 缺失（info）、前缀-目录完全分离待核查（warning）。
- 验收步骤建议：
  1. 复跑上述两条命令核对数字；
  2. `sddbench stats --dataset uaspeech --mode full ... --scan-id <scan_id>` 演示续扫（约 15 s）；
  3. 核对 report.md 第 8 节人工决策点；
  4. 未发生对原始数据的写入（数据根只读）。
- 结论：待人工验收
- 代码提交号：已推送（commit `0a2a843`，见 `git log`）

## D2（观察窗口索引：通道/裁剪/重采样/补零）——待验收

- 日期：2026-11-09
- 施工内容：`data/audio.py`、`data/windows.py`、`data/prepare.py`、
  `reporting/job_report.py`、`extract-features` CLI、两份预处理配置、
  `tests/test_crop_padding.py`。
- 实际执行：
  - 12 条 native/head/2 s 预检：
    `sddbench extract-features --dataset uaspeech --source 20261109T171421+0800_uaspeech_full_dd025653
    --preprocess configs/preprocessing/uaspeech_native_smoke.yaml --limit 12 --previews 6 --dry-run
    --paths configs/paths.local.yaml`
    → 12 条窗口（Disorder 6 + Healthy 6，12 个不同说话人），状态 `planned`；
  - 4 条 16 kHz 候选：
    `--preprocess configs/preprocessing/speech_common16k_candidate.yaml --limit 4 --previews 2`
    → 源均 16 kHz，`resample_applied=false`，4 s 目标 64,000 点，补零数已记录；
  - `python -m pytest tests -q` → 32 passed。
- 证据路径：`artifacts/feature_jobs/<job_id>/`
  （job.json、config.yaml、windows.csv、issues.csv、inspection.md、previews/）。
  两个实跑 job：`20261109T181848+0800_uaspeech_views_head_2s_ecc135d0`、
  `20261109T181900+0800_uaspeech_views_head_4s_da65659f`。
- 通过项（自检）：
  - 20 s→10 s 的 head/center/tail 起点为 0/5/10 s；3 s→10 s 有效 3 s、补零 7 s（单元测试）；
  - `fixed_random` 由 SHA256(版本|seed|sample_id) 决定，重复构建 `view_hash` 一致；
  - 原生处理前后源文件 SHA256 不变；`files.csv` 哈希不符时拒绝该条；
  - native 混合采样率会以 `blocked_mixed_sample_rate` 停止（退出码 3）。
- 已知限制：`--features` 尚未实现（D3）；2 s 仅为技术验证长度，不是主协议。
- 验收步骤建议：
  1. 打开 `inspection.md`，核对窗口起点/有效长度/补零比例与 `windows.csv`；
  2. 试听 `previews/*.wav`，确认是原始语音片段且未做音量归一化；
  3. 用 `--preprocess speech_common16k_candidate.yaml` 复跑，核对 4 s/16 kHz；
  4. 确认原始数据目录未被写入。
- 结论：待人工验收
- 代码提交号：见 `git log --grep "D2"`

## D3（MFCC 统计量与分片缓存）——待验收

- 日期：2026-11-09
- 施工内容：`features/mfcc.py`、`features/cache.py`、`configs/features/mfcc20_stats.yaml`、
  `extract-features` 接入 `--features`/`--cache-mode`、`tests/test_feature_cache.py`。
- 实际执行：
  - `sddbench extract-features --dataset uaspeech --source 20261109T171421+0800_uaspeech_full_dd025653
    --preprocess configs/preprocessing/uaspeech_native_smoke.yaml
    --features configs/features/mfcc20_stats.yaml --limit 12 --cache-mode shared --previews 6
    --paths configs/paths.local.yaml`
    → `windows built=12/12`，`features new=12 cached=0 failed=0`，状态 `completed`；
  - 同命令复跑 → `new=0 cached=12`；
  - `python -m pytest tests -q` → 40 passed。
- 证据路径：`artifacts/feature_jobs/20261109T182847+0800_uaspeech_views_head_2s_ecc135d0/`
  （job.json、feature_config.yaml、windows.csv、inspection.md 第 6 节）；
  共享缓存 `artifacts/cache/features/mfcc_stats/mfcc_stats_efe488280a7dc9a9/`
  （meta.json、index.csv 12 行、shard_000.npz 中 x=(12,40)）。
- 通过项（自检）：
  - x 形状 `[N,40]` 且全部有限；`n_mfcc=40` 时 80 维且 `feature_key` 改变；
  - 3 s 有效声音补零到 4 s 与直接 4 s 截取的统计一致；补零帧不进入统计；
  - `n_frames = 1 + (N - n_fft) // hop`，16 kHz 下 win/hop/n_fft = 400/160/512；
  - 相同源/视图/配方命中缓存；改配方或位置产生新 key；删除缓存可重算；
  - 混合 native 采样率在作业前阻断（退出码 3）。
- 已知限制：仍为 `technical_only`，`admission=pending_review`；未做全量提取，未产生训练标签。
- 验收步骤建议：
  1. 打开 `inspection.md` 第 6 节核对配方、维度、缓存路径与命中数；
  2. 检查 `index.csv` 的 12 个 `window_id` 与 `windows.csv` 一一对应；
  3. 复跑命令确认 `n_from_cache=12`、`n_new=0`；
  4. 确认测量前后原始音频 SHA256 不变。
- 结论：待人工验收
- 代码提交号：见 `git log --grep "D3"`

## D4（标签复核、批准清单与第二数据集）——待验收

- 日期：2026-11-09
- 施工内容：`data/admission.py`、`adapters/parkinson_kcl.py`、`reporting/admission_report.py`、
  `prepare` 命令、`configs/datasets/parkinson_kcl_v1.yaml`、
  `configs/preprocessing/parkinson_kcl_native_smoke.yaml`、`tests/test_admission.py`、
  `docs/datasets/parkinson_kcl.md`。
- 实际执行：
  - 第二库统计：`sddbench stats --dataset parkinson_kcl --mode full --workers 4 ...`
    → `decoded 73/73, failed 0, parse_failed 0`（15.0 s）；
    scan `20261109T183508+0800_parkinson_kcl_full_a83761e1`；
  - 第二库特征：`extract-features ... --limit 6 --cache-mode shared --previews 3`
    → `windows built=6/6`，`features new=6 cached=0 failed=0`，`completed`；
    job `20261109T183546+0800_parkinson_kcl_views_head_4s_539b0061`；
  - UAspeech 准入：`prepare ... --write-review-template` → 退出码 3，
    `blocked_label_review_incomplete`，未写 `manifest.csv`；
  - KCL 准入：同样退出码 3，`blocked_label_review_incomplete`，未写 `manifest.csv`；
  - `python -m pytest tests -q` → 47 passed。
- 证据路径：
  - `artifacts/inventory/parkinson_kcl/20261109T183508+0800_parkinson_kcl_full_a83761e1/`
    （`report.md`、`statistics.json`）；
  - `artifacts/label_review/uaspeech_label_review.csv`（28 行 pending）、
    `artifacts/label_review/parkinson_kcl_label_review.csv`（37 行 pending）；
  - `artifacts/prepared/uaspeech_v1/20261109T183604+0800_uaspeech_prepare/`
    与 `artifacts/prepared/parkinson_kcl_v1/20261109T183608+0800_parkinson_kcl_prepare/`
    （`admission.md/json`、`label_review_status.csv`、`issues.csv`、`provenance.json`）。
- 通过项（自检）：
  - 无复核文件 → `blocked_label_review_missing`；模板 pending → `blocked_label_review_incomplete`；
  - 全部 confirmed → 写出 `manifest.csv`，`group_id=<dataset>:<subject>`，origin 唯一，
    `label_source` 指向复核文件行号（合成测试覆盖）；
  - 同一 subject 两个 label → `blocked_subject_label_conflict`；
  - UAspeech 未选 mic → `blocked_mic_not_selected`；
  - KCL 适配器解析 `ID15_hc...`、`ID22hc...`、`ID02_pd..._1`；44.1 kHz 复用同一 MFCC 代码；
  - 原始音频未被修改/移动。
- 已知限制：
  - 两个库的真实 `label_review.csv` 均为 `pending`，因此没有真实 `manifest.csv`；
  - UAspeech 尚未选 mic；KCL `ID22` 两条文件名归一化冲突待人工确认；
  - D4 只做准入，不做 `split`。
- 验收步骤建议：
  1. 打开两份 `admission.md`，核对阻断项、说话人覆盖与选择规则；
  2. 检查 `label_review/*.csv` 列与行数（28 / 37），确认全部为 pending；
  3. 在测试环境把复核行改为 confirmed 并运行 `prepare`，确认能写出 `manifest.csv`；
  4. 打开 KCL `report.md` 对照 73 条、44.1 kHz、37 人、时长分位数；
  5. 确认 `artifacts/prepared/*` 下没有 `manifest.csv`，且源数据目录未变化。
- 结论：待人工验收
- 代码提交号：见 `git log --grep "D4"`
