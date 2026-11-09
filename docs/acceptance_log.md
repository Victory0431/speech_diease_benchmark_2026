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
- 代码提交号：未提交（验收后执行 `feat: add audited dataset statistics pipeline`）
