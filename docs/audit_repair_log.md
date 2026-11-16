# 审计修复记录（F01–F12）

依据：《仓库独立审计与基准实验推进建议_2026-11-15》，审计快照
`d0ca5f98ac65bfdb6d8f94ebf1bd8bc53292066f`。
修复分支：`main`（从 `aac66ed` 起，每批一提交）。
修复提交：A `e217d3c`；B `2ba17f8`；C `0028bb7`；D `721ad62`；E `5547691`
（均已推送到 `origin/main`；审计快照 `d0ca5f9` 的历史结果未改动）。
状态取值：`pending` / `in_progress` / `fixed` / `already_fixed` / `not_reproduced` /
`blocked_external_evidence`。

## 总览

| 编号 | 优先级 | 问题 | 状态 | 修复批次 | 证据 |
| --- | --- | --- | --- | --- | --- |
| F01 | P0 | UAspeech 标签冲突被“目录确认”替代回源 | fixed | A | 数据集配置/数据卡/准入/榜单统一 `technical_only/label_conflict` |
| F02 | P0 | 无人员信息默认按文件分组并自动 approved | fixed | A | `identity` 与 `label` 证据分离；`benchmark_eligible` 默认 false |
| F03 | P0 | AP/ROC 对并列分数处理错误 | fixed | B | 按唯一阈值分组；AP/ROC 与图同源；行序不变 |
| F04 | P0 | task/manifest/原件/缓存身份未闭环 | fixed | C | task 快照+内容指纹；manifest/split 重算校验；缓存绑定 dataset/source |
| F05 | P0 | aggregate 签名遗漏前端/准备版本/类别 | fixed | D | 聚合键覆盖 task/prepared/view/feature/labels/threshold；同 seed 记 attempt |
| F06 | P1 | 阈值不生效；配置校验不统一 | fixed | B | `binary_threshold` 全链路生效；严格校验 |
| F07 | P1 | 训练用源 Hz 找缓存，重采样接不通 | fixed | C | 缓存/加载/训练统一用 processed Hz，覆盖 16k→8k、44.1k→16k |
| F08 | P1 | 网页改特征只写 YAML，C0 开关不可用 | fixed | E | 网页/CLI 共用 `plan`；保存即显示缓存缺失；`/api/features` 显式建缓存；C0 开关移除 |
| F09 | P0 | run 太晚建立、缺快照、`last.pt` 语义不真 | fixed | D | 开始即建 run；失败/中断留痕；true last 权重；产物+覆盖校验后才 completed |
| F10 | P1 | `ratio_longer_than_L` 漏除样本数 | fixed | B | 比例已除以 n，并限制在 [0,1] |
| F11 | P1 | 文档称 ffmpeg 解码后备，实际只有 probe | fixed | E | `decode` 真跑 ffmpeg 后备并记 backend；probe≠decode；统计闭合 |
| F12 | P1 | 人员权重在全部集合上归一 | fixed | B | 权重只在 train 内计算与归一 |

## 批次 A（F01、F02）

### F01 UAspeech 标签冲突

- **复现**：`configs/datasets/uaspeech_v1.yaml` 与实际 `docs/datasets/uaspeech.md`
  写 pending/疑点，但 `artifacts/label_review/uaspeech_label_review.csv` 已把
  F02 等标为 confirmed，证据仅为“用户确认目录含义”；旧 `prepare` 会直接
  `status=approved`，`publish` 会把它排进榜单。
- **判定**：审计成立。“目录含义确认”不是回源证据，不能关闭已知冲突。
  本地仍缺官方说话人表与本地 `speaker_token` 的对应、下载/版本记录，
  无法证明当前目录没有倒置。
- **修复**：
  - 新增 `scientific_status`（`benchmark_eligible` / `technical_only` / `blocked`）
    与 `identity`（`subject_ids` / `one_recording_per_person` / `unknown`）；
  - UAspeech 固定为 `technical_only / label_conflict`，并列出缺失证据；
  - `prepare` 输出 `benchmark_eligible=false`、`scientific_status`、
    `label_evidence`、`identity_evidence`；
  - `run.json` 继承该状态；`publish` 只给 `benchmark_eligible=true` 的 run 排名；
  - `web/catalog` 暴露同一状态；数据卡、README、配置三处措辞统一。
- **测试**：`tests/test_scientific_status.py` 覆盖“confirmed 目录确认仍不 eligible”、
  非法 level 被拒绝、榜单排除非 eligible。
- **历史结果影响**：旧 run 没有 `benchmark_eligible` 字段，按 `false` 处理，
  保留数字但不再进入正式名次。若日后取得官方映射，应生成新的
  task/prepared/split 版本并重跑，不能只改旧结果中的疾病名称。

### F02 目录标签与人员独立性

- **复现**：`label_folders` 适配器把目录名变成标签、`review_status=pending`
  后由 `prepare` 自动 confirmed；无说话人时 onboarding 建议 `grouping=file`，
  旧逻辑直接 `approved`，没有任何“人员独立性”声明要求。
- **修复**：
  - `label_evidence` 与 `identity_evidence` 分开输出，后者不从目录名推断；
  - `grouping=file` 只有声明 `identity.evidence: one_recording_per_person`
    且给出 `reference` 才可能 eligible，否则 `identity_unverified`；
  - `scientific_status` 缺省为 `technical_only`，新增数据集不会静默进正式榜；
  - 新增精确内容重复检查：同一 `sha256` 跨组或跨类 → `blocked_duplicate_content`，
    不写 `manifest.csv`；同组内重复仅 warning；
  - 原有 `origin_id` 冲突检查保留，已知同源录音不能跨组/跨类。
- **测试**：`tests/test_scientific_status.py` 覆盖默认不 eligible、缺出处不 eligible、
  补齐出处后 eligible、跨标签重复字节阻断、同组重复仅 warning；
  既有 `test_label_folders`/`test_admission` 全绿（其合成 fixture 改为不同音调，
  因为原 fixture 的四个文件字节完全相同，新检查正确识别为跨类重复）。
- **历史结果影响**：历史 `label_folders` 技术实验保留；它们不再被自动当作
  无泄漏基准结果。

## 批次 B（F03、F06、F10、F12）

### F03 AP / ROC / 逐类指标

- **复现**：`average_precision([1,0],[0.5,0.5])=1.0`，行序交换后 0.5；
  原 `_roc_points` 逐行连线，图面积与 rank AUROC 对 ties 不一致。
- **修复**：
  - 新增 `_curve`：按**唯一分数**分组，ties 只产生一个工作点；
    AP = `Σ(Δrecall × precision)`，AUROC = 该 ROC 点列的梯形面积；
  - 报告层改用同一个 `roc_points()`，数值与图不再分叉；
  - `per_class_rows` 按 `positive_index` 取正/负统计，support 与 predicted
    从混淆矩阵的真实正类定义推导，不再固定 `class_names[1]`；
  - F1 零分母统一为 0.0（同 sklearn `zero_division=0`）。
- **测试**：`tests/test_metrics_contract.py` 覆盖 ties 行序不变、图面积=AUROC、
  全同分/完美/全错、缺类、F1 零分母、`positive_index=0/1` 的 support 与名称；
  原 `test_modeling` 的已知 AP 期望保留为 `7/12`（该值本身与阈值定义一致）。
- **历史结果影响**：旧 AP/ROC 图只影响并列分数或单类边界样本；
  如需引用旧成绩，应从已保存的 `predictions.csv` 重算指标，不必重训。

### F06 阈值与配置校验

- **复现**：`evaluation.binary_threshold` 从 0.01 改到 0.99，记录仍是 0.5、
  混淆矩阵不变；`selection_direction` 可写但总是最大化；未知字段被静默保存。
- **修复**：
  - `DEFAULTS` 增加 `evaluation.binary_threshold=0.5`；
  - 新增 `decide_binary(p, threshold)`：预测标签、指标、混淆矩阵、ROC 图
    全部使用 `p_positive >= threshold`，与 `argmax` 的 tie 行为统一；
    `metrics.json` 记录实际 threshold；
  - `validate_experiment()` 严格校验：未知顶层/分节键、类型与范围、
    `selection_direction` 仅允许 `max`（`min` 未实现，直接拒绝）、
    `primary_metric` 仅 `balanced_accuracy`、类别必须恰好 2 个且正类在其中；
  - `load_experiment`、CLI `--set` 之后、`sweep` 每个组合、
    `web.build_configs` 都调用同一个校验；
  - `load_run_data` 在训练前拒绝非二分类 manifest。
- **测试**：阈值改变混淆矩阵、等于阈值判正、未知键/非法值/未实现选项被拒绝、
  override 后重验。
- **历史结果影响**：旧 run 的 `binary_threshold` 实际按 0.5 计算；
  若论文使用过其他阈值，必须用保存的预测重算指标，不能沿用旧混淆矩阵。

### F10 长度比例

- **复现**：`ratio_longer_than_L` 返回计数（可为 73），不是 0–1 比例。
- **修复**：`count/n`，空输入返回 0.0；与 `ratio_shorter_than_L` 口径一致。
- **测试**：4 条录音、L=4 时 longer=2、比例 0.5，且落在 [0,1]。

### F12 人员权重

- **复现**：`_subject_weights(groups)` 对 train+val+test 全体计数并归一，
  改变验证/测试构成会改变训练损失尺度。
- **修复**：新增 `_sample_weights`，只在 `train_mask` 上调用
  `_subject_weights`，val/test 权重保持 1 且不参与训练。
- **测试**：修改 val/test 组分布后，train 权重逐元素不变；每个训练组总权重相同。
- **历史结果影响**：历史 run 的训练损失尺度可能受验证/测试构成影响；
  正式复现应使用新代码重训，不能把旧权重当作同一训练目标。

## 批次 C（F04、F07）

### F04 task / manifest / split / 缓存身份闭环

- **复现**（审计探针，修复前均成立）：
  - 修改 `manifest.csv` 中某样本的 `sha256`（同 `sample_id`），旧代码只比较
    admission 里保存的两个字符串，训练照常加载旧特征；
  - 修改仓库 `configs/datasets/*.yaml` 的 `grouping` 或标签映射，已 prepare 的
    旧目录仍被接受（训练时现读仓库配置）；
  - 修改 `split.csv` 的某行 `split`，训练照常使用；
  - 两个数据集存在相同 `relative_path` 时，特征加载仅按 `sample_id` 匹配。
- **修复**：
  - 新增 `src/sddbench/data/identity.py`：
    - `task_fingerprint(task_cfg)`：对解析后的 task YAML 全文（标签、类别顺序、
      正类、grouping、layout、筛选、准入规则）做内容哈希；
    - `manifest_fingerprint(manifest, task_hash)`：覆盖
      sample/relative_path/subject/group/origin/label_raw/label/variant/elicitation/
      sample_rate/duration/channels/sha256/official_split/label_source + task_hash，
      按 `sample_id` 排序；
    - `split_fingerprint(rows)`：覆盖 sample→(group,label,split) 全表。
  - `prepare` 同时写入 `task.yaml`、`task_hash`；`admission.json` 记录
    `manifest_fingerprint`（含 task_hash）；
  - `split` 从当前 `manifest.csv` **重新计算**指纹并与 admission 比对，写入
    `manifest_fingerprint/task_hash/split_fingerprint`；
  - `trainer.load_run_data` 只读 prepared 内的 `task.yaml`（不再读可变仓库配置），
    重算并核对 task/manifest/split 三个指纹；缺失快照或指纹的旧 prepared
    明确拒绝并提示重新 prepare；
  - 特征缓存 schema 升到 2.0：每行绑定 `dataset_id/source_sha256/
    processed_sample_rate`；`window_id` 改为 `dataset|sample|w0|view_hash`，
    不再包含 scan_id（同内容重扫复用同一行，不产生假多窗），`view_hash` 纳入
    `dataset_id`；
  - `load_features` 先按 dataset 作用域过滤，再检查 sample 唯一性、source hash、
    processed Hz、feature/view hash；库间同名相对路径不会串库；
  - 旧 schema 1.0 缓存（含 index.csv 无身份列）明确报错并给出重建命令，
    **不删除**、不静默沿用。
- **测试**：`tests/test_identity_chain.py` 覆盖：
  - 同 `sample_id` 改 `sha256` → 训练拒绝；
  - 改 `task.yaml` 的 `grouping` → 拒绝；删除快照 → 拒绝；
  - 改 `split.csv` 分配 → 拒绝；
  - manifest 指纹随 group/label/origin/sha/variant/subject 变化，split 指纹随分配变化；
  - 同名相对路径跨库不串、各自取自己的向量；
  - 缓存 source hash 与 manifest 不一致 → 拒绝；
  - 16k→8k、44.1k→16k：窗口 `original_sample_rate` 与
    `processed_sample_rate` 同时正确，缓存按处理后 Hz 存储并可加载；
  - 同内容不同 scan_id → 同一 `window_id`/`view_hash`；
  - schema 1.0 缓存被拒绝且文件仍在。
- **历史结果影响**：
  - 旧 prepared 目录（无 task_hash/task.yaml 或无 split_fingerprint）不能再用于
    新训练，需重新 `prepare`/`split`；旧 run 结果与预测保留不变。
  - 旧特征缓存（schema 1.0）不可复用，需按下方命令重建；不自动删除。

### F07 原始 Hz / 处理 Hz 分离

- **复现**：`load_run_data` 用 `manifest.sample_rate`（源 Hz，如 16k）去找缓存，
  而 `extract-features` 按前端处理后 Hz（如 8k）落盘；`sample_rate_policy:
  resample` 在训练端永远找不到缓存。
- **修复**：
  - 由 `preprocess.sample_rate_policy` 解析实际处理 Hz：`resample` →
    `target_sample_rate`；`native` → 要求 manifest 源 Hz 唯一；
  - `native` 且源 Hz 混合时在训练前明确拒绝（MLP 基线只支持单一处理 Hz）；
  - 缓存目录、`meta.json`、`index.csv`、`load_features` 与 `run.json`
    统一使用 processed Hz；
  - `run.json` 同时记录 `sample_rate`（处理后）与 `original_sample_rates`（源），
    不再用源 Hz 冒充。
- **测试**：`tests/test_identity_chain.py`
  `test_resampled_features_are_stored_and_loaded_at_processed_hz`（16k→8k、
  44.1k→16k）、`test_native_mixed_rates_are_rejected_before_training`。
- **历史结果影响**：旧缓存按源 Hz 建目录，属于不同协议，保留为 legacy；
  旧 run 的 `sample_rate` 字段语义为源 Hz，解读时以 run 包中的
  `feature_cache`/`original_sample_rates` 为准。

### 旧缓存重建命令（不删除旧目录）

```bash
sddbench extract-features \
  --source <scan_id> --dataset <dataset_id> \
  --preprocess configs/preprocessing/<frontend>.yaml \
  --features configs/features/<recipe>.yaml
```

新缓存目录名包含处理后 Hz 与 view 指纹；旧目录可原样保留作审计对照。

## 批次 D（F05、F09）

### F09 run 生命周期与快照

- **复现**：旧 `run_experiment` 先训练、结束后才创建 run 目录并直接写
  `status: completed`；训练中途异常无任何记录；`last.pt` 实际保存的是 best
  权重；`run.json` 只记配置文件路径；`predictions_windows.csv` 的
  `window_id` 直接等于 `sample_id`，`valid_seconds` 用原始总时长。
- **修复**：
  - 进入 `run_experiment` 先创建唯一 run 目录并写 `run.json`
    （`status=running`, `phase=created`）与 `config.yaml`；随后才加载数据；
  - `status` 状态机：`running → completed | failed | interrupted`；
    `KeyboardInterrupt/SystemExit` 记 `interrupted`，其它异常记 `failed`，
    均保存 `error`、`finished_at` 与最初配置；
  - 只有 `_verify_run_artifacts` 检查全部必需产物非空、且每个 split 的
    window/group 预测数与期望一致后，才写 `status=completed`；
    否则以 `completion_errors` 记录并抛错，run 保持 failed；
  - run 包保存展开后的 `task/preprocess/feature/model/training/evaluation`
    实际解析内容、标签定义、输入清单（manifest/split/task 快照/
    preprocess/features/feature index）、命令、Python/torch/numpy 版本、
    平台、cwd、torch 线程数；
  - git 记录 `commit/dirty/dirty_files`；dirty 时额外执行
    `git stash create` 保存可恢复的 dangling commit（不改工作区）；
  - `config_hash` 纳入标签定义、task_hash、view fingerprint、feature_hash、
    processed Hz、阈值与评价单位；
  - `best.pt`：`role=best_val_selection`，记录选择指标/方向/val 分数/epoch；
    `last.pt`：`role=last_epoch`，保存**最后实际训练 epoch** 的权重，
    并显式 `resume_supported: false`（不伪造可续训状态）；
  - `window_id` 取自特征缓存真实窗口行；`valid_seconds = valid_frames /
    processed_sample_rate`（不含 padding），不再用 sample_id/原始总时长。
- **测试**：`tests/test_modeling.py`
  `test_run_lifecycle_records_identity_window_facts_and_checkpoint_roles`、
  `test_failed_run_keeps_config_identity_and_failure_reason`、
  `test_same_config_rerun_is_recorded_as_attempt_not_new_seed`；
  `tests/test_results.py` 的 failed/attempt/sanitize 用例。
- **历史结果影响**：旧 run 仍保留原件；其 `run.json` 缺 `identity/attempt/
  checkpoints`，按 legacy 解读。需要新协议留痕或 last 权重语义的结果必须重训。

### F05 aggregate 比较键与 attempt

- **复现**：旧签名只含 model/training（去掉 seed）与 split；不同
  prepared/task/特征/类别顺序/阈值/评价单位的 run 会被平均到同一组；
  同 seed 重复执行被当作独立 seed 增加样本量。
- **修复**：
  - `_signature` 覆盖 task_id、prepared_id、feature_key/feature_hash/
    view_fingerprint/processed Hz/dim、class_order、positive_class、
    binary_threshold、evaluation_unit、selection_metric、model 与除 seed 外的
    training；split_sensitivity 只放宽 split_id；
  - `load_runs` 只收 `status in (completed, 旧无状态)`，failed/interrupted 不进入
    聚合与发布；
  - 同 seed 的多次 run 按 attempt 归并：先在 seed 内取均值，再跨 seed 统计；
    `aggregate.json` 记录 `n_seeds/n_runs/attempts_per_seed/seed_level`；
  - bootstrap 明确 `axis=training_seed, unit=seed_mean`；
  - `subject_bootstrap` 记录 `n_resamples/seed/unit/resample/n_groups/
    n_effective_*/n_skipped_missing_class/missing_class_policy`；
  - `publish_run` 拒绝非 completed run。
- **测试**：`tests/test_results.py`
  `test_aggregate_merges_same_seed_attempts_without_inflating_seed_count`
  （3 次执行、2 个 seed、均值 0.85）、
  `test_aggregate_never_mixes_feature_identity_or_threshold`、
  `test_failed_run_is_not_aggregated_or_published`。
- **历史结果影响**：旧 aggregate 目录仍是旧签名产物，不覆盖；需要新比较键时
  重新执行 `sddbench aggregate`。旧 run 指标不需要重算，但若要与新 run
  合并统计，必须确认其特征/标签/阈值身份一致。

## 批次 E（F08、F11）

### F08 网页与 CLI 的 plan / 校验 / 特征构建闭环

- **复现**：
  - `web/schema.py` 暴露 `features.include_c0`（默认 true，帮助写“关闭后第一个系数
    置零”），而 `mfcc.normalize_feature_config` 强制 `include_c0 is True`，关闭必报错：
    网页给出了一个不可用的可操作选项；
  - `/api/run` 只写 YAML 就直接起训练子进程；改 MFCC/采样率/裁剪后唯一反馈是
    训练日志里“feature cache not found”，网页事先看不到缓存失效；
  - 网页与 CLI 各自检查配置，网页无法证明训练时解析的身份与 CLI 一致。
- **修复**：
  - 新增 `modeling/plan.py::describe_plan`（只读）：调用与训练相同的
    `resolve_run_inputs`（task 快照、manifest/split 内容指纹、标签、处理 Hz、
    view 指纹）和共享的 `inspect_cache`，返回
    `identities / feature{status,reason,cache_dir,feature_hash,dim,n_missing} /
    actions`；缓存缺失、legacy、incompatible、partial 都给出原因与下一步命令；
  - `modeling/features.py` 拆出 `_resolve_cache` + `inspect_cache`：
    训练加载与 plan 共用同一套身份/覆盖检查，网页不会宣称 loader 会拒绝的
    “ready”；
  - CLI 新增 `sddbench plan --config ... --paths ...`：输出同一份 JSON；
    ready 退出 0，需先构建退出 3，脚本可据此门禁；
  - `extract-features` 新增 `--manifest <prepared manifest.csv>`：只对已准入的
    sample_id 建缓存（网页触发的构建不会把整库扫一遍），清单缺失的样本明确报错；
  - 网页：
    - `/api/experiment` 保存后直接返回 plan；
    - 新增 `/api/plan`（只读，不写文件）；
    - 新增 `/api/features`：保存配置后以子进程运行
      `extract-features --cache-mode shared --manifest ...`，作为 job 轮询，
      完成后自动刷新 plan；
    - `/api/run` 在 plan 未 ready 时返回 **409 + plan**（试运行仍可执行）；
    - “模型与训练”页展示执行计划、缓存目录/维度/view 指纹/下一步命令；
    - 移除 `features.include_c0` 字段（torchaudio MFCC 固定包含 c0，不再承诺
      无法实现的行为）。
- **测试**：`tests/test_plan.py`
  - 缓存与配方一致 → `ready`、dim/Hz/样本数/指纹正确；
  - 改 `n_mfcc` → 同一旧缓存不再算 ready，报 `missing`，给出含
    `extract-features/--manifest/--cache-mode shared` 的命令；
  - CLI `plan` 退出码 0/3 与网页判定一致。
  `tests/test_web.py`：schema 不含 `include_c0`；保存返回 blocked plan；
  `/api/plan` 只读；`/api/run` 返回 409 且带 plan。
  `tests/test_feature_cache.py`：`--manifest` 只构建清单内样本；清单样本不在
  scan 中时返回非 0，而不是静默少建。
- **历史结果影响**：无。旧 run 不受影响；旧缓存若与新配方不符会在 plan 中显示为
  `missing/incompatible`，不会被训练误用。网页生成的配置仍落到
  `configs/experiments/generated/`，与 CLI 读取的是同一文件。

### F11 probe 与 decode 能力分离、ffmpeg 真实后备

- **复现**：`decoders.decode` 只调用 soundfile；`audit_one` 对
  `ok_ffprobe` 文件从不尝试解码，`decode_status=not_attempted`；文档却写着
  “ffprobe/ffmpeg 后备”。可达成的效果是“探针成功”，不是“可解码”。
  probe 失败的文件还被记成 `decode_status=failed`，使 ok/failed/not_attempted
  统计口径不闭合。
- **修复**：
  - `decode` 在 soundfile 失败后走 `_decode_ffmpeg`：先 `ffprobe` 取真实
    sample_rate/channels，再以 `ffmpeg -nostdin -v error -i <file>
    -map 0:a:0 -f f32le -acodec pcm_f32le pipe:1` 解码，**不重采样**，
    以 ffprobe 的通道数 reshape 成 `[C, N] float32`；
  - backend 记为 `ffmpeg(<version>)`；ffmpeg 缺失、超时、空输出、非整帧字节数
    都抛 `DecodeError`，不会把“探针成功”当成“解码成功”；
  - `tool_versions()` 同时记录 `ffmpeg`；外部编解码进程使用去掉
    `LD_LIBRARY_PATH` 的环境，避免父进程注入的私有 C++ 运行时导致
    `GLIBCXX_... not found`；
  - `audit_one`：quick 模式只 probe；full 模式对所有 probe 成功
    （`ok_soundfile` 或 `ok_ffprobe`）的文件真实调用 `decode`；probe 失败改记
    `decode_status=not_attempted` + `decode_error="probe failed; decode not attempted"`；
  - 统计新增 `decode_not_attempted/decode_unclassified`，
    每个目录组细分 ok/failed/not_attempted；数据集报告新增“未尝试”列并展示
    ffmpeg 版本，`ok+failed+not_attempted == audio_candidates` 可核对。
- **测试**：`tests/test_format_fallback.py`（全部为合成音频）
  - 合成 WAV → ffmpeg 编码 AAC/M4A：`probe=ok_ffprobe`；
  - `decode` 成功，backend 为 `ffmpeg(...)`，16 kHz 原生速率、float32、近似原时长；
  - `audit_one(full)` 对该文件 `decode_status=ok` 且记录 ffmpeg backend；
  - 损坏文件的 decode 明确抛 `DecodeError`；
  - 一条 WAV + 一条损坏文件（+ 可选 M4A）的统计满足三态闭合，probe 失败记为
    not_attempted。
  `tests/test_dataset_stats.py` 相应更新为“probe failed → not_attempted”。
- **历史结果影响**：旧 scan 里 ffprobe-only 文件的 `decode_status` 仍是
  `not_attempted`（当时确实没解码），不篡改历史 CSV；需要真实音频事实时应重新
  `audit`（full）。旧统计数字不含 `decode_not_attempted`，按旧口径解读；
  新统计不覆盖旧文件。

### 新增命令与最短验证

```bash
# 训练前查看同一份计划（0=可训练，3=需先构建）
sddbench plan --config configs/experiments/<exp>.yaml --paths configs/paths.local.yaml

# 只构建已准入清单的特征（网页“构建特征缓存”按钮执行的就是这条）
sddbench extract-features --dataset <ds> --source <scan_id> \
  --preprocess <preprocess.yaml> --features <features.yaml> \
  --manifest <workspace>/prepared/<task>/<prep>/manifest.csv \
  --cache-mode shared --paths configs/paths.local.yaml

# 格式能力验证（合成音频）
python -m pytest -q tests/test_format_fallback.py tests/test_plan.py
```

## 历史结果处理清单

| 结果 | 处理 | 说明 |
| --- | --- | --- |
| UAspeech D6/D9 指标、图、run 包 | 保留，标 `technical_only` | 标签冲突未解决，不进入正式名次 |
| KCL 技术接入 | 保留 | 标签未复核，真实运行继续阻断 |
| 旧 `artifacts/cache/features/mfcc_stats/` | 保留，legacy | schema 1.0，无 dataset/source/processed-Hz 身份；新代码拒绝读取，重建命令见批次 C |
| 旧 prepared/split（无 task_hash/split_fingerprint） | 保留，不可用于新训练 | 需重新 `prepare`/`split`；旧 run 包不变 |
| 旧 run（源 Hz 语义） | 保留 | `sample_rate` 为源 Hz；新代码用 `original_sample_rates` 与 processed Hz 区分 |
