# 审计修复记录（F01–F12）

依据：《仓库独立审计与基准实验推进建议_2026-11-15》，审计快照
`d0ca5f98ac65bfdb6d8f94ebf1bd8bc53292066f`。
修复分支：`main`（从 `aac66ed` 起，每批一提交）。
状态取值：`pending` / `in_progress` / `fixed` / `already_fixed` / `not_reproduced` /
`blocked_external_evidence`。

## 总览

| 编号 | 优先级 | 问题 | 状态 | 修复批次 | 证据 |
| --- | --- | --- | --- | --- | --- |
| F01 | P0 | UAspeech 标签冲突被“目录确认”替代回源 | fixed | A | 数据集配置/数据卡/准入/榜单统一 `technical_only/label_conflict` |
| F02 | P0 | 无人员信息默认按文件分组并自动 approved | fixed | A | `identity` 与 `label` 证据分离；`benchmark_eligible` 默认 false |
| F03 | P0 | AP/ROC 对并列分数处理错误 | in_progress | B | 待补 |
| F04 | P0 | task/manifest/原件/缓存身份未闭环 | in_progress | C | 待补 |
| F05 | P0 | aggregate 签名遗漏前端/准备版本/类别 | in_progress | D | 待补 |
| F06 | P1 | 阈值不生效；配置校验不统一 | in_progress | B | 待补 |
| F07 | P1 | 训练用源 Hz 找缓存，重采样接不通 | in_progress | C | 待补 |
| F08 | P1 | 网页改特征只写 YAML，C0 开关不可用 | in_progress | E | 待补 |
| F09 | P0 | run 太晚建立、缺快照、`last.pt` 语义不真 | in_progress | D | 待补 |
| F10 | P1 | `ratio_longer_than_L` 漏除样本数 | in_progress | B | 待补 |
| F11 | P1 | 文档称 ffmpeg 解码后备，实际只有 probe | in_progress | E | 待补 |
| F12 | P1 | 人员权重在全部集合上归一 | in_progress | B | 待补 |

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

待补。

## 批次 C（F04、F07）

待补。

## 批次 D（F05、F09）

待补。

## 批次 E（F08、F11）

待补。

## 历史结果处理清单

| 结果 | 处理 | 说明 |
| --- | --- | --- |
| UAspeech D6/D9 指标、图、run 包 | 保留，标 `technical_only` | 标签冲突未解决，不进入正式名次 |
| KCL 技术接入 | 保留 | 标签未复核，真实运行继续阻断 |
| 旧 `artifacts/cache/features/mfcc_stats/` | 保留，legacy | 新代码不读；重建命令见批次 C |
