# Parkinson KCL 2017 数据集档案（第二库）

日期：2026-11-09｜状态：**已接入（技术验证）**｜标签：**pending（未确认）**

> 本文件只记录本地实际观测与候选解释，不把目录名/文件名词元当作已确认标签。
> 原始数据位置：`fresh_datasets/Parkinson_KCL_2017`（只读）。

## 1. 本地实测事实

| 维度 | 事实 |
| --- | --- |
| 文件 | 73 条 WAV，全部在 `Healthy/` 或 `Parkinson/` 一级目录下 |
| 分组 | Healthy 42 条 / 21 人；Parkinson 31 条 / 16 人；共 37 位说话人 |
| 格式 | 100% WAV 容器、PCM_24 编码、44,100 Hz、单声道 |
| 解码 | 73/73 成功；命名解析 0 失败；扩展名与容器 0 处不符 |
| 时长 | 总 10,318.5 s（≈2.87 h）；min 73.0 s，P50 141.3 s，P90 178.0 s，max 220.6 s |
| 结构 | 每位说话人通常 2 条（ID18 为 1 条）；多数成对文件共享会话前缀 |
| 统计来源 | full 扫描 `20261109T183508+0800_parkinson_kcl_full_a83761e1` |

## 2. 命名与候选解释

```text
ID<nn>[_](hc|pd)_<session tokens>[_<take>].wav
```

| 例 | 解析 |
| --- | --- |
| `ID15_hc_0_0_0.wav` | speaker=`ID15`，kind=`hc`，rest=`0_0_0` |
| `ID02_pd_2_0_0_1.wav` | speaker=`ID02`，kind=`pd`，rest=`2_0_0_1` |
| `ID22hc_0_0_0.wav` | 缺分隔符；仍解析出 speaker=`ID22`，并记 `naming_missing_separator` 告警 |

- `hc` / `pd` 是文件名 token，目录也分别为 Healthy/Parkinson；两者一致只能算
  **候选证据**，仍需在 `label_review.csv` 中确认。
- `session tokens` 的含义未核验（`0_0_0`、`2_0_0`、`3_1_1` 等）。
- 尾部 `_1` 可能是重复录音/第二遍；适配器记 `repeat_take_review`，不自动合并或删除。

## 3. 已知问题

1. **ID22 命名冲突**：`Healthy/ID22_hc_0_0_0.wav` 与 `Healthy/ID22hc_0_0_0.wav`
   归一化后同为 `ID22|hc|0_0_0`，但时长（167.44 s vs 150.24 s）和 SHA256 不同。
   `prepare` 因此报 `multiple_rows_per_origin` 并阻断；需人工确认二者关系。
2. **没有麦克风维度**：单通道、无 mic token；`selection.require_mic=false`。
3. **没有官方划分**：`official_split` 留空；后续必须按 `group_id`（subject）自行隔离。
4. **标签未确认**：`artifacts/label_review/parkinson_kcl_label_review.csv` 全为 `pending`。

## 4. 已跑通的命令

```bash
sddbench stats --dataset parkinson_kcl --mode full \
    --workers 4 --paths configs/paths.local.yaml

sddbench extract-features --dataset parkinson_kcl \
    --source 20261109T183508+0800_parkinson_kcl_full_a83761e1 \
    --preprocess configs/preprocessing/parkinson_kcl_native_smoke.yaml \
    --features configs/features/mfcc20_stats.yaml \
    --limit 6 --cache-mode shared --previews 3 \
    --paths configs/paths.local.yaml

sddbench prepare --dataset parkinson_kcl \
    --source 20261109T183508+0800_parkinson_kcl_full_a83761e1 \
    --write-review-template --paths configs/paths.local.yaml
```

- 特征作业：`artifacts/feature_jobs/20261109T183546+0800_parkinson_kcl_views_head_4s_539b0061/`
  - 6 条、native 44.1 kHz、head 4 s；`win_length=1103`、`hop_length=441`、`n_fft=2048`；
  - MFCC 40 维、`new=6 cached=0 failed=0`；缓存 key `mfcc_stats_7c1bdbbeb6594e5c`。
- 准入作业：`artifacts/prepared/parkinson_kcl_v1/20261109T183608+0800_parkinson_kcl_prepare/`
  - `status=blocked_label_review_incomplete`；37 人未确认 + ID22 origin 冲突；未写 `manifest.csv`。

## 5. 与 UAspeech 的复用关系

同一个 `scan` / `decoders` / `audit` / `stats` / `audio` / `windows` / `features.mfcc` /
`features.cache` / `admission` 代码，只新增：

- `src/sddbench/data/adapters/parkinson_kcl.py`
- `configs/datasets/parkinson_kcl_v1.yaml`
- `configs/preprocessing/parkinson_kcl_native_smoke.yaml`

没有为 44.1 kHz 或 PCM_24 写专用解码/重采样/MFCC 分支。
