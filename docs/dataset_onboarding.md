# 数据集接入指南（数据层契约）

这份文档是给"只拿到数据集路径和这份文档"的人或 agent 用的。按第 2 节判断类型，
照着对应章节做完即可，不需要读 `data_layer_plan.md`，也不需要改 Python。

---

## 1. 先记住五条规矩

| # | 规矩 | 落到代码 |
| --- | --- | --- |
| 1 | 数据集配置只说"这个库怎么解释"，不说音频视图和模型 | `configs/datasets/<id>.yaml` |
| 2 | 预处理层只管音频视图：采样率、通道、裁剪、补零、幅度 | `configs/preprocessing/*.yaml`；写了 `frame_ms/hop_ms` 会被**报错拒绝** |
| 3 | 帧长、帧移、`n_fft`、MFCC/mel 参数、统计量只属于特征层 | `configs/features/*.yaml`，`feature_type` 必填 |
| 4 | 数据集的特殊性（麦克风、命名 token）只能留在数据层，模型层只看到"向量 + 标签 + 分组" | `src/sddbench/data/` |
| 5 | 目录名就是标签时，不需要写 adapter、不需要复核 CSV | `layout: label_folders` |

**模型层永远只读三样东西**：特征矩阵（`cache/features/<feature_type>_<hash>/`）、
`manifest.csv` 的标签、`group_id`。任何数据集专属字段都不允许进入模型层。

---

## 2. 判断你的数据集属于哪一类

### A. 普通分类目录（绝大多数情况）

```
<数据集根>/
  Healthy/      ← 一个文件夹 = 一个类
    a.wav
    b.wav
  Disease/      ← 另一个文件夹 = 另一个类
    c.wav
```

用 **`layout: label_folders`**，见第 3 节。不需要写任何代码。

### B. 文件名/元数据里藏标签（少数情况）

例如 UAspeech 的 `CF02_B1_C1_M2.wav`，标签来自文件名 token；Parkinson KCL 的
`ID00_hc_0_0_0.wav`，标签来自 `hc/pd`。用 **`layout: rich`** + 专用 adapter，见第 4 节。

> 只有在"文件名/外部表包含标签或说话人/麦克风信息"时才需要 B。
> 只是想让 `Healthy` 改叫 `healthy`，用 A 的 `label_map`，不要写 adapter。

---

## 3. 接入 A：`label_folders`

### 3.1 绑定路径

编辑 `configs/paths.local.yaml`（该文件不入库）：

```yaml
workspace_root: /abs/path/to/artifacts
dataset_roots:
  my_dataset: /abs/path/to/my_dataset
```

### 3.2 写数据任务配置

新建 `configs/datasets/my_dataset_v1.yaml`：

```yaml
task_id: my_dataset_v1
dataset_id: my_dataset
task_version: 1
layout: label_folders
grouping: file            # 没有说话人信息时必须是 file
label_map:                # 可省略；省略时文件夹名小写即为标签
  Healthy: healthy
  Control: healthy
  Disorder: disorder
# 目录名只能提供标签，不能证明“不同文件是不同人”。identity 未声明时，
# 这个库只能做技术实验，不能进入宣称无人员泄漏的正式主榜。
scientific_status:
  level: technical_only      # benchmark_eligible | technical_only | blocked
  reason: identity_unverified
identity:
  evidence: unknown          # subject_ids | one_recording_per_person | unknown
  reference: ""              # one_recording_per_person 时必须给出处
```

字段说明：

- `layout`：`label_folders` 固定写法。
- `grouping`：`file` = 每条录音自成一组（无说话人）；有说话人时用 `subject`，
  但 `label_folders` 不支持，请改用 rich。
- `label_map`：
  - 省略 → 标签 = 文件夹名小写（`Healthy/` → `healthy`）；
  - 提供但漏了某个实际存在的文件夹 → **准入直接阻塞**，不会瞎猜。
- `scientific_status` / `identity`：见 3.5 节。`prepare` 的 `status=approved`
  只表示清单结构可用；是否进入正式基准由 `benchmark_eligible` 单独决定。

### 3.3 选择预处理与特征配方

直接复用，不用新建：

- 16 kHz 统一采样率：`configs/preprocessing/speech_common16k_candidate.yaml`
- 保持原始采样率：`configs/preprocessing/uaspeech_native_4s_head.yaml`
- MFCC 20 维 + mean/std：`configs/features/mfcc20_stats.yaml`

只有要改观察窗长度/位置时，才复制一份预处理 yaml 改 `observe_seconds` 和 `position`。
**不要**往预处理里写 `frame_ms/hop_ms`。

### 3.4 扫描

```bash
sddbench stats --dataset my_dataset --mode full \
  --paths configs/paths.local.yaml \
  --task configs/datasets/my_dataset_v1.yaml
```

产物：`artifacts/inventory/my_dataset/<scan_id>/`，记住打印出的 `scan_id`。
这一步会校验：音频能否解码、采样率分布、根目录下有没有直接放音频的文件。

### 3.5 准入（自动确认标签）

```bash
sddbench prepare --dataset my_dataset --source <scan_id> \
  --paths configs/paths.local.yaml \
  --task configs/datasets/my_dataset_v1.yaml
```

不需要 `--review`，也不会生成复核 CSV。成功时：

- `status=approved`，退出码 0；但 `benchmark_eligible` 默认是 **false**；
- `manifest.csv` 的 `label_source` 为 `directory:Healthy` 这种形式；
- 目录名不一致（未映射）或音频放在根目录 → 退出码 3，看 `admission.md`。

三种身份证据与可进入的轨道（F02）：

| 身份条件 | 允许的划分 | 能进入哪里 |
| --- | --- | --- |
| 人员 ID 可靠（rich 复核表 `confirmed_subject_id`） | `subject` | 正式主基准候选 |
| 有来源依据证明每人一条原始录音（`identity.evidence: one_recording_per_person` + `reference`） | `file` | 正式候选，需报告依据 |
| 身份未知/无法排除同人多录音（默认） | `file` | 只做技术实验，不进无泄漏主榜 |

要让 `benchmark_eligible` 变成 true，必须同时满足：`scientific_status.level:
benchmark_eligible`、身份证据成立、准入无 blocker，且没有跨组/跨类的精确内容重复。
仅把目录名写进 `label_map` **不会**自动提升证据等级。

记住打印出的 `prepared_id`（目录名，如 `20261115T..._my_dataset_prepare`）。

### 3.6 划分

```bash
sddbench split --dataset my_dataset --prepared <prepared_id> --seed 0 \
  --paths configs/paths.local.yaml \
  --task configs/datasets/my_dataset_v1.yaml
```

### 3.7 实验配置

复制 `configs/experiments/` 里的现成配置，改四个绑定字段：

```yaml
task_id: my_dataset_v1
dataset: my_dataset
prepared_id: <prepared_id>
split_id: <split_id>
features:
  config: configs/features/mfcc20_stats.yaml
  preprocess: configs/preprocessing/speech_common16k_candidate.yaml
evaluation:
  unit: recording          # grouping=file 时用 recording；有说话人时用 subject
  primary_metric: balanced_accuracy
```

如果标签词表不是 `control/disorder`，在实验配置里显式写：

```yaml
labels:
  class_order: [healthy, disorder]
  positive_class: disorder
```

不写时按字典序推断并在 `run.json` 记录来源，**不推荐**用于正式结果。

### 3.8 跑通

```bash
# 可选：只生成窗口索引，不训练
sddbench extract-features --dataset my_dataset --source <scan_id> \
  --preprocess configs/preprocessing/speech_common16k_candidate.yaml \
  --dry-run --paths configs/paths.local.yaml

# 正式训练（会顺带算特征；已有缓存则复用）
sddbench run --config configs/experiments/my_dataset_mlp.yaml \
  --paths configs/paths.local.yaml --dry-run   # 先验证
sddbench run --config configs/experiments/my_dataset_mlp.yaml \
  --paths configs/paths.local.yaml
```

`run.json` 的 `features` 段必须出现：

```json
"features": {
  "feature_type": "mfcc", "recipe_id": "mfcc_stats",
  "feature_key": "mfcc_<hash16>", "dim": 40,
  "preprocess_fingerprint": "<hash>"
}
```

---

## 4. 接入 B：`layout: rich`（有说话人/麦克风/命名结构）

只有这类数据集才需要写代码，代价是：

1. 新建 `src/sddbench/data/adapters/<name>.py`，实现：

```python
class MyAdapter(DatasetAdapter):
    name = "my_adapter"

    def collect(self, root, task_config, inventory) -> AdapterResult:
        # 只做事实解释：谁拥有哪个文件、候选标签在哪、哪些是副本
        # 不要解码音频、不要裁剪、不要算特征
        ...
```

2. 在 `src/sddbench/data/adapters/registry.py` 的 `ADAPTERS` 加一行。
3. 数据任务配置写 `adapter: my_adapter`（而不是 `layout`），并给出 `label_review.file`。
4. 准入前必须有复核 CSV；`prepare --write-review-template` 可生成模板。

rich 的字段（`speaker_token`、`directory_group`、`mic`、`origin_id`）只在这里出现，
`label_folders` 数据集一律不填、不感知。模型层对两者完全无差别。

现有 rich 示例：`configs/datasets/uaspeech_v1.yaml`、`configs/datasets/parkinson_kcl_v1.yaml`。

---

## 5. 每层的职责与禁止事项

| 层 | 文件 | 只允许 | 禁止 |
| --- | --- | --- | --- |
| 数据集 | `configs/datasets/*.yaml` | layout、标签来源、grouping、label_map | 帧长、模型参数、采样率 |
| 预处理 | `configs/preprocessing/*.yaml` | 采样率、通道、观察窗、位置、补零、幅度 | `frame_ms/hop_ms/n_fft`（会被拒绝） |
| 特征 | `configs/features/*.yaml` | `feature_type`、帧几何、n_mfcc/n_mels、统计量 | 数据集分支、标签逻辑 |
| 特征缓存 | `src/sddbench/features/cache.py` | 按 `feature_type` 取维度、写身份 | 写死某个特征类型的参数 |
| 准入 | `src/sddbench/data/admission.py` | 目录名/复核表 → manifest | 猜标签、跳过错配 |
| 模型 | `src/sddbench/modeling/` | 读缓存、manifest、split | 解码音频、判断数据集来源 |

**加新特征类型（如 log-mel）的唯一正确姿势**：
新增一个与 `features/mfcc.py` 同接口的模块（`normalize/dim/hash/implementation_info/extract`），
在 `features/registry.py` 的 `RECIPES` 加一行。缓存目录、索引列、`run.json`、
模型输入维度会自动跟着走，不需要改数据层或模型层。

---

## 6. 术语表（只保留这些）

| 术语 | 含义 | 出现位置 |
| --- | --- | --- |
| `dataset_id` | 数据集标识，对应 `dataset_roots` 的键 | 配置、命令行 |
| `task_id` | 数据集 + 任务版本，如 `my_dataset_v1` | 配置、prepared 目录 |
| `prepared_id` | 一次准入的产物目录名 | `artifacts/prepared/<task_id>/<prepared_id>/` |
| `split_id` | 一次划分的目录名 | `artifacts/splits/<task_id>/<split_id>/` |
| `run_id` | 一次训练的结果包目录名 | `artifacts/runs/<run_id>/` |
| `group_id` | 划分/评价的隔离单位 | manifest、split |
| `label_source` | 标签证据：`directory:<folder>` 或复核表行号 | manifest |
| `view_hash` | 音频视图指纹（影响窗口复用） | windows.csv、manifest |
| `feature_key` | 特征身份：`<feature_type>_<hash16>` | 缓存目录、run.json |

不存在的概念：`fixture` 是测试用合成音频，不是数据字段；`mic`、`origin` 只在 rich
adapter 内部使用。

---

## 7. 常见失败与处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `prepare` 退出码 3，`label_folder_unmapped` | `label_map` 漏了文件夹 | 补映射；不要靠改名掩盖 |
| `file_outside_label_folder` | 音频直接放在数据集根目录 | 移到某个标签文件夹下 |
| `layout=label_folders requires grouping=file` | 配了 subject 但无说话人 | 改成 `grouping: file` |
| `feature config needs an explicit feature_type` | 特征 yaml 缺 `feature_type` | 加 `feature_type: mfcc` |
| `... belong to the feature recipe` | 预处理里写了 `frame_ms/hop_ms` | 从预处理删除，写进特征 yaml |
| `cache ... belongs to feature_hash ...` | 同目录下换了配方 | 改 `feature_type`/`name` 或删缓存目录 |
| 混采样率被阻塞 | `native` 且多采样率 | 用 16 kHz 预处理，或显式允许混合 |

---

## 8. 验收清单

- [ ] `stats` 退出码 0，`issues.csv` 没有 error；
- [ ] `prepare` 退出码 0，`admission.json.status=approved`；
- [ ] `manifest.csv` 的 `label_source` 全部可追溯（目录名或复核行）；
- [ ] `split` 后 train/val/test 的 `group_id` 无交集；
- [ ] `run --dry-run` 通过，日志里的 `feature_key` 与预期类型一致；
- [ ] `run.json` 的 `features.feature_type` / `dim` 正确；
- [ ] 同样的命令重复执行，第二次命中缓存（`n_from_cache` 大于 0）。
