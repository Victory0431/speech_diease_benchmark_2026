"""D12: machine-readable parameter registry for the web console.

The schema is the single source of truth shared by the browser forms, the
server-side validation and the generated YAML files.  Every field carries
`impact` so the UI can warn which cached artifact becomes stale when it
changes (window index, feature cache, split, or nothing).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

SCHEMA_VERSION = "1.0"

IMPACT_LABELS = {
    "windows": "需要重新生成窗口索引与特征缓存",
    "features": "需要重新计算特征缓存",
    "split": "需要重新生成数据划分",
    "training": "只影响本次训练，可直接重跑",
}


def _f(key: str, label: str, kind: str, default, *, impact: str,
       help: str = "", options: list | None = None, minimum=None,
       maximum=None, step=None, nullable: bool = False) -> dict:
    return {
        "key": key, "label": label, "type": kind, "default": default,
        "impact": impact, "help": help, "options": options,
        "min": minimum, "max": maximum, "step": step, "nullable": nullable,
    }


SECTIONS = [
    {
        "id": "preprocess",
        "title": "音频窗口",
        "description": "决定每条录音被截成什么样子；改动后必须重新生成窗口与特征缓存。",
        "fields": [
            _f("preprocess.sample_rate_policy", "采样率策略", "select", "native",
               impact="windows", options=["native", "resample"],
               help="native 保留原始采样率；resample 统一到目标采样率。"),
            _f("preprocess.target_sample_rate", "目标采样率 (Hz)", "int", None,
               impact="windows", nullable=True, minimum=8000, maximum=96000,
               help="仅 resample 时生效，例如 16000。"),
            _f("preprocess.channel_policy", "通道策略", "select", "select",
               impact="windows", options=["select", "require_mono"],
               help="select 取指定通道；require_mono 要求单声道。"),
            _f("preprocess.channel_index", "通道序号", "int", 0,
               impact="windows", minimum=0, maximum=7),
            _f("preprocess.observe_seconds", "观察长度 (秒)", "float", 4.0,
               impact="windows", minimum=0.5, maximum=30.0, step=0.5,
               help="每条录音实际送入模型的时长。"),
            _f("preprocess.window_seconds", "窗口长度 (秒)", "float", 4.0,
               impact="windows", minimum=0.5, maximum=30.0, step=0.5,
               help="当前只支持单段视图，必须与观察长度一致。"),
            _f("preprocess.position", "截取位置", "select", "head",
               impact="windows", options=["head", "center", "tail", "fixed_random"],
               help="fixed_random 使用 crop_seed 做确定性随机。"),
            _f("preprocess.short_policy", "不足长度处理", "select", "right_zero_pad",
               impact="windows", options=["right_zero_pad"],
               help="当前只实现了右侧补零。"),
            _f("preprocess.frame_ms", "帧长 (ms)", "float", 25.0,
               impact="windows", minimum=5.0, maximum=100.0, step=1.0,
               help="用于计算窗口索引的 n_fft。"),
            _f("preprocess.hop_ms", "帧移 (ms)", "float", 10.0,
               impact="windows", minimum=1.0, maximum=50.0, step=1.0),
            _f("preprocess.amplitude_normalization", "幅度归一化", "select", "none",
               impact="windows", options=["none"],
               help="当前锁定为 none，避免引入未评审的增益策略。"),
        ],
    },
    {
        "id": "features",
        "title": "MFCC 特征",
        "description": "在有效帧上计算 MFCC 统计量；改动后必须重新计算特征缓存。",
        "fields": [
            _f("features.n_mfcc", "MFCC 维数", "int", 20,
               impact="features", minimum=1, maximum=80,
               help="每个窗口最终维度 = 2 × n_mfcc（mean + std）。"),
            _f("features.n_mels", "Mel 滤波器数", "int", 64,
               impact="features", minimum=8, maximum=256),
            _f("features.frame_ms", "帧长 (ms)", "float", 25.0,
               impact="features", minimum=5.0, maximum=100.0, step=1.0),
            _f("features.hop_ms", "帧移 (ms)", "float", 10.0,
               impact="features", minimum=1.0, maximum=50.0, step=1.0),
            _f("features.f_min_hz", "最低频率 (Hz)", "float", 0.0,
               impact="features", minimum=0.0, maximum=8000.0, step=10.0),
            _f("features.f_max_policy", "最高频率策略", "select", "nyquist",
               impact="features", options=["nyquist", "fixed"],
               help="nyquist 跟随采样率；fixed 使用下面的固定值。"),
            _f("features.f_max_hz", "最高频率 (Hz)", "float", None,
               impact="features", nullable=True, minimum=100.0, maximum=24000.0,
               help="仅 fixed 时生效。"),
            _f("features.include_c0", "包含 C0", "bool", True,
               impact="features", help="关闭后第一个系数置零。"),
        ],
    },
    {
        "id": "split",
        "title": "数据划分",
        "description": "按 group_id 做组隔离；seed 只影响哪些人/录音进入哪个划分。",
        "fields": [
            _f("split.seed", "划分 seed", "int", 0,
               impact="split", minimum=0, maximum=10 ** 9,
               help="同一个人 manifest 下，seed 决定 test 是哪一批人。"),
            _f("split.train_frac", "train 比例", "float", 0.6,
               impact="split", minimum=0.1, maximum=0.9, step=0.05),
            _f("split.val_frac", "val 比例", "float", 0.2,
               impact="split", minimum=0.05, maximum=0.5, step=0.05),
            _f("split.test_frac", "test 比例", "float", 0.2,
               impact="split", minimum=0.05, maximum=0.5, step=0.05),
        ],
    },
    {
        "id": "model",
        "title": "模型结构",
        "description": "当前只实现了冻结特征 MLP；输入维度由特征缓存自动推导。",
        "fields": [
            _f("model.hidden_dims", "隐藏层宽度", "list_int", [128],
               impact="training", help="逗号分隔，例如 128 或 64,32。"),
            _f("model.dropout", "dropout", "float", 0.2,
               impact="training", minimum=0.0, maximum=0.9, step=0.05),
            _f("model.activation", "激活函数", "select", "relu",
               impact="training", options=["relu"]),
        ],
    },
    {
        "id": "training",
        "title": "训练",
        "description": "固定随机性：Python/NumPy/PyTorch 三处 seed + 单线程 CPU。",
        "fields": [
            _f("training.seed", "训练 seed", "int", 0,
               impact="training", minimum=0, maximum=10 ** 9),
            _f("training.epochs", "最大 epoch", "int", 100,
               impact="training", minimum=1, maximum=500),
            _f("training.batch_size", "batch size", "int", 128,
               impact="training", minimum=1, maximum=4096),
            _f("training.optimizer", "优化器", "select", "adamw",
               impact="training", options=["adamw", "adam"]),
            _f("training.learning_rate", "学习率", "float", 0.001,
               impact="training", minimum=1e-6, maximum=1.0, step=0.0001),
            _f("training.weight_decay", "weight decay", "float", 0.0001,
               impact="training", minimum=0.0, maximum=1.0, step=0.0001),
            _f("training.early_stopping_patience", "早停耐心", "int", 15,
               impact="training", minimum=1, maximum=200),
            _f("training.class_weighting", "类别权重", "select", "subject_balanced",
               impact="training", options=["subject_balanced", "none"],
               help="subject_balanced 让每位说话人的总权重相等。"),
            _f("training.device", "设备", "select", "cpu",
               impact="training", options=["cpu", "cuda"]),
        ],
    },
    {
        "id": "evaluation",
        "title": "评价口径",
        "description": "主评价单位由数据集 grouping 决定：subject 库用 subject，file 库用 recording。",
        "fields": [
            _f("evaluation.unit", "主评价单位", "select", "subject",
               impact="training", options=["subject", "recording"],
               help="grouping=file 时必须为 recording。"),
        ],
    },
]


def _section_map() -> dict:
    return {s["id"]: s for s in SECTIONS}


def get_schema() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "impact_labels": IMPACT_LABELS,
        "sections": copy.deepcopy(SECTIONS),
    }


def defaults() -> dict:
    out: dict = {}
    for section in SECTIONS:
        for field in section["fields"]:
            out[field["key"]] = field["default"]
    return out


def _coerce(field: dict, raw):
    kind = field["type"]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if field.get("nullable"):
            return None
        raw = field["default"]
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if kind == "int":
        value = int(raw)
    elif kind == "float":
        value = float(raw)
    elif kind == "list_int":
        if isinstance(raw, (list, tuple)):
            value = [int(v) for v in raw]
        else:
            value = [int(p) for p in str(raw).replace(";", ",").split(",") if p.strip()]
    elif kind == "select":
        value = str(raw)
        options = field.get("options") or []
        if options and value not in options:
            raise ValueError(f"{field['label']} must be one of {options}, got {value!r}")
    else:
        value = str(raw)
    if kind in ("int", "float"):
        lo, hi = field.get("min"), field.get("max")
        if lo is not None and value < lo:
            raise ValueError(f"{field['label']} must be >= {lo}")
        if hi is not None and value > hi:
            raise ValueError(f"{field['label']} must be <= {hi}")
    if kind == "list_int" and not value:
        raise ValueError(f"{field['label']} cannot be empty")
    return value


def validate_values(values: dict) -> dict:
    """Return the flattened, type-checked values (unknown keys are rejected)."""
    known = {f["key"]: f for s in SECTIONS for f in s["fields"]}
    clean: dict = {}
    for key, raw in (values or {}).items():
        if key not in known:
            raise ValueError(f"unknown parameter {key!r}")
        clean[key] = _coerce(known[key], raw)
    for key, field in known.items():
        clean.setdefault(key, field["default"])
    if clean["split.train_frac"] + clean["split.val_frac"] + clean["split.test_frac"] > 1.0 + 1e-9:
        raise ValueError("train/val/test proportions must sum to 1.0 or less")
    if clean["preprocess.sample_rate_policy"] == "resample" \
            and not clean["preprocess.target_sample_rate"]:
        raise ValueError("resample 策略必须填写目标采样率")
    if clean["features.f_max_policy"] == "fixed" and not clean["features.f_max_hz"]:
        raise ValueError("f_max_policy=fixed 必须填写最高频率")
    if clean["features.n_mfcc"] > clean["features.n_mels"]:
        raise ValueError("n_mfcc 不能大于 n_mels")
    if clean["preprocess.observe_seconds"] != clean["preprocess.window_seconds"]:
        raise ValueError("当前只支持单段视图，观察长度必须等于窗口长度")
    return clean


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _base_files(repo_root: Path, dataset_id: str) -> tuple[dict, dict]:
    """Best-effort base configs: latest experiment for this dataset, else defaults."""
    best: tuple[float, dict] | None = None
    for path in sorted((repo_root / "configs" / "experiments").glob("*.yaml")):
        cfg = _load_yaml(path)
        if cfg.get("dataset") != dataset_id:
            continue
        mtime = path.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, cfg)
    preprocess, features = {}, {}
    if best:
        cfg = best[1]
        features_rel = (cfg.get("features") or {}).get("config")
        preprocess_rel = (cfg.get("features") or {}).get("preprocess")
        if features_rel:
            features = _load_yaml(repo_root / features_rel)
        if preprocess_rel:
            preprocess = _load_yaml(repo_root / preprocess_rel)
    if not features:
        features = _load_yaml(repo_root / "configs" / "features" / "mfcc20_stats.yaml")
    if not preprocess:
        preprocess = _load_yaml(
            repo_root / "configs" / "preprocessing" / "uaspeech_native_4s_head.yaml")
    return preprocess, features


def dataset_defaults(repo_root: Path, dataset_id: str, dataset_cfg: dict,
                     experiment_cfg: dict | None = None) -> dict:
    """Flattened defaults used to prefill the browser forms."""
    preprocess, features = _base_files(repo_root, dataset_id)
    experiment_cfg = experiment_cfg or {}
    values = defaults()
    for key, value in preprocess.items():
        flat = f"preprocess.{key}"
        if flat in values and value is not None:
            values[flat] = value
    for key, value in features.items():
        flat = f"features.{key}"
        if flat in values and value is not None:
            values[flat] = value
    for group in ("model", "training", "evaluation"):
        for key, value in (experiment_cfg.get(group) or {}).items():
            flat = f"{group}.{key}"
            if flat in values:
                values[flat] = value
    grouping = str(dataset_cfg.get("grouping") or "subject")
    values["evaluation.unit"] = "recording" if grouping == "file" else "subject"
    return values


def build_configs(repo_root: Path, dataset_id: str, dataset_cfg: dict,
                  experiment_id: str, prepared_id: str, split_id: str,
                  values: dict) -> tuple[dict, dict, dict]:
    """Return (experiment, preprocess, feature) dicts ready to dump as YAML."""
    clean = validate_values(values)
    base_pre, base_feat = _base_files(repo_root, dataset_id)
    preprocess = copy.deepcopy(base_pre)
    features = copy.deepcopy(base_feat)
    for key, value in clean.items():
        section, name = key.split(".", 1)
        if section == "preprocess":
            preprocess[name] = value
        elif section == "features":
            features[name] = value
    # Keep the base recipe names: the cache path is name + config hash, so
    # renaming would force a rebuild even when only model parameters changed.
    preprocess.setdefault("name", f"web_{experiment_id}")
    features.setdefault("name", f"web_{experiment_id}")

    if clean["preprocess.window_seconds"] != preprocess.get("window_seconds"):
        preprocess["window_seconds"] = clean["preprocess.window_seconds"]

    task_id = str(dataset_cfg.get("task_id") or f"{dataset_id}_v1")
    experiment = {
        "experiment_id": experiment_id,
        "experiment_name": f"{dataset_id} web console",
        "experiment_group": "web_console",
        "purpose": "generated from the D12 web console; edit configs/experiments for正式基线",
        "stage": "exploration",
        "task_id": task_id,
        "dataset": dataset_id,
        "prepared_id": prepared_id,
        "split_id": split_id,
        "features": {
            "config": f"configs/experiments/generated/{experiment_id}/features.yaml",
            "preprocess":
                f"configs/experiments/generated/{experiment_id}/preprocess.yaml",
        },
        "model": {
            "name": "mlp",
            "hidden_dims": clean["model.hidden_dims"],
            "dropout": clean["model.dropout"],
            "activation": clean["model.activation"],
        },
        "training": {
            "seed": clean["training.seed"],
            "epochs": clean["training.epochs"],
            "batch_size": clean["training.batch_size"],
            "optimizer": clean["training.optimizer"],
            "learning_rate": clean["training.learning_rate"],
            "weight_decay": clean["training.weight_decay"],
            "early_stopping_patience": clean["training.early_stopping_patience"],
            "class_weighting": clean["training.class_weighting"],
            "device": clean["training.device"],
            "selection_metric": f"{clean['evaluation.unit']}_balanced_accuracy",
            "selection_direction": "max",
        },
        "evaluation": {
            "unit": clean["evaluation.unit"],
            "primary_metric": "balanced_accuracy",
        },
        "labels": {
            "class_order": (dataset_cfg.get("labels") or {}).get(
                "class_order", ["control", "disorder"]),
            "positive_class": (dataset_cfg.get("labels") or {}).get(
                "positive_class", "disorder"),
        },
    }
    return experiment, preprocess, features


def write_configs(repo_root: Path, experiment: dict, preprocess: dict,
                  features: dict) -> dict:
    out_dir = repo_root / "configs" / "experiments" / "generated" / experiment["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "experiment": out_dir / "experiment.yaml",
        "preprocess": out_dir / "preprocess.yaml",
        "features": out_dir / "features.yaml",
    }
    for name, doc in (("experiment", experiment), ("preprocess", preprocess),
                      ("features", features)):
        paths[name].write_text(
            yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {k: str(v.relative_to(repo_root)) for k, v in paths.items()}
