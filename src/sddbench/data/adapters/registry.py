"""Adapter registry: dataset_id -> adapter class. New dataset = one entry."""
from __future__ import annotations

from ...config import ConfigError
from .base import DatasetAdapter
from .uaspeech import UASpeechAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "uaspeech": UASpeechAdapter,
}


def get_adapter(dataset_id: str) -> DatasetAdapter:
    if dataset_id not in ADAPTERS:
        raise ConfigError(
            f"no adapter registered for dataset_id={dataset_id!r}; "
            f"registered: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[dataset_id]()
