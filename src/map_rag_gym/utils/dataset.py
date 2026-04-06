from __future__ import annotations

import random
from typing import Any


def normalize_qa_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        row = dict(item)
        row.setdefault("id", str(idx))
        normalized.append(row)
    return normalized


def split_qa_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive number.")

    normalized = normalize_qa_records(records)
    shuffled = list(normalized)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_count = int(n * (train_ratio / total))
    val_count = int(n * (val_ratio / total))
    train_end = train_count
    val_end = train_count + val_count

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
