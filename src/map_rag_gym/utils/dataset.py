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
    train_count: int | None = None,
    val_count: int | None = None,
    test_count: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    normalized = normalize_qa_records(records)
    shuffled = list(normalized)
    random.Random(seed).shuffle(shuffled)

    exact_counts = [train_count, val_count, test_count]
    if any(count is not None for count in exact_counts):
        if not all(count is not None for count in exact_counts):
            raise ValueError("Exact split counts require train_count, val_count, and test_count together.")
        if any(count < 0 for count in exact_counts if count is not None):
            raise ValueError("Exact split counts must be non-negative.")
        total_requested = int(train_count) + int(val_count) + int(test_count)
        if total_requested > len(shuffled):
            raise ValueError(
                f"Requested {total_requested} split rows but only {len(shuffled)} records are available."
            )
        train_end = int(train_count)
        val_end = int(train_count) + int(val_count)
    else:
        total = train_ratio + val_ratio + test_ratio
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive number.")

        n = len(shuffled)
        train_count = int(n * (train_ratio / total))
        val_count = int(n * (val_ratio / total))
        train_end = train_count
        val_end = train_count + val_count

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:] if not all(count is not None for count in exact_counts) else shuffled[val_end : val_end + int(test_count)],
    }
