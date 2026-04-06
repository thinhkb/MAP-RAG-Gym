from __future__ import annotations

import platform
import random
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a declared dependency but keep this resilient.
    np = None


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(path: str | None) -> str | None:
    if not path:
        return None
    return str(Path(path).resolve())


def try_get_git_commit(cwd: str | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def detect_dataset_name(qa_path: str, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    parts = [part for part in Path(qa_path).parts if part not in {".", ".."}]
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(qa_path).stem


def detect_split_name(qa_path: str, explicit_split: str | None = None) -> str:
    if explicit_split:
        return explicit_split
    stem = Path(qa_path).stem.lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "qa": "all",
        "all": "all",
    }
    for key, value in aliases.items():
        if key in stem:
            return value
    return "custom"


def build_experiment_manifest(
    *,
    script_name: str,
    qa_path: str,
    corpus_path: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    limit: int | None = None,
    effective_questions: int | None = None,
    seed: int | None = None,
    prompt_version: str = "v1",
    router_model_path: str | None = None,
    utility_config: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "script": script_name,
        "generated_at_utc": utc_now_iso(),
        "git_commit": try_get_git_commit(Path.cwd()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "paths": {
            "qa": resolve_path(qa_path),
            "corpus": resolve_path(corpus_path),
            "router_model": resolve_path(router_model_path),
        },
        "dataset": {
            "name": detect_dataset_name(qa_path, dataset_name),
            "split": detect_split_name(qa_path, dataset_split),
            "limit": limit,
            "effective_questions": effective_questions,
        },
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
        },
        "reproducibility": {
            "seed": seed,
            "prompt_version": prompt_version,
            "utility": deepcopy(utility_config) if utility_config is not None else None,
        },
        "settings": deepcopy(settings) if settings else {},
    }
    return manifest


def get_nested_value(payload: dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def compare_manifest_fields(
    existing: dict[str, Any],
    expected: dict[str, Any],
    field_paths: list[tuple[str, ...]],
) -> list[str]:
    mismatches: list[str] = []
    for path in field_paths:
        left = get_nested_value(existing, path)
        right = get_nested_value(expected, path)
        if left != right:
            mismatches.append(".".join(path))
    return mismatches
