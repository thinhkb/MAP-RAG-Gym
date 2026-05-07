from __future__ import annotations

from typing import Iterable

SUPPORTED_RETRIEVERS = ("bm25", "tfidf", "hybrid")


def normalize_retriever_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_RETRIEVERS:
        raise ValueError(f"Unsupported retriever '{name}'. Expected one of: {', '.join(SUPPORTED_RETRIEVERS)}.")
    return normalized


def parse_workflow_retriever_overrides(items: Iterable[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not items:
        return overrides
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid workflow retriever override '{raw}'. Expected WORKFLOW=RETRIEVER.")
        workflow_id, retriever_name = raw.split("=", 1)
        workflow_key = workflow_id.strip().upper()
        if not workflow_key:
            raise ValueError(f"Invalid workflow retriever override '{raw}'. Workflow id is empty.")
        overrides[workflow_key] = normalize_retriever_name(retriever_name)
    return overrides
