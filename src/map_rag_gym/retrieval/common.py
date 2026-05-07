from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

from map_rag_gym.core.schemas import Document


class RetrieverBackend(Protocol):
    def search(self, query: str | dict, top_k: int = 3) -> list[Document]:
        ...


def load_documents(path: str | Path) -> list[Document]:
    corpus_path = Path(path)
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [Document(**row) for row in data]


def normalize_query(query: str | dict) -> str:
    if isinstance(query, dict):
        query = str(query.get("query") or query.get("question") or query.get("text") or query.get("raw_text") or "").strip()
    return str(query).strip()


def top_k_indices(scores: Iterable[float], top_k: int) -> list[int]:
    indexed = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in indexed[:top_k]]


def minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.0 for _ in scores]
    return [(score - lo) / (hi - lo) for score in scores]
