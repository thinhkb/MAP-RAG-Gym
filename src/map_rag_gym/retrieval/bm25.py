from __future__ import annotations

import json
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi

from map_rag_gym.core.schemas import Document


class LocalBM25Retriever:
    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = Path(corpus_path)
        self.docs = self._load_docs(self.corpus_path)
        self.tokens = [d.text.lower().split() for d in self.docs]
        self.bm25 = BM25Okapi(self.tokens)

    def _load_docs(self, path: Path) -> List[Document]:
        data = json.loads(path.read_text(encoding='utf-8'))
        return [Document(**row) for row in data]

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        # Handle dict inputs by extracting text
        if isinstance(query, dict):
            query = str(query.get("query") or query.get("question") or query.get("text") or query.get("raw_text") or "").strip()
        query = str(query).strip()
        scores = self.bm25.get_scores(query.lower().split())
        pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in pairs:
            doc = self.docs[idx]
            results.append(Document(doc_id=doc.doc_id, title=doc.title, text=doc.text, score=float(score), metadata=doc.metadata))
        return results
