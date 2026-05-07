from __future__ import annotations

from typing import List

from map_rag_gym.core.schemas import Document
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever
from map_rag_gym.retrieval.common import minmax_normalize, top_k_indices
from map_rag_gym.retrieval.tfidf import LocalTfidfRetriever


class LocalHybridRetriever:
    def __init__(self, corpus_path: str, bm25_weight: float = 0.5) -> None:
        self.bm25 = LocalBM25Retriever(corpus_path)
        self.tfidf = LocalTfidfRetriever(corpus_path)
        self.docs = self.bm25.docs
        self.bm25_weight = bm25_weight

    def raw_scores(self, query: str | dict) -> list[float]:
        bm25_scores = self.bm25.raw_scores(query)
        tfidf_scores = self.tfidf.raw_scores(query)
        bm25_norm = minmax_normalize(bm25_scores)
        tfidf_norm = minmax_normalize(tfidf_scores)
        return [
            self.bm25_weight * bm25_score + (1.0 - self.bm25_weight) * tfidf_score
            for bm25_score, tfidf_score in zip(bm25_norm, tfidf_norm)
        ]

    def search(self, query: str | dict, top_k: int = 3) -> List[Document]:
        scores = self.raw_scores(query)
        indices = top_k_indices(scores, top_k)
        results = []
        for idx in indices:
            doc = self.docs[idx]
            results.append(Document(doc_id=doc.doc_id, title=doc.title, text=doc.text, score=float(scores[idx]), metadata=doc.metadata))
        return results
