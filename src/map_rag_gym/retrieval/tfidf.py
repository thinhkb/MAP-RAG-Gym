from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from map_rag_gym.core.schemas import Document
from map_rag_gym.retrieval.common import load_documents, normalize_query, top_k_indices


class LocalTfidfRetriever:
    def __init__(self, corpus_path: str) -> None:
        self.docs = load_documents(corpus_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.doc_texts = [f"{doc.title} {doc.text}".strip() for doc in self.docs]
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def raw_scores(self, query: str | dict) -> list[float]:
        q = normalize_query(query)
        if not q:
            return [0.0 for _ in self.docs]
        query_vec = self.vectorizer.transform([q])
        scores = linear_kernel(query_vec, self.doc_matrix).ravel()
        return [float(score) for score in scores]

    def search(self, query: str | dict, top_k: int = 3) -> List[Document]:
        scores = self.raw_scores(query)
        indices = top_k_indices(scores, top_k)
        results = []
        for idx in indices:
            doc = self.docs[idx]
            results.append(Document(doc_id=doc.doc_id, title=doc.title, text=doc.text, score=float(scores[idx]), metadata=doc.metadata))
        return results
