from __future__ import annotations

from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _text_value(value: object, placeholder: str) -> str:
    text = str(value or "").strip()
    return text if text else placeholder


class ProcessCritic:
    def __init__(self, random_state: int = 13) -> None:
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("features", ColumnTransformer([
                ("question", TfidfVectorizer(ngram_range=(1, 2), max_features=4000), "question"),
                ("query", TfidfVectorizer(ngram_range=(1, 2), max_features=2500), "query_text"),
                ("action", TfidfVectorizer(ngram_range=(1, 2), max_features=5000), "action_text"),
                ("doc_title", TfidfVectorizer(ngram_range=(1, 2), max_features=2500), "doc_title"),
                ("history", TfidfVectorizer(ngram_range=(1, 2), max_features=1000), "history_text"),
                ("cats", OneHotEncoder(handle_unknown="ignore"), ["module", "workflow_id", "retriever_type", "budget_mode"]),
                (
                    "nums",
                    "passthrough",
                    [
                        "step_id",
                        "action_len",
                        "num_actions_in_step",
                        "num_docs",
                        "selected",
                        "tokens",
                        "retrieval_calls",
                        "latency_ms",
                        "doc_rank",
                        "doc_score",
                    ],
                ),
            ])),
            ("reg", Ridge(alpha=1.0)),
        ])

    def _frame(self, rows: Iterable[dict]) -> pd.DataFrame:
        data = []
        for row in rows:
            data.append({
                "question": _text_value(row.get("question", ""), "__empty_question__"),
                "query_text": _text_value(row.get("query_text", ""), "__empty_query__"),
                "action_text": _text_value(row.get("action_text", ""), "__empty_action__"),
                "doc_title": _text_value(row.get("doc_title", ""), "__empty_doc_title__"),
                "history_text": _text_value(row.get("history_text", ""), "__empty_history__"),
                "module": row.get("module", ""),
                "workflow_id": row.get("workflow_id", ""),
                "retriever_type": row.get("retriever_type", ""),
                "budget_mode": row.get("budget_mode", "medium"),
                "step_id": int(row.get("step_id", 0)),
                "action_len": int(row.get("action_len", 0)),
                "num_actions_in_step": int(row.get("num_actions_in_step", 1)),
                "num_docs": int(row.get("num_docs", 0)),
                "selected": int(row.get("selected", 0)),
                "tokens": float(row.get("tokens", 0.0)),
                "retrieval_calls": float(row.get("retrieval_calls", 0.0)),
                "latency_ms": float(row.get("latency_ms", 0.0)),
                "doc_rank": int(row.get("doc_rank", 0)),
                "doc_score": float(row.get("doc_score", 0.0)),
            })
        return pd.DataFrame(data)

    def fit(self, rows: list[dict], targets: list[float]) -> None:
        X = self._frame(rows)
        self.pipeline.fit(X, targets)

    def predict(self, rows: list[dict]) -> list[float]:
        X = self._frame(rows)
        preds = self.pipeline.predict(X)
        return [float(value) for value in preds]

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        self.pipeline = joblib.load(path)
