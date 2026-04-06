from __future__ import annotations

from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ProcessCritic:
    def __init__(self, random_state: int = 13) -> None:
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("features", ColumnTransformer([
                ("question", TfidfVectorizer(ngram_range=(1, 2), max_features=4000), "question"),
                ("action", TfidfVectorizer(ngram_range=(1, 2), max_features=5000), "action_text"),
                ("history", TfidfVectorizer(ngram_range=(1, 2), max_features=1000), "history_text"),
                ("cats", OneHotEncoder(handle_unknown="ignore"), ["module", "workflow_id"]),
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
                    ],
                ),
            ])),
            ("reg", Ridge(alpha=1.0)),
        ])

    def _frame(self, rows: Iterable[dict]) -> pd.DataFrame:
        data = []
        for row in rows:
            data.append({
                "question": row.get("question", ""),
                "action_text": row.get("action_text", ""),
                "history_text": row.get("history_text", ""),
                "module": row.get("module", ""),
                "workflow_id": row.get("workflow_id", ""),
                "step_id": int(row.get("step_id", 0)),
                "action_len": int(row.get("action_len", 0)),
                "num_actions_in_step": int(row.get("num_actions_in_step", 1)),
                "num_docs": int(row.get("num_docs", 0)),
                "selected": int(row.get("selected", 0)),
                "tokens": float(row.get("tokens", 0.0)),
                "retrieval_calls": float(row.get("retrieval_calls", 0.0)),
                "latency_ms": float(row.get("latency_ms", 0.0)),
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
