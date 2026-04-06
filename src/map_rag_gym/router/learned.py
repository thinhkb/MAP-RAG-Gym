from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from map_rag_gym.core.features import extract_question_features


class LearnedRouter:
    def __init__(self, random_state: int = 13) -> None:
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("features", ColumnTransformer([
                ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=4000), "question"),
                ("cats", OneHotEncoder(handle_unknown="ignore"), ["wh_word"]),
                ("nums", "passthrough", ["token_len", "comparative_flag", "conjunction_flag", "ambiguity_flag", "estimated_hops"]),
            ])),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))
        ])

    def _frame(self, questions: Iterable[str]) -> pd.DataFrame:
        rows = []
        for q in questions:
            feat = asdict(extract_question_features(q))
            rows.append(feat)
        return pd.DataFrame(rows)

    def fit(self, questions: List[str], labels: List[str]) -> None:
        X = self._frame(questions)
        self.pipeline.fit(X, labels)

    def predict(self, question: str) -> Tuple[str, float]:
        X = self._frame([question])
        probs = self.pipeline.predict_proba(X)[0]
        classes = list(self.pipeline.named_steps["clf"].classes_)
        idx = int(probs.argmax())
        return str(classes[idx]), float(probs[idx])

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        self.pipeline = joblib.load(path)
