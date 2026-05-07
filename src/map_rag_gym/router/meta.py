from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from map_rag_gym.core.features import extract_question_features
from map_rag_gym.core.schemas import PlannerDecision
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.router.rule_based import RuleBasedRouter


class MetaRouterGate:
    def __init__(self, random_state: int = 13) -> None:
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("features", ColumnTransformer([
                ("question", TfidfVectorizer(ngram_range=(1, 2), max_features=4000), "question"),
                ("cats", OneHotEncoder(handle_unknown="ignore"), ["learned_workflow", "rule_workflow", "wh_word", "budget_mode"]),
                (
                    "nums",
                    "passthrough",
                    [
                        "learned_confidence",
                        "rule_confidence",
                        "token_len",
                        "comparative_flag",
                        "conjunction_flag",
                        "ambiguity_flag",
                        "estimated_hops",
                    ],
                ),
            ])),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])

    def _frame(self, rows: Iterable[dict]) -> pd.DataFrame:
        data = []
        for row in rows:
            feat = asdict(extract_question_features(row["question"]))
            data.append({
                "question": row["question"],
                "learned_workflow": row["learned_workflow"],
                "rule_workflow": row["rule_workflow"],
                "learned_confidence": float(row["learned_confidence"]),
                "rule_confidence": float(row["rule_confidence"]),
                "budget_mode": str(row.get("budget_mode", "medium")),
                "wh_word": feat["wh_word"],
                "token_len": feat["token_len"],
                "comparative_flag": feat["comparative_flag"],
                "conjunction_flag": feat["conjunction_flag"],
                "ambiguity_flag": feat["ambiguity_flag"],
                "estimated_hops": feat["estimated_hops"],
            })
        return pd.DataFrame(data)

    def fit(self, rows: list[dict], labels: list[str], sample_weight: list[float] | None = None) -> None:
        X = self._frame(rows)
        fit_kwargs = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
        self.pipeline.fit(X, labels, **fit_kwargs)

    def predict(self, row: dict) -> tuple[str, float]:
        X = self._frame([row])
        probs = self.pipeline.predict_proba(X)[0]
        classes = [str(item) for item in self.pipeline.named_steps["clf"].classes_]
        idx = int(probs.argmax())
        return classes[idx], float(probs[idx])

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        self.pipeline = joblib.load(path)


class MetaRouterPolicy:
    def __init__(
        self,
        gate: MetaRouterGate,
        learned_router: LearnedRouter,
        rule_router: RuleBasedRouter | None = None,
    ) -> None:
        self.gate = gate
        self.learned_router = learned_router
        self.rule_router = rule_router or RuleBasedRouter()

    def decide(self, question: str, budget_mode: str = "medium") -> PlannerDecision:
        learned_workflow, learned_conf = self.learned_router.predict(question, budget_mode=budget_mode)
        rule_decision = self.rule_router.decide(question, budget_mode=budget_mode)
        choice, confidence = self.gate.predict({
            "question": question,
            "learned_workflow": learned_workflow,
            "learned_confidence": learned_conf,
            "rule_workflow": rule_decision.workflow_id,
            "rule_confidence": rule_decision.confidence,
            "budget_mode": budget_mode,
        })
        if choice == "learned":
            return PlannerDecision(
                workflow_id=learned_workflow,
                confidence=confidence,
                reason=(
                    f"meta:learned(gate={confidence:.4f}, learned={learned_workflow}, "
                    f"learned_conf={learned_conf:.4f}, budget={budget_mode})"
                ),
            )
        return PlannerDecision(
            workflow_id=rule_decision.workflow_id,
            confidence=confidence,
            reason=(
                f"meta:rule(gate={confidence:.4f}, rule={rule_decision.workflow_id}, "
                f"learned={learned_workflow}, learned_conf={learned_conf:.4f}, budget={budget_mode})"
            ),
        )
