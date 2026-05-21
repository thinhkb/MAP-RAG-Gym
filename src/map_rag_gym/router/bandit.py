from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from map_rag_gym.core.features import extract_question_features
from map_rag_gym.evaluation.heuristics import token_overlap
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.router.rule_based import RuleBasedRouter
from map_rag_gym.router.budget import ALLOWED_WORKFLOWS_BY_BUDGET, normalize_budget_mode


class BanditRouter:
    def __init__(
        self,
        random_state: int = 13,
        alpha: float = 1.0,
        default_budget_mode: str = "low",
        allowed_workflows: list[str] | None = None,
        preference_margin: float = 0.0,
        preferred_workflows: list[str] | None = None,
    ) -> None:
        self.random_state = random_state
        self.alpha = alpha
        self.default_budget_mode = normalize_budget_mode(default_budget_mode)
        self.allowed_workflows = [str(workflow).upper() for workflow in (allowed_workflows or [])]
        self.preference_margin = float(preference_margin)
        self.preferred_workflows = [str(workflow).upper() for workflow in (preferred_workflows or [])]
        self.workflow_models: dict[str, Pipeline] = {}
        self.rule_router = RuleBasedRouter()
        self.aux_learned_router: LearnedRouter | None = None
        self.probe_retriever = None
        self._context_cache: dict[tuple[str, str], dict] = {}

    def _normalize_allowed_workflows(self, budget_mode: str, candidate_workflows: list[str] | None = None) -> list[str]:
        allowed = [str(workflow).upper() for workflow in (candidate_workflows or self.allowed_workflows or ALLOWED_WORKFLOWS_BY_BUDGET[budget_mode])]
        unique = []
        for workflow in allowed:
            if workflow not in unique:
                unique.append(workflow)
        return unique

    def attach_learned_router(self, router: LearnedRouter | None) -> None:
        self.aux_learned_router = router
        self._context_cache = {}

    def attach_probe_retriever(self, retriever) -> None:
        self.probe_retriever = retriever
        self._context_cache = {}

    def _build_context(self, question: str, budget_mode: str) -> dict:
        cache_key = (question, budget_mode)
        if cache_key in self._context_cache:
            return dict(self._context_cache[cache_key])
        feat = asdict(extract_question_features(question))
        rule_decision = self.rule_router.decide(question, budget_mode=budget_mode)
        context = {
            "question": question,
            "budget_mode": budget_mode,
            "wh_word": feat["wh_word"],
            "token_len": feat["token_len"],
            "comparative_flag": feat["comparative_flag"],
            "conjunction_flag": feat["conjunction_flag"],
            "ambiguity_flag": feat["ambiguity_flag"],
            "temporal_flag": feat.get("temporal_flag", 0),
            "negation_flag": feat.get("negation_flag", 0),
            "superlative_flag": feat.get("superlative_flag", 0),
            "multi_entity_flag": feat.get("multi_entity_flag", 0),
            "entity_density": feat.get("entity_density", 0.0),
            "estimated_hops": feat["estimated_hops"],
            "rule_workflow": rule_decision.workflow_id,
            "rule_confidence": float(rule_decision.confidence),
            "learned_top_workflow": "NONE",
            "learned_confidence": 0.0,
            "learned_margin": 0.0,
            "probe_top1_score": 0.0,
            "probe_top2_score": 0.0,
            "probe_top3_mean_score": 0.0,
            "probe_score_gap12": 0.0,
            "probe_doc_overlap_mean": 0.0,
            "probe_title_overlap_max": 0.0,
            "probe_num_docs": 0,
        }
        for workflow_id in ("W1", "W2", "W3", "W4", "W5", "W6"):
            context[f"learned_prob_{workflow_id}"] = 0.0

        if self.aux_learned_router is not None:
            score_map = self.aux_learned_router.predict_scores(question, budget_mode=budget_mode)
            ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
            if ranked:
                context["learned_top_workflow"] = ranked[0][0]
                context["learned_confidence"] = float(ranked[0][1])
                second = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
                context["learned_margin"] = float(ranked[0][1] - second)
            for workflow_id in ("W1", "W2", "W3", "W4", "W5", "W6"):
                context[f"learned_prob_{workflow_id}"] = float(score_map.get(workflow_id, 0.0))
        if self.probe_retriever is not None:
            docs = list(self.probe_retriever.search(question, top_k=3))
            probe_scores = [float(doc.score) for doc in docs]
            doc_overlaps = [token_overlap(question, f"{doc.title} {doc.text}") for doc in docs]
            title_overlaps = [token_overlap(question, doc.title) for doc in docs]
            context["probe_top1_score"] = probe_scores[0] if probe_scores else 0.0
            context["probe_top2_score"] = probe_scores[1] if len(probe_scores) > 1 else 0.0
            context["probe_top3_mean_score"] = sum(probe_scores) / len(probe_scores) if probe_scores else 0.0
            context["probe_score_gap12"] = (
                probe_scores[0] - probe_scores[1]
                if len(probe_scores) > 1
                else (probe_scores[0] if probe_scores else 0.0)
            )
            context["probe_doc_overlap_mean"] = sum(doc_overlaps) / len(doc_overlaps) if doc_overlaps else 0.0
            context["probe_title_overlap_max"] = max(title_overlaps) if title_overlaps else 0.0
            context["probe_num_docs"] = len(docs)
        self._context_cache[cache_key] = dict(context)
        return context

    def _frame(self, rows: Iterable[dict]) -> pd.DataFrame:
        data = []
        for row in rows:
            mode = normalize_budget_mode(row.get("budget_mode", self.default_budget_mode))
            data.append(self._build_context(row["question"], mode))
        return pd.DataFrame(data)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("features", ColumnTransformer([
                ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=4000), "question"),
                ("cats", OneHotEncoder(handle_unknown="ignore"), ["wh_word", "budget_mode", "rule_workflow", "learned_top_workflow"]),
                (
                    "nums",
                    "passthrough",
                    [
                        "token_len",
                        "comparative_flag",
                        "conjunction_flag",
                        "ambiguity_flag",
                        "temporal_flag",
                        "negation_flag",
                        "superlative_flag",
                        "multi_entity_flag",
                        "entity_density",
                        "estimated_hops",
                        "rule_confidence",
                        "learned_confidence",
                        "learned_margin",
                        "learned_prob_W1",
                        "learned_prob_W2",
                        "learned_prob_W3",
                        "learned_prob_W4",
                        "learned_prob_W5",
                        "learned_prob_W6",
                        "probe_top1_score",
                        "probe_top2_score",
                        "probe_top3_mean_score",
                        "probe_score_gap12",
                        "probe_doc_overlap_mean",
                        "probe_title_overlap_max",
                        "probe_num_docs",
                    ],
                ),
            ])),
            ("reg", Ridge(alpha=self.alpha)),
        ])

    def fit(self, rows: list[dict], rewards: list[float], sample_weight: list[float] | None = None) -> None:
        grouped_rows: dict[str, list[dict]] = {}
        grouped_rewards: dict[str, list[float]] = {}
        grouped_weights: dict[str, list[float]] = {}
        for idx, row in enumerate(rows):
            workflow_id = str(row["workflow_id"]).upper()
            grouped_rows.setdefault(workflow_id, []).append(row)
            grouped_rewards.setdefault(workflow_id, []).append(float(rewards[idx]))
            if sample_weight is not None:
                grouped_weights.setdefault(workflow_id, []).append(float(sample_weight[idx]))

        self.workflow_models = {}
        for workflow_id, workflow_rows in grouped_rows.items():
            pipeline = self._build_pipeline()
            X = self._frame(workflow_rows)
            fit_kwargs = {"reg__sample_weight": grouped_weights[workflow_id]} if sample_weight is not None else {}
            pipeline.fit(X, grouped_rewards[workflow_id], **fit_kwargs)
            self.workflow_models[workflow_id] = pipeline

    def predict_scores(
        self,
        question: str,
        budget_mode: str | None = None,
        candidate_workflows: list[str] | None = None,
    ) -> dict[str, float]:
        mode = normalize_budget_mode(budget_mode or self.default_budget_mode)
        workflows = self._normalize_allowed_workflows(mode, candidate_workflows)
        scores = {}
        for workflow in workflows:
            if workflow not in self.workflow_models:
                raise ValueError(f"Workflow '{workflow}' is not available in this bandit model.")
            row = {"question": question, "budget_mode": mode}
            preds = self.workflow_models[workflow].predict(self._frame([row]))
            scores[workflow] = float(preds[0])
        return scores

    def predict_row_rewards(self, rows: list[dict]) -> list[float]:
        rewards: list[float] = []
        for row in rows:
            workflow = str(row["workflow_id"]).upper()
            if workflow not in self.workflow_models:
                raise ValueError(f"Workflow '{workflow}' is not available in this bandit model.")
            pred = self.workflow_models[workflow].predict(self._frame([row]))
            rewards.append(float(pred[0]))
        return rewards

    def predict_with_scores(
        self,
        question: str,
        budget_mode: str | None = None,
        candidate_workflows: list[str] | None = None,
    ) -> tuple[str, float, dict[str, float]]:
        scores = self.predict_scores(question, budget_mode=budget_mode, candidate_workflows=candidate_workflows)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_workflow, best_score = ranked[0]
        if self.preferred_workflows and self.preference_margin > 0:
            preferred_candidates = [
                item
                for item in ranked
                if item[0] in self.preferred_workflows and (best_score - item[1]) <= self.preference_margin
            ]
            if preferred_candidates:
                best_workflow, best_score = preferred_candidates[0]
                ranked = sorted(
                    scores.items(),
                    key=lambda item: (item[0] != best_workflow, -item[1]),
                )
        second_score = ranked[1][1] if len(ranked) > 1 else best_score
        confidence = 1.0 / (1.0 + math.exp(-4.0 * max(0.0, best_score - second_score)))
        return best_workflow, float(confidence), scores

    def predict_with_gate(
        self,
        question: str,
        *,
        budget_mode: str | None = None,
        candidate_workflows: list[str] | None = None,
        baseline_workflow: str | None = None,
        minimum_advantage: float = 0.0,
        minimum_confidence: float = 0.0,
        allowed_switch_workflows: list[str] | None = None,
    ) -> tuple[str, float, dict[str, float], dict[str, float | str | bool]]:
        workflow_id, confidence, scores = self.predict_with_scores(
            question,
            budget_mode=budget_mode,
            candidate_workflows=candidate_workflows,
        )
        mode = normalize_budget_mode(budget_mode or self.default_budget_mode)
        baseline = str(baseline_workflow or "").upper()
        if not baseline or baseline not in scores:
            return workflow_id, confidence, scores, {
                "gate_applied": False,
                "baseline_workflow": baseline,
                "minimum_advantage": float(minimum_advantage),
                "predicted_advantage": 0.0,
            }

        baseline_score = float(scores[baseline])
        best_score = float(scores[workflow_id])
        predicted_advantage = best_score - baseline_score
        allowed_switches = {str(workflow).upper() for workflow in (allowed_switch_workflows or [])}
        blocked_by_workflow = bool(allowed_switches) and workflow_id not in allowed_switches
        blocked_by_confidence = float(confidence) < float(minimum_confidence)
        blocked_by_advantage = predicted_advantage < float(minimum_advantage)
        if workflow_id != baseline and (blocked_by_advantage or blocked_by_confidence or blocked_by_workflow):
            ranked_without_baseline = sorted(
                ((wf, score) for wf, score in scores.items() if wf != baseline),
                key=lambda item: item[1],
                reverse=True,
            )
            second_score = ranked_without_baseline[0][1] if ranked_without_baseline else baseline_score
            gated_confidence = 1.0 / (1.0 + math.exp(-4.0 * max(0.0, baseline_score - second_score)))
            return baseline, float(gated_confidence), scores, {
                "gate_applied": True,
                "baseline_workflow": baseline,
                "minimum_advantage": float(minimum_advantage),
                "minimum_confidence": float(minimum_confidence),
                "allowed_switch_workflows": sorted(allowed_switches),
                "predicted_advantage": round(float(predicted_advantage), 4),
                "blocked_by_confidence": blocked_by_confidence,
                "blocked_by_workflow": blocked_by_workflow,
                "blocked_by_advantage": blocked_by_advantage,
                "budget_mode": mode,
            }

        return workflow_id, confidence, scores, {
            "gate_applied": False,
            "baseline_workflow": baseline,
            "minimum_advantage": float(minimum_advantage),
            "minimum_confidence": float(minimum_confidence),
            "allowed_switch_workflows": sorted(allowed_switches),
            "predicted_advantage": round(float(predicted_advantage), 4) if baseline else 0.0,
            "budget_mode": mode,
        }

    def predict(
        self,
        question: str,
        budget_mode: str | None = None,
        candidate_workflows: list[str] | None = None,
    ) -> tuple[str, float]:
        workflow_id, confidence, _ = self.predict_with_scores(
            question,
            budget_mode=budget_mode,
            candidate_workflows=candidate_workflows,
        )
        return workflow_id, confidence

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "workflow_models": self.workflow_models,
                "random_state": self.random_state,
                "alpha": self.alpha,
                "default_budget_mode": self.default_budget_mode,
                "allowed_workflows": self.allowed_workflows,
                "preference_margin": self.preference_margin,
                "preferred_workflows": self.preferred_workflows,
                "aux_learned_pipeline": self.aux_learned_router.pipeline if self.aux_learned_router is not None else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        if isinstance(payload, dict) and "workflow_models" in payload:
            self.workflow_models = payload["workflow_models"]
            self.random_state = int(payload.get("random_state", self.random_state))
            self.alpha = float(payload.get("alpha", self.alpha))
            self.default_budget_mode = normalize_budget_mode(payload.get("default_budget_mode", self.default_budget_mode))
            self.allowed_workflows = [str(workflow).upper() for workflow in payload.get("allowed_workflows", self.allowed_workflows)]
            self.preference_margin = float(payload.get("preference_margin", self.preference_margin))
            self.preferred_workflows = [str(workflow).upper() for workflow in payload.get("preferred_workflows", self.preferred_workflows)]
            aux_pipeline = payload.get("aux_learned_pipeline")
            if aux_pipeline is not None:
                aux_router = LearnedRouter(random_state=self.random_state)
                aux_router.pipeline = aux_pipeline
                self.aux_learned_router = aux_router
            else:
                self.aux_learned_router = None
            self.probe_retriever = None
            self._context_cache = {}
            return
        raise ValueError("Unsupported bandit model format.")
