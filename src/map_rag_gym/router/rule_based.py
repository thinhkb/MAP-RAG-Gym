from __future__ import annotations

from map_rag_gym.core.features import extract_question_features
from map_rag_gym.core.schemas import PlannerDecision


class RuleBasedRouter:
    def decide(self, question: str, budget_mode: str = "medium") -> PlannerDecision:
        feat = extract_question_features(question)
        if feat.comparative_flag and feat.estimated_hops >= 2:
            return PlannerDecision("W4", 0.82, "comparative or independent multi-hop")
        if " after " in question.lower() or "before" in question.lower():
            return PlannerDecision("W5", 0.78, "serial dependency question")
        if feat.ambiguity_flag:
            return PlannerDecision("W2", 0.70, "ambiguous wording, rewrite first")
        if feat.token_len <= 8 and budget_mode == "low":
            return PlannerDecision("W1", 0.72, "short low-cost factual path")
        if feat.estimated_hops == 1:
            return PlannerDecision("W3", 0.68, "retrieve-select-answer")
        return PlannerDecision("W6", 0.55, "fallback reflective retrieval")
