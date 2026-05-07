from __future__ import annotations

from map_rag_gym.core.features import extract_question_features
from map_rag_gym.core.schemas import PlannerDecision
from map_rag_gym.router.budget import budget_fallback, is_workflow_allowed, normalize_budget_mode


class RuleBasedRouter:
    def decide(self, question: str, budget_mode: str = "medium") -> PlannerDecision:
        mode = normalize_budget_mode(budget_mode)
        feat = extract_question_features(question)
        if feat.comparative_flag and feat.estimated_hops >= 2:
            decision = PlannerDecision("W4", 0.82, "comparative or independent multi-hop")
        elif " after " in question.lower() or "before" in question.lower():
            decision = PlannerDecision("W5", 0.78, "serial dependency question")
        elif feat.ambiguity_flag:
            decision = PlannerDecision("W2", 0.70, "ambiguous wording, rewrite first")
        elif feat.token_len <= 8 and mode == "low":
            decision = PlannerDecision("W1", 0.72, "short low-cost factual path")
        elif feat.estimated_hops == 1:
            decision = PlannerDecision("W3", 0.68, "retrieve-select-answer")
        else:
            decision = PlannerDecision("W6", 0.55, "fallback reflective retrieval")

        if is_workflow_allowed(decision.workflow_id, mode):
            return PlannerDecision(decision.workflow_id, decision.confidence, f"{decision.reason} | budget={mode}")

        fallback = budget_fallback(question, mode)
        return PlannerDecision(
            fallback.workflow_id,
            fallback.confidence,
            f"{decision.reason} | budget={mode} -> {fallback.workflow_id}",
        )
