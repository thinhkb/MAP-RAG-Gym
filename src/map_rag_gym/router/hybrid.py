from __future__ import annotations

from map_rag_gym.core.schemas import PlannerDecision
from map_rag_gym.router.budget import is_workflow_allowed, normalize_budget_mode
from map_rag_gym.router.learned import LearnedRouter
from map_rag_gym.router.rule_based import RuleBasedRouter


class HybridRouter:
    def __init__(
        self,
        learned_router: LearnedRouter,
        rule_router: RuleBasedRouter | None = None,
        min_confidence: float = 0.5,
        low_cost_workflow_confidence: float = 0.55,
        low_cost_workflows: list[str] | None = None,
    ) -> None:
        self.learned_router = learned_router
        self.rule_router = rule_router or RuleBasedRouter()
        self.min_confidence = min_confidence
        self.low_cost_workflow_confidence = low_cost_workflow_confidence
        self.low_cost_workflows = tuple(low_cost_workflows or ["W1"])

    def decide(self, question: str, budget_mode: str = "medium") -> PlannerDecision:
        mode = normalize_budget_mode(budget_mode)
        workflow_id, confidence = self.learned_router.predict(question, budget_mode=mode)
        threshold = self.low_cost_workflow_confidence if workflow_id in self.low_cost_workflows else self.min_confidence
        if confidence >= threshold and is_workflow_allowed(workflow_id, mode):
            return PlannerDecision(
                workflow_id=workflow_id,
                confidence=confidence,
                reason=f"hybrid:learned(conf={confidence:.4f}, threshold={threshold:.4f}, budget={mode})",
            )

        fallback = self.rule_router.decide(question, budget_mode=mode)
        if confidence >= threshold and not is_workflow_allowed(workflow_id, mode):
            return PlannerDecision(
                workflow_id=fallback.workflow_id,
                confidence=fallback.confidence,
                reason=(
                    f"hybrid:budget_fallback(rule={fallback.workflow_id}, learned={workflow_id}, "
                    f"conf={confidence:.4f}, budget={mode})"
                ),
            )
        return PlannerDecision(
            workflow_id=fallback.workflow_id,
            confidence=fallback.confidence,
            reason=(
                f"hybrid:fallback(rule={fallback.workflow_id}, learned={workflow_id}, "
                f"conf={confidence:.4f}, threshold={threshold:.4f}, budget={mode})"
            ),
        )
