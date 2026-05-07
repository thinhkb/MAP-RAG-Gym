from __future__ import annotations

from map_rag_gym.core.features import extract_question_features
from map_rag_gym.core.schemas import PlannerDecision

ALLOWED_WORKFLOWS_BY_BUDGET = {
    "low": {"W1", "W3"},
    "medium": {"W1", "W2", "W3", "W6"},
    "high": {"W1", "W2", "W3", "W4", "W5", "W6"},
}


def normalize_budget_mode(budget_mode: str | None) -> str:
    mode = str(budget_mode or "medium").strip().lower()
    aliases = {
        "cheap": "low",
        "budget": "low",
        "balanced": "medium",
        "default": "medium",
        "quality": "high",
        "max_quality": "high",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in ALLOWED_WORKFLOWS_BY_BUDGET:
        raise ValueError(f"Unknown budget mode '{budget_mode}'. Expected one of {sorted(ALLOWED_WORKFLOWS_BY_BUDGET)}.")
    return normalized


def is_workflow_allowed(workflow_id: str, budget_mode: str | None) -> bool:
    mode = normalize_budget_mode(budget_mode)
    return str(workflow_id).upper() in ALLOWED_WORKFLOWS_BY_BUDGET[mode]


def budget_fallback(question: str, budget_mode: str | None) -> PlannerDecision:
    mode = normalize_budget_mode(budget_mode)
    feat = extract_question_features(question)
    if mode == "low":
        if feat.token_len <= 8 and feat.estimated_hops <= 1 and not feat.ambiguity_flag:
            return PlannerDecision("W1", 0.62, "budget:low direct path")
        return PlannerDecision("W3", 0.64, "budget:low retrieve-select path")
    if mode == "medium":
        if feat.ambiguity_flag:
            return PlannerDecision("W2", 0.66, "budget:medium rewrite first")
        if feat.estimated_hops <= 1:
            return PlannerDecision("W3", 0.68, "budget:medium retrieve-select path")
        return PlannerDecision("W6", 0.6, "budget:medium reflective retrieval")
    if feat.comparative_flag and feat.estimated_hops >= 2:
        return PlannerDecision("W4", 0.74, "budget:high comparative multi-hop")
    if " after " in question.lower() or "before" in question.lower():
        return PlannerDecision("W5", 0.7, "budget:high serial dependency")
    if feat.ambiguity_flag:
        return PlannerDecision("W2", 0.68, "budget:high rewrite first")
    if feat.estimated_hops <= 1:
        return PlannerDecision("W3", 0.7, "budget:high retrieve-select path")
    return PlannerDecision("W6", 0.62, "budget:high reflective retrieval")
