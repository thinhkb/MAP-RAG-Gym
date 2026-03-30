from __future__ import annotations

from typing import Dict, List

from map_rag_gym.core.schemas import Document, PipelineState, StepRecord


STOPWORDS = {"the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "was", "were", "by"}


def _normalize(text: str) -> List[str]:
    return [t for t in text.lower().replace("?", "").replace(",", " ").replace(".", " ").split() if t and t not in STOPWORDS]


def token_overlap(a: str, b: str) -> float:
    sa = set(_normalize(a))
    sb = set(_normalize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score_query(question: str, query: str) -> float:
    q_overlap = token_overlap(question, query)
    penalty = 0.0 if len(query.split()) >= 2 else 0.2
    return round(max(0.0, q_overlap - penalty), 4)


def score_retrieval(question: str, docs: List[Document], gold_answer: str = "") -> float:
    if not docs:
        return 0.0
    joined = " ".join((d.title + " " + d.text).lower() for d in docs)
    ans_hit = float(gold_answer.lower() in joined) if gold_answer else 0.0
    q_hit = token_overlap(question, joined)
    diversity = min(1.0, len({d.doc_id for d in docs}) / max(1, len(docs)))
    return round(0.5 * ans_hit + 0.3 * q_hit + 0.2 * diversity, 4)


def score_grounding(answer: str, docs: List[Document]) -> float:
    joined = " ".join((d.title + " " + d.text).lower() for d in docs)
    return round(token_overlap(answer, joined), 4)


def score_answer(answer: str, gold_answer: str) -> Dict[str, float]:
    exact = float(answer.strip().lower() == gold_answer.strip().lower())
    overlap = token_overlap(answer, gold_answer)
    return {"em": exact, "f1_proxy": round(overlap, 4)}


def compute_utility(final_scores: Dict[str, float], total_cost: Dict[str, float], process_score: float) -> float:
    token_cost = total_cost.get("tokens", 0.0) / 2000.0
    retrieval_cost = total_cost.get("retrieval_calls", 0.0) / 4.0
    latency_cost = total_cost.get("latency_ms", 0.0) / 10000.0
    utility = final_scores.get("f1_proxy", 0.0) + 0.35 * process_score - 0.08 * token_cost - 0.1 * retrieval_cost - 0.05 * latency_cost
    return round(utility, 4)


def evaluate_step(state: PipelineState, step: StepRecord) -> StepRecord:
    if step.module == "QR":
        candidates = step.output_data.get("query_candidates", [])
        step.scores["query_quality"] = max([score_query(state.question, c) for c in candidates] or [0.0])
    elif step.module == "RA":
        docs = state.working_memory.get("retrieved_docs", [])
        step.scores["retrieval_utility"] = score_retrieval(state.question, docs, state.gold_answer)
    elif step.module == "DS":
        docs = state.working_memory.get("selected_docs", [])
        step.scores["selection_precision"] = score_retrieval(state.question, docs, state.gold_answer)
    elif step.module in {"AG", "AS"}:
        candidates = step.output_data.get("answer_candidates") or [step.output_data.get("final_answer", "")]
        docs = state.working_memory.get("selected_docs") or state.working_memory.get("retrieved_docs") or []
        best = max(candidates, key=lambda x: score_grounding(x, docs)) if candidates else ""
        step.scores["grounding"] = score_grounding(best, docs)
        if state.gold_answer:
            step.scores.update(score_answer(best, state.gold_answer))
    elif step.module in {"QDS", "QDP"}:
        subs = step.output_data.get("sub_questions", [])
        step.scores["decomposition_quality"] = round(sum(score_query(state.question, s) for s in subs) / max(1, len(subs)), 4)
    elif step.module == "REFLECT":
        q = step.output_data.get("reflected_query", "")
        step.scores["reflection_quality"] = score_query(state.question, q)
    elif step.module == "DRAFT":
        draft = step.output_data.get("draft_reasoning", "")
        step.scores["draft_relevance"] = token_overlap(state.question, draft)
    return step


def aggregate_process_score(steps: List[StepRecord]) -> float:
    vals = []
    for step in steps:
        vals.extend(v for v in step.scores.values() if isinstance(v, (int, float)))
    return round(sum(vals) / len(vals), 4) if vals else 0.0
