from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from map_rag_gym.core.schemas import Document
from map_rag_gym.evaluation.heuristics import (
    UTILITY_CONFIG,
    score_answer,
    score_grounding,
    score_query,
    score_retrieval,
    token_overlap,
)


def _extract_text(item: str | dict[str, Any] | None) -> str:
    if item is None:
        return ""
    if isinstance(item, dict):
        return str(item.get("query") or item.get("question") or item.get("text") or item.get("raw_text") or "").strip()
    return str(item).strip()


def _stringify_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        parts = []
        for key in ["query", "question", "text", "answer", "final_answer", "draft_reasoning", "reflected_query", "title", "doc_id"]:
            value = payload.get(key)
            if value:
                parts.append(str(value))
        if parts:
            return " | ".join(parts)
        return str(payload)
    if isinstance(payload, list):
        return " | ".join(_stringify_payload(item) for item in payload if item is not None)
    return str(payload)


def _numeric_values(values: Iterable[Any]) -> list[float]:
    nums: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            nums.append(float(value))
    return nums


def _mean_numeric(values: dict[str, Any]) -> float:
    nums = _numeric_values(values.values())
    return round(mean(nums), 4) if nums else 0.0


def _cost_penalty(cost: dict[str, Any], divisor: int = 1) -> float:
    divisor = max(1, divisor)
    token_cost = float(cost.get("tokens", 0.0)) / divisor / UTILITY_CONFIG["cost_normalizers"]["tokens"]
    retrieval_cost = float(cost.get("retrieval_calls", 0.0)) / divisor / UTILITY_CONFIG["cost_normalizers"]["retrieval_calls"]
    latency_cost = float(cost.get("latency_ms", 0.0)) / divisor / UTILITY_CONFIG["cost_normalizers"]["latency_ms"]
    penalty = (
        UTILITY_CONFIG["cost_weights"]["tokens"] * token_cost
        + UTILITY_CONFIG["cost_weights"]["retrieval_calls"] * retrieval_cost
        + UTILITY_CONFIG["cost_weights"]["latency_ms"] * latency_cost
    )
    return round(penalty, 4)


def _hydrate_docs(raw_docs: list[dict[str, Any]], corpus_lookup: dict[str, Document]) -> list[Document]:
    docs: list[Document] = []
    for item in raw_docs:
        doc_id = str(item.get("doc_id", ""))
        corpus_doc = corpus_lookup.get(doc_id)
        docs.append(
            Document(
                doc_id=doc_id,
                title=str(item.get("title") or (corpus_doc.title if corpus_doc else doc_id)),
                text=corpus_doc.text if corpus_doc else "",
                score=float(item.get("score", 0.0)),
                metadata=corpus_doc.metadata if corpus_doc else {},
            )
        )
    return docs


def _selected_docs(doc_ids: list[str], retrieved_docs: list[Document]) -> list[Document]:
    selected_ids = {str(doc_id) for doc_id in doc_ids}
    return [doc for doc in retrieved_docs if doc.doc_id in selected_ids]


def _history_text(history_modules: list[str]) -> str:
    return " -> ".join(history_modules)


def _make_example(
    *,
    run: dict[str, Any],
    step: dict[str, Any],
    module: str,
    action_index: int,
    action_text: str,
    question: str,
    gold_answer: str,
    history_modules: list[str],
    signal: float,
    docs_used: list[Document],
    selected: bool,
    num_actions_in_step: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step_cost = step.get("cost", {})
    local_penalty = _cost_penalty(step_cost, divisor=num_actions_in_step)
    local_reward = round(signal - local_penalty, 4)
    final_scores = run.get("final_scores", {})
    outcome_utility = float(final_scores.get("utility_total", 0.0))
    blended_reward = round(0.7 * local_reward + 0.3 * outcome_utility, 4)
    action_text = action_text.strip()

    example = {
        "example_id": f"{run['run_id']}:{step['step_id']}:{action_index}",
        "run_id": run["run_id"],
        "question_id": run.get("metadata", {}).get("question_id"),
        "workflow_id": run["workflow_id"],
        "module": module,
        "step_id": int(step["step_id"]),
        "action_index": action_index,
        "question": question,
        "gold_answer": gold_answer,
        "history_text": _history_text(history_modules),
        "history_modules": list(history_modules),
        "action_text": action_text,
        "action_len": len(action_text.split()),
        "num_actions_in_step": num_actions_in_step,
        "num_docs": len(docs_used),
        "selected": int(selected),
        "step_scores": dict(step.get("scores", {})),
        "step_cost": dict(step_cost),
        "signal": round(signal, 4),
        "step_score_mean": _mean_numeric(step.get("scores", {})),
        "local_cost_penalty": local_penalty,
        "local_reward": local_reward,
        "outcome_utility": round(outcome_utility, 4),
        "outcome_em": float(final_scores.get("em", 0.0)),
        "outcome_f1_proxy": float(final_scores.get("f1_proxy", 0.0)),
        "outcome_process_score": float(final_scores.get("process_score", 0.0)),
        "blended_reward": blended_reward,
        "binary_label": int(blended_reward > 0.0),
        "tokens": float(step_cost.get("tokens", 0.0)) / max(1, num_actions_in_step),
        "retrieval_calls": float(step_cost.get("retrieval_calls", 0.0)) / max(1, num_actions_in_step),
        "latency_ms": float(step_cost.get("latency_ms", 0.0)) / max(1, num_actions_in_step),
    }
    if extra:
        example.update(extra)
    return example


def build_process_dataset(
    runs: list[dict[str, Any]],
    corpus_lookup: dict[str, Document],
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    module_stats: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        question = str(run.get("question", ""))
        gold_answer = str(run.get("gold_answer", ""))
        history_modules: list[str] = []
        retrieved_docs: list[Document] = []
        selected_docs: list[Document] = []

        for step in run.get("steps", []):
            module = str(step.get("module", ""))
            output_data = step.get("output_data", {})
            action_rows: list[dict[str, Any]] = []

            if module == "RA":
                retrieved_docs = _hydrate_docs(output_data.get("docs", []), corpus_lookup)
                docs = list(retrieved_docs)
                action_rows.append(
                    _make_example(
                        run=run,
                        step=step,
                        module=module,
                        action_index=0,
                        action_text=" | ".join(f"{doc.title} ({doc.doc_id})" for doc in docs),
                        question=question,
                        gold_answer=gold_answer,
                        history_modules=history_modules,
                        signal=score_retrieval(question, docs, gold_answer),
                        docs_used=docs,
                        selected=True,
                        num_actions_in_step=1,
                        extra={"query_text": _extract_text(step.get("input_data", {}).get("query"))},
                    )
                )

            elif module == "DS":
                selected_docs = _selected_docs(output_data.get("selected_doc_ids", []), retrieved_docs)
                docs = list(selected_docs)
                action_rows.append(
                    _make_example(
                        run=run,
                        step=step,
                        module=module,
                        action_index=0,
                        action_text=" | ".join(f"{doc.title} ({doc.doc_id})" for doc in docs),
                        question=question,
                        gold_answer=gold_answer,
                        history_modules=history_modules,
                        signal=score_retrieval(question, docs, gold_answer),
                        docs_used=docs,
                        selected=True,
                        num_actions_in_step=1,
                    )
                )

            elif module in {"QR", "QDP", "QDS"}:
                field = "query_candidates" if module == "QR" else "sub_questions"
                candidates = [_extract_text(item) for item in output_data.get(field, [])]
                if not candidates:
                    candidates = [_stringify_payload(output_data)]
                scores = [score_query(question, candidate) for candidate in candidates]
                best_idx = max(range(len(candidates)), key=lambda idx: scores[idx]) if candidates else 0
                for idx, candidate in enumerate(candidates):
                    action_rows.append(
                        _make_example(
                            run=run,
                            step=step,
                            module=module,
                            action_index=idx,
                            action_text=candidate,
                            question=question,
                            gold_answer=gold_answer,
                            history_modules=history_modules,
                            signal=scores[idx],
                            docs_used=selected_docs or retrieved_docs,
                            selected=idx == best_idx,
                            num_actions_in_step=len(candidates),
                        )
                    )

            elif module == "DRAFT":
                draft = _extract_text(output_data.get("draft_reasoning"))
                action_rows.append(
                    _make_example(
                        run=run,
                        step=step,
                        module=module,
                        action_index=0,
                        action_text=draft,
                        question=question,
                        gold_answer=gold_answer,
                        history_modules=history_modules,
                        signal=token_overlap(question, draft),
                        docs_used=selected_docs or retrieved_docs,
                        selected=True,
                        num_actions_in_step=1,
                    )
                )

            elif module == "REFLECT":
                query = _extract_text(output_data.get("reflected_query"))
                action_rows.append(
                    _make_example(
                        run=run,
                        step=step,
                        module=module,
                        action_index=0,
                        action_text=query,
                        question=question,
                        gold_answer=gold_answer,
                        history_modules=history_modules,
                        signal=score_query(question, query),
                        docs_used=selected_docs or retrieved_docs,
                        selected=True,
                        num_actions_in_step=1,
                    )
                )

            elif module in {"AG", "AS"}:
                docs = selected_docs or retrieved_docs
                if module == "AS":
                    answers = [_extract_text(output_data.get("final_answer"))]
                else:
                    answers = [_extract_text(item) for item in output_data.get("answer_candidates", [])]
                answers = [answer for answer in answers if answer]
                if not answers:
                    answers = [_stringify_payload(output_data)]
                grounding_scores = [score_grounding(answer, docs) for answer in answers]
                best_idx = max(range(len(answers)), key=lambda idx: grounding_scores[idx]) if answers else 0
                for idx, answer in enumerate(answers):
                    answer_score = score_answer(answer, gold_answer)
                    signal = round(0.6 * answer_score["f1_proxy"] + 0.4 * grounding_scores[idx], 4)
                    action_rows.append(
                        _make_example(
                            run=run,
                            step=step,
                            module=module,
                            action_index=idx,
                            action_text=answer,
                            question=question,
                            gold_answer=gold_answer,
                            history_modules=history_modules,
                            signal=signal,
                            docs_used=docs,
                            selected=idx == best_idx,
                            num_actions_in_step=len(answers),
                            extra={
                                "answer_f1_proxy": answer_score["f1_proxy"],
                                "answer_em": answer_score["em"],
                                "answer_grounding": grounding_scores[idx],
                            },
                        )
                    )

            else:
                action_rows.append(
                    _make_example(
                        run=run,
                        step=step,
                        module=module,
                        action_index=0,
                        action_text=_stringify_payload(output_data),
                        question=question,
                        gold_answer=gold_answer,
                        history_modules=history_modules,
                        signal=_mean_numeric(step.get("scores", {})),
                        docs_used=selected_docs or retrieved_docs,
                        selected=True,
                        num_actions_in_step=1,
                    )
                )

            for row in action_rows:
                examples.append(row)
                module_stats[module].append(row["blended_reward"])

            history_modules.append(module)

    summary = {}
    for module, rewards in module_stats.items():
        summary[module] = {
            "count": len(rewards),
            "avg_blended_reward": round(mean(rewards), 4),
            "positive_rate": round(sum(1 for value in rewards if value > 0.0) / len(rewards), 4) if rewards else 0.0,
        }

    return {
        "examples": examples,
        "module_stats": summary,
    }
