from __future__ import annotations

import time
from typing import Dict, List

from map_rag_gym.core.schemas import PipelineState, RunResult
from map_rag_gym.core.workflows import WORKFLOWS
from map_rag_gym.evaluation.heuristics import aggregate_process_score, compute_utility, evaluate_step, score_answer, score_grounding
from map_rag_gym.executors.modules import (
    AnswerGenerator,
    AnswerSummarizer,
    DocumentSelector,
    DraftReasoner,
    ParallelDecomposer,
    QueryRewriter,
    Reflector,
    RetrieverAgent,
    SerialDecomposer,
)
from map_rag_gym.llm.providers import build_llm
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever


class MAPRAGGym:
    def __init__(self, corpus_path: str, llm_provider: str = "dummy", llm_model: str | None = None) -> None:
        llm = build_llm(llm_provider, llm_model)
        retriever = LocalBM25Retriever(corpus_path)
        self.executors = {
            "QR": QueryRewriter(llm),
            "RA": RetrieverAgent(retriever),
            "DS": DocumentSelector(),
            "QDP": ParallelDecomposer(llm),
            "QDS": SerialDecomposer(llm),
            "DRAFT": DraftReasoner(llm),
            "REFLECT": Reflector(llm),
            "AG": AnswerGenerator(llm),
            "AS": AnswerSummarizer(llm),
        }

    def _pick_best_query(self, state: PipelineState) -> None:
        candidates = state.working_memory.get("query_candidates", [])
        if candidates:
            scored = sorted(candidates, key=lambda x: len(set(x.lower().split()) & set(state.question.lower().split())), reverse=True)
            state.working_memory["selected_query"] = scored[0]

    def _pick_best_answer(self, state: PipelineState) -> None:
        candidates = state.working_memory.get("answer_candidates", [])
        docs = state.working_memory.get("selected_docs") or state.working_memory.get("retrieved_docs") or []
        if candidates:
            best = max(candidates, key=lambda x: score_grounding(x, docs))
            state.working_memory["final_answer"] = best

    def _run_step(self, state: PipelineState, module_name: str, total_cost: Dict[str, float], **kwargs):
        t0 = time.time()
        step = self.executors[module_name].run(state, **kwargs)
        step.cost["latency_ms"] = round((time.time() - t0) * 1000.0, 2)
        state.add_step(evaluate_step(state, step))
        if module_name == "QR":
            self._pick_best_query(state)
        if module_name == "AG":
            self._pick_best_answer(state)
        for k, v in step.cost.items():
            total_cost[k] = total_cost.get(k, 0.0) + float(v)
        return step

    def run(self, question: str, gold_answer: str, workflow_id: str, planner_reason: str = "manual", n_candidates: int = 3) -> RunResult:
        state = PipelineState(question=question, gold_answer=gold_answer, workflow_id=workflow_id)
        steps = WORKFLOWS[workflow_id]
        total_cost: Dict[str, float] = {"tokens": 0.0, "retrieval_calls": 0.0, "latency_ms": 0.0}

        for module_name in steps:
            if module_name in {"QR", "AG"}:
                self._run_step(state, module_name, total_cost, n=n_candidates)
            else:
                self._run_step(state, module_name, total_cost)

            if module_name in {"QDP", "QDS"}:
                sub_answers = []
                for sub_q in state.working_memory.get("sub_questions", []):
                    self._run_step(state, "RA", total_cost, query=sub_q)
                    sub_ag = self._run_step(state, "AG", total_cost, n=1, question=sub_q)
                    sub_answers.append((sub_ag.output_data.get("answer_candidates") or [""])[0])
                state.working_memory["sub_answers"] = sub_answers

        final_answer = state.working_memory.get("final_answer") or (state.working_memory.get("answer_candidates") or [""])[0]
        final_scores = score_answer(final_answer, gold_answer)
        process_score = aggregate_process_score(state.history)
        final_scores["process_score"] = process_score
        final_scores["utility_total"] = compute_utility(final_scores, total_cost, process_score)
        return RunResult(
            run_id=state.run_id,
            question=question,
            gold_answer=gold_answer,
            workflow_id=workflow_id,
            planner_reason=planner_reason,
            final_answer=final_answer,
            final_scores=final_scores,
            total_cost=total_cost,
            steps=state.history,
            metadata={"n_candidates": n_candidates, "elapsed_ms": round((time.time() - state.start_time) * 1000.0, 2)},
        )
