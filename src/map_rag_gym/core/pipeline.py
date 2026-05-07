from __future__ import annotations

import time
from typing import Dict, List

from map_rag_gym.critic.model import ProcessCritic
from map_rag_gym.core.schemas import Document, PipelineState, RunResult
from map_rag_gym.core.workflows import WORKFLOWS
from map_rag_gym.evaluation.heuristics import (
    aggregate_process_score,
    compute_budgeted_utility,
    compute_utility,
    evaluate_step,
    get_utility_profile,
    score_answer,
    score_grounding,
    score_query,
)
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
from map_rag_gym.retrieval.hybrid import LocalHybridRetriever
from map_rag_gym.retrieval.policy import normalize_retriever_name
from map_rag_gym.retrieval.tfidf import LocalTfidfRetriever


class MAPRAGGym:
    def __init__(
        self,
        corpus_path: str,
        llm_provider: str = "dummy",
        llm_model: str | None = None,
        retriever_type: str = "bm25",
        retriever_bm25_weight: float = 0.5,
        workflow_retriever_overrides: dict[str, str] | None = None,
        critic_model_path: str | None = None,
        critic_modules: list[str] | None = None,
        critic_model_overrides: dict[str, str] | None = None,
    ) -> None:
        llm = build_llm(llm_provider, llm_model)
        self.corpus_path = corpus_path
        self.llm_provider = getattr(llm, "provider", llm_provider)
        self.llm_model = getattr(llm, "model", llm_model)
        self.retriever_type = normalize_retriever_name(retriever_type)
        self.retriever_bm25_weight = retriever_bm25_weight
        self.workflow_retriever_overrides = {
            str(workflow_id).upper(): normalize_retriever_name(retriever_name)
            for workflow_id, retriever_name in (workflow_retriever_overrides or {}).items()
        }
        retriever_names = {self.retriever_type, *self.workflow_retriever_overrides.values()}
        self.retrievers = {name: self._build_retriever(name) for name in sorted(retriever_names)}
        self.critic_model_path = critic_model_path
        self.critic_model_overrides = {
            str(module).upper(): path
            for module, path in (critic_model_overrides or {}).items()
        }
        self.critic_modules = tuple(sorted({str(module).upper() for module in (critic_modules or [])} | set(self.critic_model_overrides)))
        self.critic: ProcessCritic | None = None
        self.critics_by_module: dict[str, ProcessCritic] = {}
        if critic_model_path:
            self.critic = ProcessCritic()
            self.critic.load(critic_model_path)
        for module_name, model_path in self.critic_model_overrides.items():
            critic = ProcessCritic()
            critic.load(model_path)
            self.critics_by_module[module_name] = critic
        self.executors = {
            "QR": QueryRewriter(llm),
            "RA": RetrieverAgent(self.retrievers[self.retriever_type], retriever_name=self.retriever_type),
            "DS": DocumentSelector(),
            "QDP": ParallelDecomposer(llm),
            "QDS": SerialDecomposer(llm),
            "DRAFT": DraftReasoner(llm),
            "REFLECT": Reflector(llm),
            "AG": AnswerGenerator(llm),
            "AS": AnswerSummarizer(llm),
        }

    def _build_retriever(self, retriever_type: str):
        if retriever_type == "tfidf":
            return LocalTfidfRetriever(self.corpus_path)
        if retriever_type == "hybrid":
            return LocalHybridRetriever(self.corpus_path, bm25_weight=self.retriever_bm25_weight)
        return LocalBM25Retriever(self.corpus_path)

    def _resolve_retriever_name(self, workflow_id: str | None) -> str:
        workflow_key = str(workflow_id or "").upper()
        return self.workflow_retriever_overrides.get(workflow_key, self.retriever_type)

    def _get_critic_for_module(self, module: str) -> ProcessCritic | None:
        module_key = str(module).upper()
        if module_key in self.critics_by_module:
            return self.critics_by_module[module_key]
        if self.critic and module_key in self.critic_modules:
            return self.critic
        return None

    def _build_critic_rows(self, state: PipelineState, step, module: str, candidates: list[str]) -> list[dict]:
        docs = state.working_memory.get("selected_docs") or state.working_memory.get("retrieved_docs") or []
        history_modules = [item.module for item in state.history[:-1]]
        num_actions = max(1, len(candidates))
        tokens = float(step.cost.get("tokens", 0.0)) / num_actions
        retrieval_calls = float(step.cost.get("retrieval_calls", 0.0)) / num_actions
        latency_ms = float(step.cost.get("latency_ms", 0.0)) / num_actions
        query_text = str(step.input_data.get("query") or state.working_memory.get("selected_query") or state.question)
        budget_mode = str(state.working_memory.get("budget_mode") or "medium")
        rows = []
        for candidate in candidates:
            action_text = str(candidate).strip()
            rows.append({
                "question": state.question,
                "query_text": query_text,
                "action_text": action_text,
                "doc_title": "",
                "history_text": " -> ".join(history_modules),
                "module": module,
                "workflow_id": state.workflow_id,
                "retriever_type": "",
                "budget_mode": budget_mode,
                "step_id": int(step.step_id),
                "action_len": len(action_text.split()),
                "num_actions_in_step": num_actions,
                "num_docs": len(docs),
                "selected": 0,
                "tokens": tokens,
                "retrieval_calls": retrieval_calls,
                "latency_ms": latency_ms,
                "doc_rank": 0,
                "doc_score": 0.0,
            })
        return rows

    def _build_doc_critic_rows(
        self,
        state: PipelineState,
        *,
        module: str,
        docs: list[Document],
        query_text: str,
        step_id: int,
        cost: dict | None = None,
    ) -> list[dict]:
        history_modules = [item.module for item in state.history]
        num_actions = max(1, len(docs))
        step_cost = cost or {}
        tokens = float(step_cost.get("tokens", 0.0)) / num_actions
        retrieval_calls = float(step_cost.get("retrieval_calls", 0.0)) / num_actions
        latency_ms = float(step_cost.get("latency_ms", 0.0)) / num_actions
        retriever_type = self._resolve_retriever_name(state.workflow_id)
        budget_mode = str(state.working_memory.get("budget_mode") or "medium")
        rows = []
        for idx, doc in enumerate(docs):
            snippet = " ".join(doc.text.split())[:320].strip()
            action_text = " | ".join(part for part in [doc.title.strip(), snippet] if part)
            rows.append({
                "question": state.question,
                "query_text": str(query_text or state.question),
                "action_text": action_text,
                "doc_title": doc.title,
                "history_text": " -> ".join(history_modules),
                "module": module,
                "workflow_id": state.workflow_id,
                "retriever_type": retriever_type,
                "budget_mode": budget_mode,
                "step_id": int(step_id),
                "action_len": len(action_text.split()),
                "num_actions_in_step": num_actions,
                "num_docs": 1,
                "selected": 0,
                "tokens": tokens,
                "retrieval_calls": retrieval_calls,
                "latency_ms": latency_ms,
                "doc_rank": idx + 1,
                "doc_score": float(doc.score),
            })
        return rows

    def _rerank_docs_with_critic(
        self,
        state: PipelineState,
        *,
        module: str,
        docs: list[Document],
        query_text: str,
        step_id: int,
        cost: dict | None = None,
    ) -> tuple[list[Document], list[float]]:
        critic = self._get_critic_for_module(module)
        if not (critic and len(docs) > 1):
            return docs, []
        rows = self._build_doc_critic_rows(
            state,
            module=module,
            docs=docs,
            query_text=query_text,
            step_id=step_id,
            cost=cost,
        )
        preds = critic.predict(rows)
        ranked = sorted(zip(docs, preds), key=lambda item: item[1], reverse=True)
        reranked_docs = [doc for doc, _ in ranked]
        reranked_scores = [float(score) for _, score in ranked]
        return reranked_docs, reranked_scores

    def _pick_best_query(self, state: PipelineState, step) -> None:
        candidates = state.working_memory.get("query_candidates", [])
        if candidates:
            critic = self._get_critic_for_module("QR")
            if critic and len(candidates) > 1:
                preds = critic.predict(self._build_critic_rows(state, step, "QR", candidates))
                best_idx = max(range(len(candidates)), key=lambda idx: preds[idx])
                selected = candidates[best_idx]
                step.output_data["critic_scores"] = [round(value, 4) for value in preds]
                step.output_data["critic_selected_idx"] = best_idx
                step.output_data["critic_selected_text"] = selected
                step.notes = f"{step.notes} | critic_reranked" if step.notes else "critic_reranked"
            else:
                scored = sorted(candidates, key=lambda x: len(set(x.lower().split()) & set(state.question.lower().split())), reverse=True)
                selected = scored[0]
            state.working_memory["selected_query"] = selected
            step.scores["query_quality"] = score_query(state.question, selected)

    def _pick_best_answer(self, state: PipelineState, step) -> None:
        candidates = state.working_memory.get("answer_candidates", [])
        docs = state.working_memory.get("selected_docs") or state.working_memory.get("retrieved_docs") or []
        if candidates:
            critic = self._get_critic_for_module("AG")
            if critic and len(candidates) > 1:
                preds = critic.predict(self._build_critic_rows(state, step, "AG", candidates))
                best_idx = max(range(len(candidates)), key=lambda idx: preds[idx])
                best = candidates[best_idx]
                step.output_data["critic_scores"] = [round(value, 4) for value in preds]
                step.output_data["critic_selected_idx"] = best_idx
                step.output_data["critic_selected_text"] = best
                step.notes = f"{step.notes} | critic_reranked" if step.notes else "critic_reranked"
            else:
                best = max(candidates, key=lambda x: score_grounding(x, docs))
            state.working_memory["final_answer"] = best
            step.scores["grounding"] = score_grounding(best, docs)
            if state.gold_answer:
                step.scores.update(score_answer(best, state.gold_answer))

    def _run_step(self, state: PipelineState, module_name: str, total_cost: Dict[str, float], **kwargs):
        t0 = time.time()
        step_kwargs = dict(kwargs)
        pending_doc_rerank: dict | None = None
        if module_name == "RA":
            retriever_name = step_kwargs.pop("retriever_name", self._resolve_retriever_name(state.workflow_id))
            step_kwargs["retriever"] = self.retrievers[retriever_name]
            step_kwargs["retriever_name"] = retriever_name
        elif module_name == "DS":
            docs = list(state.working_memory.get("retrieved_docs", []))
            query_text = str(state.working_memory.get("selected_query") or state.question)
            reranked_docs, reranked_scores = self._rerank_docs_with_critic(
                state,
                module="DS",
                docs=docs,
                query_text=query_text,
                step_id=state.next_step_id(),
            )
            if reranked_scores:
                state.working_memory["retrieved_docs"] = reranked_docs
                pending_doc_rerank = {
                    "critic_doc_scores": [
                        {"doc_id": doc.doc_id, "score": round(score, 4)}
                        for doc, score in zip(reranked_docs, reranked_scores)
                    ],
                    "critic_reranked_doc_ids": [doc.doc_id for doc in reranked_docs],
                }
        step = self.executors[module_name].run(state, **step_kwargs)
        if module_name == "RA":
            docs = list(state.working_memory.get("retrieved_docs", []))
            query_text = str(step.input_data.get("query") or state.working_memory.get("selected_query") or state.question)
            reranked_docs, reranked_scores = self._rerank_docs_with_critic(
                state,
                module="RA",
                docs=docs,
                query_text=query_text,
                step_id=step.step_id,
                cost=step.cost,
            )
            if reranked_scores:
                state.working_memory["retrieved_docs"] = reranked_docs
                step.output_data["docs"] = [{"doc_id": d.doc_id, "title": d.title, "score": d.score} for d in reranked_docs]
                step.output_data["critic_doc_scores"] = [
                    {"doc_id": doc.doc_id, "score": round(score, 4)}
                    for doc, score in zip(reranked_docs, reranked_scores)
                ]
                step.output_data["critic_reranked_doc_ids"] = [doc.doc_id for doc in reranked_docs]
                step.notes = f"{step.notes} | critic_reranked" if step.notes else "critic_reranked"
        step.cost["latency_ms"] = round((time.time() - t0) * 1000.0, 2)
        if pending_doc_rerank:
            step.output_data.update(pending_doc_rerank)
            step.notes = f"{step.notes} | critic_reranked" if step.notes else "critic_reranked"
        state.add_step(evaluate_step(state, step))
        if module_name == "QR":
            self._pick_best_query(state, step)
        if module_name == "AG":
            self._pick_best_answer(state, step)
        for k, v in step.cost.items():
            total_cost[k] = total_cost.get(k, 0.0) + float(v)
        return step

    def run(
        self,
        question: str,
        gold_answer: str,
        workflow_id: str,
        planner_reason: str = "manual",
        n_candidates: int = 3,
        budget_mode: str = "medium",
        module_candidate_counts: dict[str, int] | None = None,
    ) -> RunResult:
        state = PipelineState(question=question, gold_answer=gold_answer, workflow_id=workflow_id)
        state.working_memory["budget_mode"] = budget_mode
        steps = WORKFLOWS[workflow_id]
        total_cost: Dict[str, float] = {"tokens": 0.0, "retrieval_calls": 0.0, "latency_ms": 0.0}
        module_candidate_counts = {str(key).upper(): int(value) for key, value in (module_candidate_counts or {}).items()}

        for module_name in steps:
            if module_name in {"QR", "AG"}:
                step_n = module_candidate_counts.get(module_name, n_candidates)
                self._run_step(state, module_name, total_cost, n=step_n)
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
        final_scores["utility_default"] = compute_utility(final_scores, total_cost, process_score)
        final_scores["utility_total"] = compute_budgeted_utility(
            final_scores=final_scores,
            total_cost=total_cost,
            process_score=process_score,
            budget_mode=budget_mode,
        )
        workflow_retriever_type = self._resolve_retriever_name(workflow_id)
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
            metadata={
                "n_candidates": n_candidates,
                "module_candidate_counts": dict(module_candidate_counts),
                "elapsed_ms": round((time.time() - state.start_time) * 1000.0, 2),
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "budget_mode": budget_mode,
                "utility_profile": get_utility_profile(budget_mode),
                "corpus_path": self.corpus_path,
                "retriever_type": self.retriever_type,
                "retriever_bm25_weight": self.retriever_bm25_weight,
                "workflow_retriever_type": workflow_retriever_type,
                "workflow_retriever_overrides": dict(self.workflow_retriever_overrides),
                "available_retrievers": sorted(self.retrievers),
                "workflow_steps": steps,
                "critic_model_path": self.critic_model_path,
                "critic_model_overrides": dict(self.critic_model_overrides),
                "critic_modules": list(self.critic_modules),
            },
        )
