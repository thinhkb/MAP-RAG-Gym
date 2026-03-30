from __future__ import annotations

from typing import Any, Dict, List

from map_rag_gym.core.schemas import Document, PipelineState
from map_rag_gym.executors.base import Executor
from map_rag_gym.llm.providers import BaseLLM, try_parse_json
from map_rag_gym.prompts.templates import (
    answer_prompt,
    draft_reasoning_prompt,
    parallel_decompose_prompt,
    query_rewrite_prompt,
    reflect_prompt,
    serial_decompose_prompt,
    summarize_prompt,
)
from map_rag_gym.retrieval.bm25 import LocalBM25Retriever


class QueryRewriter(Executor):
    name = "QR"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, state: PipelineState, question: str | None = None, n: int = 1, **_: Any):
        q = question or state.question
        responses = self.llm.generate(query_rewrite_prompt(q), n=n)
        payloads = [try_parse_json(r.text) for r in responses]
        candidates = [str(p.get("query") or p.get("raw_text") or q).strip() for p in payloads]
        step = self.step(state, {"question": q, "n": n}, {"query_candidates": candidates, "payloads": payloads}, {"tokens": sum(r.estimated_tokens for r in responses)})
        state.working_memory["query_candidates"] = candidates
        return step


class RetrieverAgent(Executor):
    name = "RA"

    def __init__(self, retriever: LocalBM25Retriever) -> None:
        self.retriever = retriever

    def run(self, state: PipelineState, query: str | None = None, top_k: int = 4, **_: Any):
        q = query or state.working_memory.get("selected_query") or state.question
        docs = self.retriever.search(q, top_k=top_k)
        state.working_memory["retrieved_docs"] = docs
        return self.step(
            state,
            {"query": q, "top_k": top_k},
            {"docs": [{"doc_id": d.doc_id, "title": d.title, "score": d.score} for d in docs]},
            {"retrieval_calls": 1},
        )


class DocumentSelector(Executor):
    name = "DS"

    def run(self, state: PipelineState, keep_k: int = 2, **_: Any):
        docs: List[Document] = state.working_memory.get("retrieved_docs", [])
        selected = docs[:keep_k]
        state.working_memory["selected_docs"] = selected
        return self.step(state, {"num_docs": len(docs), "keep_k": keep_k}, {"selected_doc_ids": [d.doc_id for d in selected]})


class ParallelDecomposer(Executor):
    name = "QDP"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, state: PipelineState, **_: Any):
        resp = self.llm.generate(parallel_decompose_prompt(state.question), n=1)[0]
        payload = try_parse_json(resp.text)
        parts = payload.get("sub_questions") or [state.question, f"supporting fact for {state.question}"]
        state.working_memory["sub_questions"] = parts
        return self.step(state, {"question": state.question}, {"sub_questions": parts, "payload": payload}, {"tokens": resp.estimated_tokens})


class SerialDecomposer(Executor):
    name = "QDS"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, state: PipelineState, **_: Any):
        resp = self.llm.generate(serial_decompose_prompt(state.question), n=1)[0]
        payload = try_parse_json(resp.text)
        parts = payload.get("sub_questions") or [state.question, f"Find bridge entity for {state.question}"]
        state.working_memory["sub_questions"] = parts
        return self.step(state, {"question": state.question}, {"sub_questions": parts, "payload": payload}, {"tokens": resp.estimated_tokens})


class DraftReasoner(Executor):
    name = "DRAFT"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, state: PipelineState, **_: Any):
        resp = self.llm.generate(draft_reasoning_prompt(state.question), n=1)[0]
        payload = try_parse_json(resp.text)
        draft = str(payload.get("draft_reasoning") or payload.get("raw_text") or state.question)
        state.working_memory["draft_reasoning"] = draft
        return self.step(state, {"question": state.question}, {"draft_reasoning": draft, "payload": payload}, {"tokens": resp.estimated_tokens})


class Reflector(Executor):
    name = "REFLECT"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, state: PipelineState, **_: Any):
        draft = state.working_memory.get("draft_reasoning", state.question)
        resp = self.llm.generate(reflect_prompt(state.question, draft), n=1)[0]
        payload = try_parse_json(resp.text)
        query = str(payload.get("query") or payload.get("raw_text") or state.question).strip()
        state.working_memory["selected_query"] = query
        return self.step(state, {"draft_reasoning": draft}, {"reflected_query": query, "payload": payload}, {"tokens": resp.estimated_tokens})


class AnswerGenerator(Executor):
    name = "AG"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, state: PipelineState, n: int = 1, question: str | None = None, docs: List[Document] | None = None, **_: Any):
        q = question or state.question
        used_docs = docs or state.working_memory.get("selected_docs") or state.working_memory.get("retrieved_docs") or []
        responses = self.llm.generate(answer_prompt(q, used_docs), n=n)
        payloads = [try_parse_json(r.text) for r in responses]
        candidates = [str(p.get("answer") or p.get("raw_text") or "").strip() for p in payloads]
        state.working_memory["answer_candidates"] = candidates
        return self.step(state, {"question": q, "num_docs": len(used_docs), "n": n}, {"answer_candidates": candidates, "payloads": payloads}, {"tokens": sum(r.estimated_tokens for r in responses)})


class AnswerSummarizer(Executor):
    name = "AS"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, state: PipelineState, **_: Any):
        subs = state.working_memory.get("sub_answers") or state.working_memory.get("answer_candidates") or []
        resp = self.llm.generate(summarize_prompt(state.question, subs), n=1)[0]
        payload = try_parse_json(resp.text)
        final = str(payload.get("final_answer") or payload.get("raw_text") or (subs[0] if subs else ""))
        state.working_memory["final_answer"] = final.strip()
        return self.step(state, {"items": subs}, {"final_answer": final.strip(), "payload": payload}, {"tokens": resp.estimated_tokens})
