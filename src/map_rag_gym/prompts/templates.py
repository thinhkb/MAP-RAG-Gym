from __future__ import annotations

import json
from typing import List

from map_rag_gym.core.schemas import Document


def _json_instruction(schema: dict) -> str:
    return (
        "Return only valid JSON matching this schema exactly. "
        f"Schema example: {json.dumps(schema, ensure_ascii=False)}"
    )


def query_rewrite_prompt(question: str) -> str:
    schema = {"query": "short search query", "rationale": "brief reason"}
    return (
        "You are the Query Rewriter module in an adaptive RAG system.\n"
        "Task: rewrite the user question into a concise search query for retrieval.\n"
        "Rules:\n"
        "- preserve the original intent\n"
        "- keep named entities and critical constraints\n"
        "- remove filler words\n"
        "- do not answer the question\n"
        f"{_json_instruction(schema)}\n"
        f"Question: {question}"
    )


def parallel_decompose_prompt(question: str) -> str:
    schema = {"sub_questions": ["sub question 1", "sub question 2"], "reason": "brief reason"}
    return (
        "You are the Parallel Decomposition module in an adaptive RAG system.\n"
        "Break the question into independent sub-questions that can be answered separately.\n"
        "Rules:\n"
        "- use 2 to 4 sub-questions only if helpful\n"
        "- each sub-question must preserve key entities\n"
        "- sub-questions should be independent, not sequential\n"
        f"{_json_instruction(schema)}\n"
        f"Question: {question}"
    )


def serial_decompose_prompt(question: str) -> str:
    schema = {"sub_questions": ["step 1 question", "step 2 question"], "reason": "brief reason"}
    return (
        "You are the Serial Decomposition module in an adaptive RAG system.\n"
        "Break the question into sequential sub-questions where later steps depend on earlier answers.\n"
        "Rules:\n"
        "- produce 2 to 4 sequential steps only if needed\n"
        "- step 1 should identify missing entity or bridge information\n"
        "- later steps should depend on previous results\n"
        f"{_json_instruction(schema)}\n"
        f"Question: {question}"
    )


def draft_reasoning_prompt(question: str) -> str:
    schema = {"draft_reasoning": "short reasoning draft", "predicted_answer": "best guess or unknown"}
    return (
        "You are the Draft Reasoner in a reflective RAG workflow.\n"
        "Write a short reasoning draft and best current guess before retrieval.\n"
        "Rules:\n"
        "- keep it under 80 words\n"
        "- explicitly mention what is still missing\n"
        f"{_json_instruction(schema)}\n"
        f"Question: {question}"
    )


def reflect_prompt(question: str, draft: str) -> str:
    schema = {"missing_information": ["gap 1"], "query": "one retrieval query"}
    return (
        "You are the Reflection module in a reflective RAG workflow.\n"
        "Inspect the draft reasoning, identify missing evidence, and produce one search query.\n"
        "Rules:\n"
        "- focus on unsupported claims\n"
        "- the query should target the missing evidence\n"
        "- do not answer the question\n"
        f"{_json_instruction(schema)}\n"
        f"Question: {question}\n"
        f"Draft reasoning: {draft}"
    )


def answer_prompt(question: str, docs: List[Document]) -> str:
    context = "\n\n".join(f"[{i+1}] {d.title}: {d.text}" for i, d in enumerate(docs))
    schema = {"answer": "final answer", "evidence_ids": [1], "confidence": 0.5}
    return (
        "You are the Answer Generator module in an adaptive RAG system.\n"
        "Answer the question using only the provided context when possible.\n"
        "Rules:\n"
        "- prefer short exact answers for factual questions\n"
        "- if evidence is missing, say the answer is uncertain\n"
        "- cite evidence_ids from the context list\n"
        f"{_json_instruction(schema)}\n"
        f"Question: {question}\n"
        f"Context:\n{context}"
    )


def summarize_prompt(question: str, partial_answers: List[str]) -> str:
    schema = {"final_answer": "one fused answer", "consistency_note": "brief note"}
    return (
        "You are the Answer Summarizer module in an adaptive RAG system.\n"
        "Fuse the partial answers into one final answer to the original question.\n"
        "Rules:\n"
        "- prefer the most specific consistent answer\n"
        "- if partial answers conflict, say so briefly\n"
        f"{_json_instruction(schema)}\n"
        f"Original question: {question}\n"
        f"Partial answers: {json.dumps(partial_answers, ensure_ascii=False)}"
    )
