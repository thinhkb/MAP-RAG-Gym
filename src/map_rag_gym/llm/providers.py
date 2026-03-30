from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    estimated_tokens: int


class BaseLLM:
    def generate(self, prompt: str, n: int = 1) -> List[LLMResponse]:
        raise NotImplementedError


class DummyLLM(BaseLLM):
    def __init__(self, model: str = "dummy-free") -> None:
        self.model = model

    def _extract(self, prompt: str, key: str) -> str:
        match = re.search(rf"{key}:\s*(.*)", prompt, re.S)
        return match.group(1).strip() if match else ""

    def _question(self, prompt: str) -> str:
        for key in ["Question", "Original question"]:
            m = re.search(rf"{key}:\s*(.*?)($|\n[A-Z][^\n]*:)", prompt, re.S | re.M)
            if m:
                return m.group(1).strip()
        return ""

    def _context(self, prompt: str) -> str:
        m = re.search(r"Context:\s*(.*)", prompt, re.S)
        return m.group(1).strip() if m else ""

    def _json(self, payload: Dict[str, Any], est: int) -> List[LLMResponse]:
        return [LLMResponse(text=json.dumps(payload, ensure_ascii=False), provider="dummy", model=self.model, estimated_tokens=est)]

    def _answer_from_context(self, question: str, context: str) -> str:
        q = question.lower()
        ctx = context.lower()
        if "who wrote pride and prejudice" in q and "jane austen" in ctx:
            return "Jane Austen"
        if "published posthumously" in q and "persuasion" in ctx:
            return "Persuasion"
        if "compare california and japan" in q:
            return "California has a GDP larger than Japan in this sample corpus"
        if not context:
            return "uncertain"
        first_sent = re.split(r"[.!?]", context)[0].strip()
        return first_sent or "uncertain"

    def generate(self, prompt: str, n: int = 1) -> List[LLMResponse]:
        est = max(1, len(prompt.split()) // 2)
        prompt_low = prompt.lower()
        question = self._question(prompt)
        context = self._context(prompt)
        if "query rewriter module" in prompt_low:
            query = re.sub(r"\?$", "", question).strip()
            return self._json({"query": query, "rationale": "preserve key entities"}, est)
        if "parallel decomposition module" in prompt_low:
            parts = [p.strip() for p in re.split(r"\band\b|\bvs\b|\bversus\b", question, flags=re.I) if p.strip()]
            if len(parts) < 2:
                parts = [question, f"Find supporting fact for: {question}"]
            return self._json({"sub_questions": parts[:3], "reason": "split by independent components"}, est)
        if "serial decomposition module" in prompt_low:
            if "author of" in question.lower() and "published posthumously" in question.lower():
                parts = ["Who is the author of Pride and Prejudice?", "Which novel by that author was published posthumously?"]
            elif "after" in question.lower():
                left, right = question.split(" after ", 1)
                parts = [left.strip(), "after " + right.strip()]
            else:
                parts = [question, f"Find the bridge entity for: {question}"]
            return self._json({"sub_questions": parts, "reason": "later step depends on earlier answer"}, est)
        if "draft reasoner" in prompt_low:
            return self._json({"draft_reasoning": f"Need evidence to answer: {question}", "predicted_answer": "unknown"}, est)
        if "reflection module" in prompt_low:
            return self._json({"missing_information": ["missing supporting fact"], "query": question.replace("?", "")}, est)
        if "answer generator module" in prompt_low:
            answer = self._answer_from_context(question, context)
            return self._json({"answer": answer, "evidence_ids": [1] if context else [], "confidence": 0.6 if context else 0.2}, est)
        if "answer summarizer module" in prompt_low:
            m = re.search(r"Partial answers:\s*(.*)", prompt, re.S)
            items = []
            if m:
                try:
                    items = json.loads(m.group(1).strip())
                except Exception:
                    items = [m.group(1).strip()]
            final = items[0] if items else "uncertain"
            return self._json({"final_answer": final, "consistency_note": "dummy summarizer"}, est)
        base = prompt.strip().splitlines()[-1][:180]
        return [LLMResponse(text=base, provider="dummy", model=self.model, estimated_tokens=est) for _ in range(n)]


class GeminiLLM(BaseLLM):
    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        # Load from .env file if it exists
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def generate(self, prompt: str, n: int = 1) -> List[LLMResponse]:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        out = []
        for _ in range(n):
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                },
            }
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            est = max(1, len(prompt.split()) // 2)
            out.append(LLMResponse(text=text, provider="gemini", model=self.model, estimated_tokens=est))
        return out


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "llama3.1:8b-instruct", base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(self, prompt: str, n: int = 1) -> List[LLMResponse]:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        out = []
        for _ in range(n):
            payload = {"model": self.model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.2}}
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            out.append(LLMResponse(text=text, provider="ollama", model=self.model, estimated_tokens=max(1, len(prompt.split()) // 2)))
        return out


def try_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {"raw_text": text}
    return {"raw_text": text}


def build_llm(provider: str = "dummy", model: str | None = None) -> BaseLLM:
    provider = provider.lower()
    if provider == "gemini":
        return GeminiLLM(model or "gemini-2.5-flash")
    if provider == "ollama":
        return OllamaLLM(model or "llama3.1:8b-instruct")
    return DummyLLM(model or "dummy-free")
