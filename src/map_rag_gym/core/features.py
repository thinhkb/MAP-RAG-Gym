from __future__ import annotations

from map_rag_gym.core.schemas import QuestionFeatures


COMPARATIVE_WORDS = {"compare", "difference", "more", "less", "higher", "lower", "older", "younger", "than", "versus", "vs"}
AMBIGUOUS_WORDS = {"it", "they", "this", "that", "he", "she", "there"}
WH_WORDS = ["what", "who", "when", "where", "which", "why", "how"]


def extract_question_features(question: str) -> QuestionFeatures:
    tokens = question.lower().replace("?", "").split()
    comparative_flag = int(any(t in COMPARATIVE_WORDS for t in tokens))
    conjunction_flag = int(any(t in {"and", "or", ","} for t in tokens))
    ambiguity_flag = int(any(t in AMBIGUOUS_WORDS for t in tokens))
    wh_word = next((w for w in WH_WORDS if question.lower().startswith(w + " ")), "unknown")
    estimated_hops = 2 if comparative_flag or conjunction_flag else 1
    if "after" in tokens or "before" in tokens:
        estimated_hops = max(estimated_hops, 2)
    return QuestionFeatures(
        question=question,
        token_len=len(tokens),
        comparative_flag=comparative_flag,
        conjunction_flag=conjunction_flag,
        ambiguity_flag=ambiguity_flag,
        wh_word=wh_word,
        estimated_hops=estimated_hops,
    )
