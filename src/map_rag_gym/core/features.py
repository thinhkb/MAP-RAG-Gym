from __future__ import annotations

from map_rag_gym.core.schemas import QuestionFeatures


COMPARATIVE_WORDS = {"compare", "difference", "more", "less", "higher", "lower", "older", "younger", "than", "versus", "vs"}
AMBIGUOUS_WORDS = {"it", "they", "this", "that", "he", "she", "there"}
WH_WORDS = ["what", "who", "when", "where", "which", "why", "how"]
TEMPORAL_WORDS = {"after", "before", "during", "when", "until", "since", "while", "year", "century", "decade"}
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "none"}
SUPERLATIVE_WORDS = {"most", "least", "best", "worst", "first", "last", "largest", "smallest", "oldest", "youngest", "highest", "lowest"}
STOPWORDS_FOR_ENTITY = {"the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "was", "were", "by",
                        "at", "with", "from", "that", "this", "it", "are", "be", "has", "had", "have",
                        "do", "did", "does", "what", "who", "when", "where", "which", "why", "how",
                        "both", "also", "than", "but", "not", "if", "its", "as", "their", "his", "her"}


def _count_likely_entities(question: str) -> int:
    """Rough count of capitalized multi-word spans (likely named entities)."""
    words = question.replace("?", "").replace(",", " ").split()
    entity_count = 0
    in_entity = False
    for i, word in enumerate(words):
        if i == 0:
            # First word is always capitalized, check if it's an entity or just sentence start
            if word.lower() not in STOPWORDS_FOR_ENTITY and len(word) > 1:
                in_entity = True
            continue
        if word[0].isupper() and word.lower() not in STOPWORDS_FOR_ENTITY:
            if not in_entity:
                entity_count += 1
                in_entity = True
        else:
            in_entity = False
    return entity_count


def extract_question_features(question: str) -> QuestionFeatures:
    tokens = question.lower().replace("?", "").split()
    comparative_flag = int(any(t in COMPARATIVE_WORDS for t in tokens))
    conjunction_flag = int(any(t in {"and", "or", ","} for t in tokens))
    ambiguity_flag = int(any(t in AMBIGUOUS_WORDS for t in tokens))
    temporal_flag = int(any(t in TEMPORAL_WORDS for t in tokens))
    negation_flag = int(any(t in NEGATION_WORDS for t in tokens))
    superlative_flag = int(any(t in SUPERLATIVE_WORDS for t in tokens))
    wh_word = next((w for w in WH_WORDS if question.lower().startswith(w + " ")), "unknown")
    entity_count = _count_likely_entities(question)
    multi_entity_flag = int(entity_count >= 2)
    entity_density = round(entity_count / max(1, len(tokens)), 4)
    estimated_hops = 2 if comparative_flag or conjunction_flag else 1
    if "after" in tokens or "before" in tokens:
        estimated_hops = max(estimated_hops, 2)
    if multi_entity_flag and comparative_flag:
        estimated_hops = max(estimated_hops, 3)
    return QuestionFeatures(
        question=question,
        token_len=len(tokens),
        comparative_flag=comparative_flag,
        conjunction_flag=conjunction_flag,
        ambiguity_flag=ambiguity_flag,
        temporal_flag=temporal_flag,
        negation_flag=negation_flag,
        superlative_flag=superlative_flag,
        multi_entity_flag=multi_entity_flag,
        entity_density=entity_density,
        wh_word=wh_word,
        estimated_hops=estimated_hops,
    )
