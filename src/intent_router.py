import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

from src.text_search_index import tokenize_text


INTENT_EXAMPLES = {
    "recommend_similar": [
        "выдай похожие фильмы на крестный отец",
        "посоветуй фильмы похожие на этот",
        "дай список фильмов похожих на интерстеллар",
        "я посмотрел фильм и хочу похожее",
        "подбери что нибудь похожее на этот фильм",
        "какие фильмы похожи на матрицу",
        "хочу 10 фильмов как крестный отец",
    ],
    "recommend_by_description": [
        "хочу фильм про космос и путешествия",
        "подбери триллер про маньяка",
        "посоветуй фантастику про искусственный интеллект",
        "найди фильм по описанию сюжета",
    ],
    "refine_filters": [
        "без комедии",
        "после 2010",
        "только фантастика",
        "что нибудь поновее",
        "убери драму",
        "дай 10 штук",
    ],
    "reset_topic": [
        "давай начнем заново",
        "новая тема",
        "сбрось контекст",
        "очисти диалог",
    ],
}

INTENT_SCORE_THRESHOLD = 0.17
LIMIT_PATTERN = re.compile(
    r"\b(?:топ\s*)?(?P<value>\d{1,2})\s*(?:штук|фильмов|фильма|фильм|вариантов|варианта)?\b",
    flags=re.IGNORECASE,
)
WORD_NUMBER_MAP = {
    "один": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "пятнадцать": 15,
    "двадцать": 20,
}


def _build_vector(text):
    return Counter(tokenize_text(text))


def _cosine_similarity(left_vector, right_vector):
    if not left_vector or not right_vector:
        return 0.0

    dot_product = 0.0
    for token, value in left_vector.items():
        dot_product += value * right_vector.get(token, 0.0)

    left_norm = math.sqrt(sum(value * value for value in left_vector.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vector.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


@lru_cache(maxsize=1)
def _compiled_examples():
    compiled = defaultdict(list)
    for intent, examples in INTENT_EXAMPLES.items():
        for example in examples:
            compiled[intent].append(_build_vector(example))
    return dict(compiled)


def classify_intent(text):
    query_vector = _build_vector(text)
    if not query_vector:
        return {"intent": "unknown", "score": 0.0}

    best_intent = "unknown"
    best_score = 0.0
    for intent, vectors in _compiled_examples().items():
        score = max((_cosine_similarity(query_vector, vector) for vector in vectors), default=0.0)
        if score > best_score:
            best_intent = intent
            best_score = score

    if best_score < INTENT_SCORE_THRESHOLD:
        return {"intent": "unknown", "score": round(best_score, 4)}
    return {"intent": best_intent, "score": round(best_score, 4)}


def extract_requested_limit(text, default_limit=5, min_limit=1, max_limit=20):
    normalized_text = str(text or "").lower()
    digit_match = LIMIT_PATTERN.search(normalized_text)
    if digit_match:
        value = int(digit_match.group("value"))
        return max(min_limit, min(max_limit, value))

    for word, value in WORD_NUMBER_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized_text):
            return max(min_limit, min(max_limit, value))

    return max(min_limit, min(max_limit, int(default_limit or 5)))
