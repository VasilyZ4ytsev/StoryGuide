import csv
import re
from collections import Counter
from functools import lru_cache

from src.dataset_loader import REVIEWS_DATASET_PATH, dataset_file_exists


TOKEN_PATTERN = re.compile(r"[a-z']{3,}", re.IGNORECASE)
LATIN_PATTERN = re.compile(r"[A-Za-z]")
STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "was", "are", "but", "not", "you",
    "have", "had", "his", "her", "its", "they", "them", "who", "too", "out", "all",
    "one", "what", "would", "there", "when", "from", "about", "into", "than", "then",
    "she", "him", "our", "were", "has", "been", "their", "because", "very", "just",
    "really", "more", "most", "can", "could", "did", "does", "isn", "don", "didn",
    "film", "movie", "movies", "show",
}


def _tokenize(text):
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(str(text or ""))
        if token.lower() not in STOPWORDS
    ]


@lru_cache(maxsize=1)
def load_sentiment_lexicon():
    if not dataset_file_exists(REVIEWS_DATASET_PATH):
        raise FileNotFoundError(
            "IMDB Dataset.csv не найден в data/raw. Добавьте датасет в проект."
        )

    positive_counts = Counter()
    negative_counts = Counter()

    with open(REVIEWS_DATASET_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            review = row.get("review", "")
            sentiment = str(row.get("sentiment", "")).strip().lower()
            tokens = _tokenize(review)
            if not tokens:
                continue

            if sentiment == "positive":
                positive_counts.update(tokens)
            elif sentiment == "negative":
                negative_counts.update(tokens)

    lexicon = {}
    vocabulary = set(positive_counts) | set(negative_counts)
    for token in vocabulary:
        positive_value = positive_counts[token]
        negative_value = negative_counts[token]
        total = positive_value + negative_value
        if total < 80:
            continue

        score = (positive_value - negative_value) / total
        if abs(score) < 0.22:
            continue
        lexicon[token] = round(score, 3)

    return lexicon


def analyze_review_sentiment(text):
    if not LATIN_PATTERN.search(str(text or "")):
        return None

    tokens = _tokenize(text)
    if len(tokens) < 6:
        return None

    lexicon = load_sentiment_lexicon()
    matched = [(token, lexicon[token]) for token in tokens if token in lexicon]
    if len(matched) < 2:
        return None

    score = sum(value for _, value in matched) / len(matched)
    if score > 0.08:
        label_text = "положительная"
    elif score < -0.08:
        label_text = "отрицательная"
    else:
        label_text = "нейтральная"

    dominant_words = []
    for token, _ in sorted(matched, key=lambda item: abs(item[1]), reverse=True):
        if token not in dominant_words:
            dominant_words.append(token)
        if len(dominant_words) == 5:
            break

    return {
        "label_text": label_text,
        "score": round(score, 3),
        "matched_words": dominant_words,
    }


def build_review_sentiment_summary(text):
    analysis = analyze_review_sentiment(text)
    if not analysis:
        return ""

    confidence = round(abs(analysis["score"]) * 100)
    words = ", ".join(analysis["matched_words"])
    return (
        f"Анализ отзыва по IMDB Dataset: тональность {analysis['label_text']} "
        f"({confidence}%). Опорные слова: {words}."
    )
