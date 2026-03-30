import math
import os
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache

from src.dataset_loader import MERGED_METADATA_CACHE_PATH, load_movie_metadata


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_SEARCH_INDEX_PATH = os.path.join(BASE_DIR, "data", "processed", "text_search_index.pkl")
TEXT_INDEX_VERSION = 2
TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "по", "для", "про", "как", "что", "это", "или",
    "хочу", "мне", "нравится", "посоветуй", "посоветовать", "найти", "похожий", "похожее",
    "фильм", "фильмы", "movie", "movies", "like", "find", "about", "with", "from", "into",
    "through", "after", "before", "inside", "story", "about", "film",
    "a", "an", "and", "the", "is", "are", "to", "of", "for", "on",
}
TOKEN_ALIASES = {
    "dreams": "dream",
    "dreamlike": "dream",
    "dreamscape": "dream",
    "subconsciousness": "subconscious",
    "subconsciouses": "subconscious",
    "theft": "heist",
    "thief": "heist",
    "thieves": "heist",
    "steal": "heist",
    "steals": "heist",
    "stealing": "heist",
    "stolen": "heist",
    "espionage": "heist",
    "simulated": "simulation",
    "simulator": "simulation",
    "hackers": "hacker",
    "discovers": "discover",
    "discovered": "discover",
    "discovering": "discover",
    "discovery": "discover",
    "humanity": "human",
    "humans": "human",
}


def _normalize_token(token):
    token = str(token or "").lower()
    if not token:
        return ""

    if re.fullmatch(r"[a-z0-9]+", token):
        if token.endswith("ness") and len(token) > 7:
            token = token[:-4]
        elif token.endswith("ies") and len(token) > 5:
            token = f"{token[:-3]}y"
        elif token.endswith("ing") and len(token) > 6:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif (
            token.endswith("s")
            and len(token) > 3
            and not token.endswith(("ss", "us", "is", "ous"))
        ):
            token = token[:-1]

    return TOKEN_ALIASES.get(token, token)


def tokenize_text(text):
    normalized_tokens = []
    for raw_token in TOKEN_PATTERN.findall(str(text or "").lower()):
        token = _normalize_token(raw_token)
        if len(token) < 2 or token in STOPWORDS:
            continue
        normalized_tokens.append(token)

    bigrams = [
        f"{left}_{right}"
        for left, right in zip(normalized_tokens, normalized_tokens[1:])
        if len(left) >= 3 and len(right) >= 3
    ]
    return normalized_tokens + bigrams


def build_semantic_document(record):
    parts = [
        record.get("title_ru", ""),
        record.get("display_title", ""),
        record.get("tagline_ru", ""),
        record.get("tagline", ""),
        record.get("overview_ru", ""),
        record.get("overview", ""),
        " ".join(record.get("keywords", [])),
        " ".join(record.get("cast", [])),
        record.get("director", ""),
        " ".join(record.get("movielens_tags", [])),
        " ".join(record.get("genres_ru", [])),
        " ".join(record.get("genres_en", [])),
    ]
    return " ".join(part for part in parts if part)


def _cache_signature():
    if os.path.exists(MERGED_METADATA_CACHE_PATH):
        stat = os.stat(MERGED_METADATA_CACHE_PATH)
        return (
            TEXT_INDEX_VERSION,
            os.path.abspath(MERGED_METADATA_CACHE_PATH),
            stat.st_size,
            round(stat.st_mtime, 3),
        )

    records = load_movie_metadata()
    return (TEXT_INDEX_VERSION, "records", len(records))


def _load_cache(signature):
    if not os.path.exists(TEXT_SEARCH_INDEX_PATH):
        return None
    try:
        with open(TEXT_SEARCH_INDEX_PATH, "rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError):
        return None

    if payload.get("signature") != signature:
        return None
    return payload.get("data")


def _save_cache(signature, data):
    os.makedirs(os.path.dirname(TEXT_SEARCH_INDEX_PATH), exist_ok=True)
    with open(TEXT_SEARCH_INDEX_PATH, "wb") as file:
        pickle.dump({"signature": signature, "data": data}, file, protocol=pickle.HIGHEST_PROTOCOL)


@lru_cache(maxsize=1)
def load_text_search_index():
    signature = _cache_signature()
    cached = _load_cache(signature)
    if cached is not None:
        return cached

    records = load_movie_metadata()
    if not records:
        raise RuntimeError("Movie metadata is empty; cannot build text search index.")
    document_tokens = []
    document_frequency = Counter()

    for record in records:
        tokens = tokenize_text(build_semantic_document(record))
        token_counts = Counter(tokens)
        document_tokens.append(token_counts)
        document_frequency.update(token_counts.keys())

    document_count = len(records)
    idf = {
        token: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }

    postings = defaultdict(list)
    doc_norms = []
    record_ids = []
    doc_token_counts = []

    for record, token_counts in zip(records, document_tokens):
        total = sum(token_counts.values())
        if total == 0:
            doc_norms.append(0.0)
            record_ids.append(record.get("imdb_id", ""))
            doc_token_counts.append(0)
            continue

        weighted_terms = {}
        squared_norm = 0.0
        for token, count in token_counts.items():
            weight = (count / total) * idf.get(token, 1.0)
            weighted_terms[token] = weight
            squared_norm += weight * weight

        doc_idx = len(record_ids)
        norm = math.sqrt(squared_norm) if squared_norm > 0.0 else 1.0
        for token, weight in weighted_terms.items():
            postings[token].append((doc_idx, weight))

        doc_norms.append(norm)
        record_ids.append(record.get("imdb_id", ""))
        doc_token_counts.append(len(token_counts))

    index = {
        "idf": idf,
        "postings": dict(postings),
        "doc_norms": doc_norms,
        "record_ids": record_ids,
        "doc_token_counts": doc_token_counts,
    }
    _save_cache(signature, index)
    return index


def search_text_index(query_text, limit=300):
    index = load_text_search_index()
    query_tokens = tokenize_text(query_text)
    if not query_tokens:
        return []

    token_counts = Counter(query_tokens)
    total = sum(token_counts.values())
    query_weights = {}
    for token, count in token_counts.items():
        idf = index["idf"].get(token)
        if idf is None:
            continue
        query_weights[token] = (count / total) * idf

    if not query_weights:
        return []

    query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
    if query_norm == 0.0:
        return []

    dot_scores = defaultdict(float)
    matched_terms = defaultdict(int)
    matched_weight_sums = defaultdict(float)

    for token, query_weight in query_weights.items():
        for doc_idx, doc_weight in index["postings"].get(token, []):
            dot_scores[doc_idx] += query_weight * doc_weight
            matched_terms[doc_idx] += 1
            matched_weight_sums[doc_idx] += query_weight

    results = []
    query_weight_total = sum(query_weights.values()) or 1.0
    for doc_idx, dot_score in dot_scores.items():
        doc_norm = index["doc_norms"][doc_idx]
        if doc_norm == 0.0:
            continue
        cosine_score = dot_score / (query_norm * doc_norm)
        matched_ratio = matched_terms[doc_idx] / max(len(query_weights), 1)
        matched_weight_ratio = matched_weight_sums[doc_idx] / query_weight_total
        results.append(
            {
                "doc_idx": doc_idx,
                "imdb_id": index["record_ids"][doc_idx],
                "cosine_score": round(cosine_score, 4),
                "matched_ratio": round(matched_ratio, 4),
                "matched_weight_ratio": round(matched_weight_ratio, 4),
            }
        )

    results.sort(
        key=lambda item: (
            item["cosine_score"],
            item["matched_weight_ratio"],
            item["matched_ratio"],
        ),
        reverse=True,
    )
    return results[:limit]
