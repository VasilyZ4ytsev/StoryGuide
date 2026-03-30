import math

from src.dataset_loader import load_movie_metadata, match_movie_title
from src.text_search_index import search_text_index, tokenize_text


GENRE_KEYWORDS = {
    "боевик": {"боевик", "action"},
    "приключения": {"приключения", "adventure"},
    "анимация": {"анимация", "мультфильм", "animation"},
    "комедия": {"комедия", "comedy"},
    "криминал": {"криминал", "crime"},
    "драма": {"драма", "drama"},
    "семейный": {"семейный", "family"},
    "фэнтези": {"фэнтези", "fantasy"},
    "ужасы": {"ужасы", "horror"},
    "детектив": {"детектив", "mystery", "detective"},
    "романтика": {"романтика", "romance", "romantic"},
    "фантастика": {"фантастика", "sci-fi", "science", "fiction"},
    "триллер": {"триллер", "thriller"},
}


def _passes_year_filter(record, year_filter):
    if not year_filter:
        return True

    year = record.get("release_year")
    if not isinstance(year, int):
        return False

    if "exact_year" in year_filter:
        return year == year_filter["exact_year"]

    min_year = year_filter.get("min_year")
    max_year = year_filter.get("max_year")
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def _passes_genre_filters(record, include_genres=None, exclude_genres=None):
    record_genres = {genre.strip().lower() for genre in record.get("genres_ru", []) if genre}
    include_set = {genre.strip().lower() for genre in include_genres or [] if genre}
    exclude_set = {genre.strip().lower() for genre in exclude_genres or [] if genre}

    if include_set and not (record_genres & include_set):
        return False
    if exclude_set and (record_genres & exclude_set):
        return False
    return True


def detect_genres_in_text(text):
    normalized_text = str(text or "").lower()
    detected = []
    for genre, variants in GENRE_KEYWORDS.items():
        if any(variant in normalized_text for variant in variants):
            detected.append(genre)
    return detected


def _prefix_similarity(query_tokens, candidate_tokens):
    if not query_tokens or not candidate_tokens:
        return 0.0

    matches = 0
    for query_token in query_tokens:
        if any(
            query_token == candidate_token
            or (
                min(len(query_token), len(candidate_token)) >= 4
                and (
                    query_token.startswith(candidate_token)
                    or candidate_token.startswith(query_token)
                )
            )
            for candidate_token in candidate_tokens
        ):
            matches += 1

    return matches / max(len(query_tokens), 1)


def _is_title_like_query(query_tokens):
    unigram_tokens = [token for token in query_tokens if "_" not in token]
    return 0 < len(unigram_tokens) <= 4


def _semantic_unigram_tokens(record):
    return set(
        token
        for token in tokenize_text(
            " ".join(
                [
                    record.get("overview", ""),
                    record.get("tagline", ""),
                    " ".join(record.get("keywords", [])),
                    " ".join(record.get("movielens_tags", [])),
                ]
            )
        )
        if "_" not in token
    )


def recommend_similar_movies(
    source_record,
    year_filter=None,
    include_genres=None,
    exclude_genres=None,
    limit=5,
):
    source_genres = set(source_record.get("genres_ru", []))
    source_keywords = set(tokenize_text(" ".join(source_record.get("keywords", []))))
    source_tags = set(tokenize_text(" ".join(source_record.get("movielens_tags", []))))
    source_director = str(source_record.get("director", "")).strip().lower()
    source_year = source_record.get("release_year")
    recommendations = []

    for candidate in load_movie_metadata():
        if candidate.get("imdb_id") == source_record.get("imdb_id"):
            continue
        if not _passes_year_filter(candidate, year_filter):
            continue
        if not _passes_genre_filters(
            candidate,
            include_genres=include_genres,
            exclude_genres=exclude_genres,
        ):
            continue

        candidate_genres = set(candidate.get("genres_ru", []))
        genre_overlap = len(source_genres & candidate_genres)
        candidate_keywords = set(tokenize_text(" ".join(candidate.get("keywords", []))))
        candidate_tags = set(tokenize_text(" ".join(candidate.get("movielens_tags", []))))
        keyword_overlap = len(source_keywords & candidate_keywords)
        tag_overlap = len(source_tags & candidate_tags)
        same_director = (
            bool(source_director)
            and source_director == str(candidate.get("director", "")).strip().lower()
        )

        if genre_overlap == 0 and keyword_overlap == 0 and tag_overlap == 0 and not same_director:
            continue

        year = candidate.get("release_year")
        if isinstance(source_year, int) and isinstance(year, int):
            year_distance = abs(source_year - year)
            year_score = max(0.0, 1.0 - min(year_distance, 20) / 20.0)
        else:
            year_score = 0.3

        rating = float(candidate.get("rating", 0.0))
        community_rating = float(candidate.get("movielens_rating", 0.0))
        include_bonus = 0.4 if include_genres else 0.0
        score = (
            genre_overlap * 2.7
            + keyword_overlap * 1.4
            + tag_overlap * 0.6
            + (1.2 if same_director else 0.0)
            + year_score * 1.6
            + rating / 6.0
            + community_rating / 10.0
            + include_bonus
        )
        recommendations.append(
            {
                "record": candidate,
                "score": round(score, 4),
                "genre_overlap": genre_overlap,
                "keyword_overlap": keyword_overlap,
                "tag_overlap": tag_overlap,
                "year_score": round(year_score, 4),
            }
        )

    recommendations.sort(key=lambda item: (item["score"], item["record"]["rating"]), reverse=True)
    return recommendations[:limit]


def search_movies_by_query(
    query_text,
    year_filter=None,
    include_genres=None,
    exclude_genres=None,
    limit=5,
):
    query_text = str(query_text or "").strip()
    if not query_text:
        return {"mode": "empty", "source": None, "matches": []}

    title_match = match_movie_title(query_text)
    if title_match is not None:
        source_record = title_match["record"]
        return {
            "mode": "title_match",
            "source": source_record,
            "match_score": title_match["score"],
            "matches": recommend_similar_movies(
                source_record,
                year_filter=year_filter,
                include_genres=include_genres,
                exclude_genres=exclude_genres,
                limit=limit,
            ),
        }

    query_tokens = tokenize_text(query_text)
    detected_genres = detect_genres_in_text(query_text)
    if not query_tokens and not detected_genres:
        return {"mode": "empty", "source": None, "matches": []}

    records_by_imdb = {record.get("imdb_id"): record for record in load_movie_metadata()}
    semantic_candidates = search_text_index(query_text, limit=400)
    matches = []
    title_like_query = _is_title_like_query(query_tokens)
    query_token_set = set(query_tokens)
    query_unigram_tokens = {token for token in query_tokens if "_" not in token}

    for semantic_candidate in semantic_candidates:
        record = records_by_imdb.get(semantic_candidate["imdb_id"])
        if record is None:
            continue
        if not _passes_year_filter(record, year_filter):
            continue
        if not _passes_genre_filters(
            record,
            include_genres=include_genres,
            exclude_genres=exclude_genres,
        ):
            continue

        cosine_score = semantic_candidate["cosine_score"]
        semantic_overlap = semantic_candidate.get("matched_weight_ratio", semantic_candidate["matched_ratio"])
        term_coverage = semantic_candidate["matched_ratio"]
        fuzzy_score = _prefix_similarity(query_tokens, tokenize_text(record.get("title", "")))
        keyword_tokens = set(
            tokenize_text(" ".join(record.get("keywords", []) + record.get("movielens_tags", [])))
        )
        keyword_overlap = len(query_token_set & keyword_tokens) / max(len(query_token_set), 1)
        genre_bonus = len(set(detected_genres) & set(record.get("genres_ru", []))) * 0.25
        semantic_unigrams = _semantic_unigram_tokens(record)
        concept_hits = len(query_unigram_tokens & semantic_unigrams)
        concept_bonus = 0.0
        if concept_hits >= 3:
            concept_bonus = 0.08 + min(concept_hits - 3, 2) * 0.02
        elif concept_hits == 2 and not title_like_query:
            concept_bonus = 0.03

        include_bonus = 0.08 if include_genres else 0.0
        evidence_score = (
            cosine_score * 0.56
            + semantic_overlap * 0.24
            + term_coverage * 0.08
            + keyword_overlap * 0.1
            + fuzzy_score * (0.08 if title_like_query else 0.015)
            + genre_bonus
            + concept_bonus
            + include_bonus
        )
        if evidence_score < 0.1:
            continue

        rating_bonus = min(float(record.get("rating", 0.0)) / 20.0, 0.5) * 0.12
        community_bonus = min(float(record.get("movielens_rating", 0.0)) / 10.0, 0.5) * 0.12
        final_score = evidence_score + rating_bonus + community_bonus

        matches.append(
            {
                "record": record,
                "score": round(final_score, 4),
                "cosine_score": round(cosine_score, 4),
                "fuzzy_score": round(fuzzy_score, 4),
                "semantic_overlap": round(semantic_overlap, 4),
                "keyword_overlap": round(keyword_overlap, 4),
                "genre_bonus": round(genre_bonus, 4),
                "concept_bonus": round(concept_bonus, 4),
            }
        )

    deduplicated_matches = []
    seen_ids = set()
    for item in sorted(matches, key=lambda item: (item["score"], item["record"]["rating"]), reverse=True):
        imdb_id = item["record"].get("imdb_id")
        if imdb_id in seen_ids:
            continue
        seen_ids.add(imdb_id)
        deduplicated_matches.append(item)

    return {
        "mode": "hybrid_query",
        "source": None,
        "detected_genres": detected_genres,
        "matches": deduplicated_matches[:limit],
    }
