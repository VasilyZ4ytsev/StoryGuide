import ast
import csv
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from io import StringIO
from functools import lru_cache


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MERGED_METADATA_CACHE_PATH = os.path.join(PROCESSED_DIR, "merged_movie_metadata.pkl")
MOVIE_LOCALIZATIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "movie_localizations.json")

MOVIE_DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "MovieGenre.csv")
REVIEWS_DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "IMDB Dataset.csv")
THE_MOVIES_METADATA_PATH = os.path.join(BASE_DIR, "data", "raw", "movies_metadata.csv")
THE_MOVIES_CREDITS_PATH = os.path.join(BASE_DIR, "data", "raw", "credits.csv")
THE_MOVIES_KEYWORDS_PATH = os.path.join(BASE_DIR, "data", "raw", "keywords.csv")
THE_MOVIES_LINKS_PATH = os.path.join(BASE_DIR, "data", "raw", "links.csv")
MOVIELENS_DIR = os.path.join(BASE_DIR, "data", "raw", "ml-25m")
MOVIELENS_MOVIES_PATH = os.path.join(MOVIELENS_DIR, "movies.csv")
MOVIELENS_RATINGS_PATH = os.path.join(MOVIELENS_DIR, "ratings.csv")
MOVIELENS_TAGS_PATH = os.path.join(MOVIELENS_DIR, "tags.csv")
MOVIELENS_LINKS_PATH = os.path.join(MOVIELENS_DIR, "links.csv")

TITLE_YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")
NORMALIZE_PATTERN = re.compile(r"[^a-zа-яё0-9]+", re.IGNORECASE)

GENRE_TRANSLATIONS = {
    "action": "боевик",
    "adventure": "приключения",
    "animation": "анимация",
    "biography": "биография",
    "comedy": "комедия",
    "crime": "криминал",
    "documentary": "документальный",
    "drama": "драма",
    "family": "семейный",
    "fantasy": "фэнтези",
    "film-noir": "нуар",
    "history": "исторический",
    "horror": "ужасы",
    "music": "музыка",
    "musical": "мюзикл",
    "mystery": "детектив",
    "romance": "романтика",
    "sci-fi": "фантастика",
    "science fiction": "фантастика",
    "short": "короткометражный",
    "sport": "спорт",
    "thriller": "триллер",
    "war": "военный",
    "western": "вестерн",
}


def _normalize_lookup_text(value):
    lowered = str(value or "").strip().lower()
    return NORMALIZE_PATTERN.sub(" ", lowered).strip()


def _normalize_imdb_id(value):
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("tt"):
        raw = raw[2:]
    raw = "".join(char for char in raw if char.isdigit())
    if not raw:
        return ""
    return str(int(raw))


def _normalize_tmdb_id(value):
    raw = "".join(char for char in str(value or "").strip() if char.isdigit())
    if not raw:
        return ""
    return str(int(raw))


def _parse_release_year(title):
    match = TITLE_YEAR_PATTERN.search(title)
    if not match:
        return None
    return int(match.group(1))


def _parse_release_year_from_date(value):
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    if not match:
        return None
    return int(match.group(0))


def _clean_title(title):
    return TITLE_YEAR_PATTERN.sub("", str(title or "")).strip()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return default


def _translated_genres(raw_genres):
    translated = []
    for genre in raw_genres:
        key = genre.strip().lower()
        translated.append(GENRE_TRANSLATIONS.get(key, key))
    return [genre for genre in translated if genre]


def dataset_file_exists(path):
    return os.path.exists(path)


def _open_dataset_csv(path):
    encodings = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
    raw_bytes = open(path, "rb").read()
    last_error = None

    for encoding in encodings:
        try:
            return StringIO(raw_bytes.decode(encoding))
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "Unsupported encoding")


def _parse_jsonish_list(raw_value):
    raw_text = str(raw_value or "").strip()
    if not raw_text or raw_text in {"[]", "{}", "nan", "None"}:
        return []

    try:
        parsed = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, list):
        return parsed
    return []


def _extract_names(raw_value, limit=None):
    names = []
    for item in _parse_jsonish_list(raw_value):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
        if limit is not None and len(names) >= limit:
            break
    return names


def _extract_director(raw_value):
    for item in _parse_jsonish_list(raw_value):
        if not isinstance(item, dict):
            continue
        if str(item.get("job", "")).strip().lower() == "director":
            return str(item.get("name", "")).strip()
    return ""


def _translate_movielens_genres(raw_value):
    items = [item for item in str(raw_value or "").split("|") if item and item != "(no genres listed)"]
    return items, _translated_genres(items)


def _has_cyrillic(text):
    return bool(re.search(r"[А-Яа-яЁё]", str(text or "")))


def _build_display_full_title(title, year):
    title = str(title or "").strip()
    if not title:
        return ""
    if isinstance(year, int):
        return f"{title} ({year})"
    return title


def _normalize_localization_entry(raw_entry):
    if not isinstance(raw_entry, dict):
        return {}

    normalized_entry = {}
    for key in ("title_ru", "overview_ru", "tagline_ru"):
        value = str(raw_entry.get(key, "")).strip()
        if value:
            normalized_entry[key] = value
    return normalized_entry


@lru_cache(maxsize=1)
def _load_movie_localizations():
    by_imdb = {}
    by_tmdb = {}
    by_title_year = {}

    if not dataset_file_exists(MOVIE_LOCALIZATIONS_PATH):
        return {"by_imdb": by_imdb, "by_tmdb": by_tmdb, "by_title_year": by_title_year}

    with open(MOVIE_LOCALIZATIONS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    entries = payload.get("movies", []) if isinstance(payload, dict) else []
    for entry in entries:
        normalized_entry = _normalize_localization_entry(entry)
        if not normalized_entry:
            continue

        imdb_id = _normalize_imdb_id(entry.get("imdb_id"))
        tmdb_id = _normalize_tmdb_id(entry.get("tmdb_id"))
        title = _normalize_lookup_text(entry.get("title"))
        year = _safe_int(entry.get("year"))

        if imdb_id:
            by_imdb[imdb_id] = normalized_entry
        if tmdb_id:
            by_tmdb[tmdb_id] = normalized_entry
        if title and year is not None:
            by_title_year[(title, year)] = normalized_entry

    return {"by_imdb": by_imdb, "by_tmdb": by_tmdb, "by_title_year": by_title_year}


def _pick_localization(base_record, themovies, movielens):
    localization_store = _load_movie_localizations()
    imdb_id = base_record.get("imdb_id", "")
    tmdb_id = (themovies or {}).get("tmdb_id") or (movielens or {}).get("tmdb_id") or ""
    title_key = (base_record.get("normalized_title", ""), base_record.get("release_year"))

    return (
        localization_store["by_imdb"].get(imdb_id)
        or localization_store["by_tmdb"].get(tmdb_id)
        or localization_store["by_title_year"].get(title_key)
        or {}
    )


def _source_signature(paths):
    signature = []
    for path in paths:
        if os.path.exists(path):
            stat = os.stat(path)
            signature.append((os.path.abspath(path), stat.st_size, round(stat.st_mtime, 3)))
        else:
            signature.append((os.path.abspath(path), None, None))
    return tuple(signature)


def _load_cache(path, signature):
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError):
        return None

    if payload.get("signature") != signature:
        return None
    return payload.get("data")


def _save_cache(path, signature, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump({"signature": signature, "data": data}, file, protocol=pickle.HIGHEST_PROTOCOL)


@lru_cache(maxsize=1)
def _load_themovies_enrichment():
    required_paths = [
        THE_MOVIES_METADATA_PATH,
        THE_MOVIES_CREDITS_PATH,
        THE_MOVIES_KEYWORDS_PATH,
    ]
    if not all(dataset_file_exists(path) for path in required_paths):
        return {"by_imdb": {}, "by_tmdb": {}, "by_title_year": {}}

    keywords_by_tmdb = {}
    with _open_dataset_csv(THE_MOVIES_KEYWORDS_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            tmdb_id = _normalize_tmdb_id(row.get("id"))
            if not tmdb_id:
                continue
            keywords_by_tmdb[tmdb_id] = _extract_names(row.get("keywords"), limit=12)

    credits_by_tmdb = {}
    with _open_dataset_csv(THE_MOVIES_CREDITS_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            tmdb_id = _normalize_tmdb_id(row.get("id"))
            if not tmdb_id:
                continue
            credits_by_tmdb[tmdb_id] = {
                "cast": _extract_names(row.get("cast"), limit=8),
                "director": _extract_director(row.get("crew")),
            }

    by_imdb = {}
    by_tmdb = {}
    by_title_year = {}
    with _open_dataset_csv(THE_MOVIES_METADATA_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            tmdb_id = _normalize_tmdb_id(row.get("id"))
            imdb_id = _normalize_imdb_id(row.get("imdb_id"))
            title = str(row.get("title") or row.get("original_title") or "").strip()
            normalized_title = _normalize_lookup_text(title)
            release_year = _parse_release_year_from_date(row.get("release_date"))

            genres_en = _extract_names(row.get("genres"), limit=8)
            enrichment = {
                "tmdb_id": tmdb_id,
                "imdb_id": imdb_id,
                "metadata_title": title,
                "normalized_metadata_title": normalized_title,
                "original_title": str(row.get("original_title", "")).strip(),
                "overview": str(row.get("overview", "")).strip(),
                "tagline": str(row.get("tagline", "")).strip(),
                "metadata_release_year": release_year,
                "metadata_genres_en": genres_en,
                "metadata_genres_ru": _translated_genres(genres_en),
                "keywords": keywords_by_tmdb.get(tmdb_id, []),
                "cast": credits_by_tmdb.get(tmdb_id, {}).get("cast", []),
                "director": credits_by_tmdb.get(tmdb_id, {}).get("director", ""),
                "tmdb_popularity": _safe_float(row.get("popularity"), 0.0),
            }

            if imdb_id:
                by_imdb[imdb_id] = enrichment
            if tmdb_id:
                by_tmdb[tmdb_id] = enrichment
            if normalized_title and release_year is not None:
                by_title_year[(normalized_title, release_year)] = enrichment

    return {"by_imdb": by_imdb, "by_tmdb": by_tmdb, "by_title_year": by_title_year}


@lru_cache(maxsize=1)
def _load_movielens_enrichment():
    required_paths = [
        MOVIELENS_MOVIES_PATH,
        MOVIELENS_RATINGS_PATH,
        MOVIELENS_TAGS_PATH,
        MOVIELENS_LINKS_PATH,
    ]
    if not all(dataset_file_exists(path) for path in required_paths):
        return {"by_imdb": {}, "by_tmdb": {}}

    links_by_movie_id = {}
    with _open_dataset_csv(MOVIELENS_LINKS_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie_id = _safe_int(row.get("movieId"))
            if movie_id is None:
                continue
            links_by_movie_id[movie_id] = {
                "imdb_id": _normalize_imdb_id(row.get("imdbId")),
                "tmdb_id": _normalize_tmdb_id(row.get("tmdbId")),
            }

    movie_info_by_id = {}
    with _open_dataset_csv(MOVIELENS_MOVIES_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie_id = _safe_int(row.get("movieId"))
            if movie_id is None:
                continue
            genres_en, genres_ru = _translate_movielens_genres(row.get("genres"))
            movie_info_by_id[movie_id] = {
                "movielens_title": str(row.get("title", "")).strip(),
                "movielens_genres_en": genres_en,
                "movielens_genres_ru": genres_ru,
            }

    rating_sum = defaultdict(float)
    rating_count = defaultdict(int)
    with _open_dataset_csv(MOVIELENS_RATINGS_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie_id = _safe_int(row.get("movieId"))
            if movie_id is None:
                continue
            rating = _safe_float(row.get("rating"), 0.0)
            rating_sum[movie_id] += rating
            rating_count[movie_id] += 1

    tag_counters = defaultdict(Counter)
    with _open_dataset_csv(MOVIELENS_TAGS_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie_id = _safe_int(row.get("movieId"))
            if movie_id is None:
                continue
            tag = str(row.get("tag", "")).strip().lower()
            if len(tag) < 2:
                continue
            tag_counters[movie_id][tag] += 1

    by_imdb = {}
    by_tmdb = {}
    for movie_id, link in links_by_movie_id.items():
        avg_rating = rating_sum[movie_id] / rating_count[movie_id] if rating_count[movie_id] else 0.0
        movie_info = movie_info_by_id.get(movie_id, {})
        enrichment = {
            "movielens_movie_id": movie_id,
            "movielens_title": movie_info.get("movielens_title", ""),
            "movielens_genres_en": movie_info.get("movielens_genres_en", []),
            "movielens_genres_ru": movie_info.get("movielens_genres_ru", []),
            "movielens_rating": round(avg_rating, 3),
            "movielens_rating_count": rating_count[movie_id],
            "movielens_tags": [tag for tag, _ in tag_counters[movie_id].most_common(10)],
            "tmdb_id": link["tmdb_id"],
            "imdb_id": link["imdb_id"],
        }

        if link["imdb_id"]:
            by_imdb[link["imdb_id"]] = enrichment
        if link["tmdb_id"]:
            by_tmdb[link["tmdb_id"]] = enrichment

    return {"by_imdb": by_imdb, "by_tmdb": by_tmdb}


def _merge_record(base_record, themovies, movielens):
    merged = dict(base_record)

    themovies = themovies or {}
    movielens = movielens or {}
    localization = _pick_localization(base_record, themovies, movielens)
    tmdb_id = themovies.get("tmdb_id") or movielens.get("tmdb_id") or ""
    original_title = themovies.get("original_title", "")

    metadata_genres_ru = themovies.get("metadata_genres_ru", [])
    metadata_genres_en = themovies.get("metadata_genres_en", [])
    movielens_genres_ru = movielens.get("movielens_genres_ru", [])
    movielens_genres_en = movielens.get("movielens_genres_en", [])

    genres_ru = list(dict.fromkeys(base_record.get("genres_ru", []) + metadata_genres_ru + movielens_genres_ru))
    genres_en = list(dict.fromkeys(base_record.get("genres_en", []) + metadata_genres_en + movielens_genres_en))

    overview = themovies.get("overview", "")
    tagline = themovies.get("tagline", "")
    keywords = themovies.get("keywords", [])
    cast = themovies.get("cast", [])
    director = themovies.get("director", "")
    movielens_tags = movielens.get("movielens_tags", [])

    if not merged.get("release_year"):
        merged["release_year"] = themovies.get("metadata_release_year")

    imdb_rating = _safe_float(base_record.get("rating"), 0.0)
    movielens_rating = _safe_float(movielens.get("movielens_rating"), 0.0)
    if imdb_rating > 0 and movielens_rating > 0:
        merged["rating"] = round(imdb_rating * 0.7 + movielens_rating * 0.3, 3)
    elif movielens_rating > 0:
        merged["rating"] = movielens_rating
    else:
        merged["rating"] = imdb_rating

    title_ru = str(localization.get("title_ru", "")).strip()
    if not title_ru and _has_cyrillic(base_record.get("title", "")):
        title_ru = str(base_record.get("title", "")).strip()

    overview_ru = str(localization.get("overview_ru", "")).strip()
    tagline_ru = str(localization.get("tagline_ru", "")).strip()
    display_title = title_ru or base_record.get("title", "")
    display_full_title = _build_display_full_title(display_title, merged.get("release_year"))

    merged.update(
        {
            "tmdb_id": tmdb_id,
            "original_title": original_title,
            "normalized_original_title": _normalize_lookup_text(original_title),
            "title_ru": title_ru,
            "normalized_title_ru": _normalize_lookup_text(title_ru),
            "display_title": display_title,
            "display_full_title": display_full_title,
            "overview": overview,
            "overview_ru": overview_ru,
            "overview_display": overview_ru or overview,
            "tagline": tagline,
            "tagline_ru": tagline_ru,
            "tagline_display": tagline_ru or tagline,
            "keywords": keywords,
            "cast": cast,
            "director": director,
            "movielens_movie_id": movielens.get("movielens_movie_id"),
            "movielens_rating": movielens_rating,
            "movielens_rating_count": movielens.get("movielens_rating_count", 0),
            "movielens_tags": movielens_tags,
            "genres_ru": genres_ru,
            "genres_en": genres_en,
            "semantic_terms": list(dict.fromkeys(keywords + cast + movielens_tags)),
        }
    )
    return merged


def _base_movie_records():
    if not dataset_file_exists(MOVIE_DATASET_PATH):
        raise FileNotFoundError(
            "MovieGenre.csv не найден в data/raw. Добавьте датасет в проект."
        )

    records = []
    with _open_dataset_csv(MOVIE_DATASET_PATH) as file:
        reader = csv.DictReader(file)
        for row in reader:
            full_title = str(row.get("Title", "")).strip()
            if not full_title:
                continue

            clean_title = _clean_title(full_title)
            imdb_score = _safe_float(row.get("IMDB Score"), 0.0)
            genres_en = [item.strip() for item in str(row.get("Genre", "")).split("|") if item.strip()]
            genres_ru = _translated_genres(genres_en)
            imdb_id = _normalize_imdb_id(row.get("imdbId"))

            records.append(
                {
                    "imdb_id": imdb_id,
                    "title": clean_title,
                    "full_title": full_title,
                    "title_ru": clean_title if _has_cyrillic(clean_title) else "",
                    "display_title": clean_title,
                    "display_full_title": full_title,
                    "normalized_title": _normalize_lookup_text(clean_title),
                    "normalized_full_title": _normalize_lookup_text(full_title),
                    "normalized_title_ru": _normalize_lookup_text(clean_title if _has_cyrillic(clean_title) else ""),
                    "rating": imdb_score,
                    "raw_imdb_rating": imdb_score,
                    "genres_en": genres_en,
                    "genres_ru": genres_ru,
                    "release_year": _parse_release_year(full_title),
                    "poster_url": str(row.get("Poster", "")).strip(),
                    "imdb_link": str(row.get("Imdb Link", "")).strip(),
                }
            )

    return records


@lru_cache(maxsize=1)
def load_movie_metadata():
    required_paths = [
        MOVIE_DATASET_PATH,
        MOVIE_LOCALIZATIONS_PATH,
        THE_MOVIES_METADATA_PATH,
        THE_MOVIES_CREDITS_PATH,
        THE_MOVIES_KEYWORDS_PATH,
        MOVIELENS_MOVIES_PATH,
        MOVIELENS_RATINGS_PATH,
        MOVIELENS_TAGS_PATH,
        MOVIELENS_LINKS_PATH,
    ]
    signature = _source_signature(required_paths)
    cached = _load_cache(MERGED_METADATA_CACHE_PATH, signature)
    if cached is not None:
        return cached

    base_records = _base_movie_records()
    themovies = _load_themovies_enrichment()
    movielens = _load_movielens_enrichment()

    merged_records = []
    for record in base_records:
        imdb_id = record["imdb_id"]
        themovies_row = themovies["by_imdb"].get(imdb_id)
        if themovies_row is None and record.get("release_year") is not None:
            key = (record["normalized_title"], record["release_year"])
            themovies_row = themovies["by_title_year"].get(key)

        tmdb_id = themovies_row.get("tmdb_id") if themovies_row else ""
        movielens_row = movielens["by_imdb"].get(imdb_id)
        if movielens_row is None and tmdb_id:
            movielens_row = movielens["by_tmdb"].get(tmdb_id)

        merged_records.append(_merge_record(record, themovies_row, movielens_row))

    _save_cache(MERGED_METADATA_CACHE_PATH, signature, merged_records)
    return merged_records


def _title_match_score(candidate, record):
    normalized_candidate = _normalize_lookup_text(candidate)
    if not normalized_candidate:
        return 0.0

    exact_targets = {
        record["normalized_title"],
        record["normalized_full_title"],
        record.get("normalized_original_title", ""),
        record.get("normalized_title_ru", ""),
    }
    if normalized_candidate in exact_targets:
        return 1.0

    candidate_tokens = set(normalized_candidate.split())
    title_candidates = [
        record["normalized_title"],
        record["normalized_full_title"],
        record.get("normalized_original_title", ""),
        record.get("normalized_title_ru", ""),
    ]

    best_score = 0.0
    for target in title_candidates:
        if not target:
            continue
        record_tokens = set(target.split())
        token_overlap = len(candidate_tokens & record_tokens) / max(len(candidate_tokens), 1)
        sequence_score = SequenceMatcher(None, normalized_candidate, target).ratio()
        if normalized_candidate in target or target in normalized_candidate:
            sequence_score += 0.15
        best_score = max(best_score, token_overlap, sequence_score)

    return min(1.0, best_score)


def _is_plausible_title_match(normalized_candidate, record, score):
    candidate_tokens = normalized_candidate.split()
    title_targets = [
        record["normalized_title"],
        record["normalized_full_title"],
        record.get("normalized_original_title", ""),
        record.get("normalized_title_ru", ""),
    ]

    if normalized_candidate in title_targets:
        return True

    best_sequence = 0.0
    best_overlap = 0
    for target in title_targets:
        if not target:
            continue
        target_tokens = set(target.split())
        best_overlap = max(best_overlap, len(set(candidate_tokens) & target_tokens))
        best_sequence = max(best_sequence, SequenceMatcher(None, normalized_candidate, target).ratio())

    if best_overlap > 0 and score >= 0.62:
        return True
    if len(candidate_tokens) >= 2 and best_sequence >= 0.82:
        return True
    if len(normalized_candidate) >= 7 and best_sequence >= 0.9:
        return True
    return False


@lru_cache(maxsize=512)
def match_movie_title(candidate):
    normalized_candidate = _normalize_lookup_text(candidate)
    if len(normalized_candidate) < 2:
        return None

    best_record = None
    best_score = 0.0
    for record in load_movie_metadata():
        score = _title_match_score(normalized_candidate, record)
        if score > best_score:
            best_record = record
            best_score = score

    if best_record is None or best_score < 0.62:
        return None
    if not _is_plausible_title_match(normalized_candidate, best_record, best_score):
        return None

    return {
        "record": best_record,
        "score": round(best_score, 3),
    }
