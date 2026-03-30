import json
import os
import re
from functools import lru_cache

from src.dataset_loader import load_movie_metadata
from src.movie_recommender import GENRE_KEYWORDS


FOLLOW_UP_MARKERS = (
    "после",
    "до",
    "между",
    "без",
    "кроме",
    "только",
    "ещё",
    "еще",
    "поновее",
    "новее",
    "старее",
    "похоже",
    "похож",
    "подоб",
    "другое",
    "другой",
    "этот",
    "этого",
    "такой",
    "такого",
    "выдай",
    "подбери",
    "посоветуй",
    "список",
)

GENRE_FORMS = {
    "боевик": ("боевик", "боевика", "боевики"),
    "приключения": ("приключения", "приключений"),
    "анимация": ("анимация", "анимации", "мультфильм", "мультфильма", "мультфильмы"),
    "комедия": ("комедия", "комедии", "комедию", "комедий"),
    "криминал": ("криминал", "криминала"),
    "драма": ("драма", "драмы", "драму"),
    "семейный": ("семейный", "семейного"),
    "фэнтези": ("фэнтези",),
    "ужасы": ("ужасы", "ужасов"),
    "детектив": ("детектив", "детектива"),
    "романтика": ("романтика", "романтики"),
    "фантастика": ("фантастика", "фантастики", "фантастику"),
    "триллер": ("триллер", "триллера"),
}


def default_conversation_state():
    return {
        "anchor_movie": None,
        "filters": {
            "year_filter": {},
            "include_genres": [],
            "exclude_genres": [],
        },
        "result_limit": 5,
        "last_query": "",
        "turn_count": 0,
    }


def normalize_conversation_state(raw_state):
    state = default_conversation_state()
    if not isinstance(raw_state, dict):
        return state

    state["anchor_movie"] = raw_state.get("anchor_movie")
    filters = raw_state.get("filters", {})
    if isinstance(filters, dict):
        state["filters"]["year_filter"] = dict(filters.get("year_filter") or {})
        state["filters"]["include_genres"] = list(filters.get("include_genres") or [])
        state["filters"]["exclude_genres"] = list(filters.get("exclude_genres") or [])
    state["result_limit"] = int(raw_state.get("result_limit", 5) or 5)
    state["last_query"] = str(raw_state.get("last_query", "") or "")
    state["turn_count"] = int(raw_state.get("turn_count", 0) or 0)
    return state


def get_state_path(base_dir, session_id):
    state_dir = os.path.join(base_dir, "data", "processed", "conversation_states")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"{session_id}.json")


def load_conversation_state(path):
    if not os.path.exists(path):
        return default_conversation_state()

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return default_conversation_state()

    return normalize_conversation_state(data)


def save_conversation_state(path, state):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(normalize_conversation_state(state), file, ensure_ascii=False, indent=2)


@lru_cache(maxsize=1)
def _records_by_imdb_id():
    return {
        str(record.get("imdb_id", "")).strip(): record
        for record in load_movie_metadata()
        if str(record.get("imdb_id", "")).strip()
    }


def get_movie_by_imdb_id(imdb_id):
    return _records_by_imdb_id().get(str(imdb_id or "").strip())


def build_anchor_payload(record):
    if not record:
        return None
    return {
        "imdb_id": record.get("imdb_id"),
        "title": record.get("title"),
        "display_full_title": record.get("display_full_title", record.get("full_title")),
    }


def extract_genre_preferences(text):
    normalized_text = str(text or "").lower()
    exclude_genres = []
    for genre, forms in GENRE_FORMS.items():
        if any(
            re.search(rf"\b(?:без|кроме)\s+{re.escape(form)}\b", normalized_text)
            for form in forms
        ):
            exclude_genres.append(genre)

    include_genres = []
    for genre, forms in GENRE_FORMS.items():
        if genre in exclude_genres:
            continue
        if any(form in normalized_text for form in forms):
            include_genres.append(genre)

    return {
        "include_genres": list(dict.fromkeys(include_genres)),
        "exclude_genres": list(dict.fromkeys(exclude_genres)),
    }


def merge_year_filters(existing_filter, new_filter):
    existing_filter = dict(existing_filter or {})
    new_filter = dict(new_filter or {})
    if not existing_filter:
        return new_filter
    if not new_filter:
        return existing_filter

    if "exact_year" in new_filter:
        return {"exact_year": new_filter["exact_year"]}
    if "exact_year" in existing_filter:
        exact_year = existing_filter["exact_year"]
        converted = {"min_year": exact_year, "max_year": exact_year}
    else:
        converted = dict(existing_filter)

    min_year = converted.get("min_year")
    max_year = converted.get("max_year")
    if new_filter.get("min_year") is not None:
        min_year = max(min_year, new_filter["min_year"]) if min_year is not None else new_filter["min_year"]
    if new_filter.get("max_year") is not None:
        max_year = min(max_year, new_filter["max_year"]) if max_year is not None else new_filter["max_year"]

    merged = {}
    if min_year is not None:
        merged["min_year"] = min_year
    if max_year is not None:
        merged["max_year"] = max_year
    return merged


def merge_filter_lists(existing_values, new_values):
    return list(dict.fromkeys(list(existing_values or []) + list(new_values or [])))


def has_follow_up_markers(text):
    normalized_text = str(text or "").lower()
    return any(marker in normalized_text for marker in FOLLOW_UP_MARKERS)


def has_active_filters(state):
    state = normalize_conversation_state(state)
    filters = state["filters"]
    return bool(filters["year_filter"] or filters["include_genres"] or filters["exclude_genres"])


def infer_year_filter_from_relative_follow_up(text, anchor_record, current_year_filter=None):
    normalized_text = str(text or "").lower()
    if current_year_filter:
        return {}
    if not anchor_record:
        return {}

    release_year = anchor_record.get("release_year")
    if not isinstance(release_year, int):
        return {}

    if any(marker in normalized_text for marker in ("поновее", "новее", "посвежее")):
        return {"min_year": release_year + 1}
    if any(marker in normalized_text for marker in ("постарее", "старее", "пораньше")):
        return {"max_year": release_year - 1}
    return {}


def build_context_summary(state):
    state = normalize_conversation_state(state)
    anchor_movie = state.get("anchor_movie")
    filters = state["filters"]

    lines = []
    if anchor_movie:
        lines.append(f"Фильм-опора: {anchor_movie['display_full_title']}")
    else:
        lines.append("Фильм-опора: пока не выбран")

    year_filter = filters["year_filter"]
    if year_filter:
        if "exact_year" in year_filter:
            lines.append(f"Год: {year_filter['exact_year']}")
        else:
            parts = []
            if year_filter.get("min_year") is not None:
                parts.append(f"от {year_filter['min_year']}")
            if year_filter.get("max_year") is not None:
                parts.append(f"до {year_filter['max_year']}")
            lines.append(f"Годы: {' '.join(parts)}")
    else:
        lines.append("Годы: без ограничений")

    lines.append(f"Количество рекомендаций: {state.get('result_limit', 5)}")

    if filters["include_genres"]:
        lines.append(f"Включить жанры: {', '.join(filters['include_genres'])}")
    if filters["exclude_genres"]:
        lines.append(f"Исключить жанры: {', '.join(filters['exclude_genres'])}")
    if state.get("last_query"):
        lines.append(f"Последний запрос: {state['last_query']}")
    lines.append(f"Ходов в теме: {state['turn_count']}")
    return lines
