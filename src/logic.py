import json
import os
import re
from functools import lru_cache

try:
    from .movie_recommender import search_movies_by_query
    from .nlp_processor import analyze_text
except ImportError:
    from movie_recommender import search_movies_by_query
    from nlp_processor import analyze_text


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, "data", "raw", "rules.json")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def check_rules(data):
    rules = load_rules()

    if rules["critical_rules"]["must_be_verified"] and not data["is_verified"]:
        return "Критическая ошибка: профиль не подтвержден."
    if data["release_year"] < rules["thresholds"]["min_year"]:
        return "Отказ: год релиза ниже минимального порога."
    if data["release_year"] > rules["thresholds"]["max_year"]:
        return "Отказ: год релиза выше максимального порога."

    for genre in data["genres"]:
        if genre in rules["lists"]["blacklist"]:
            return f"Предупреждение: найден запрещенный жанр ({genre})."

    return f"Успех: объект соответствует сценарию '{rules['scenario_name']}'."


def _format_rating(value):
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "-"


def _extract_years_from_analysis(nlp_analysis):
    years = set()
    for raw_date in nlp_analysis.get("dates", []):
        for match in YEAR_PATTERN.findall(raw_date):
            years.add(int(match))
    return sorted(years)


def _extract_year_filter(normalized_text, nlp_analysis):
    years = set(_extract_years_from_analysis(nlp_analysis))
    for match in YEAR_PATTERN.findall(normalized_text):
        years.add(int(match))
    years = sorted(years)
    if not years:
        return {}

    after_markers = ("после", "новее", "позже", "свежее")
    before_markers = ("до", "раньше", "старее", "старше")

    if "между" in normalized_text and len(years) >= 2:
        return {"min_year": min(years), "max_year": max(years)}
    if any(marker in normalized_text for marker in after_markers):
        return {"min_year": max(years)}
    if any(marker in normalized_text for marker in before_markers):
        return {"max_year": min(years)}
    if len(years) >= 2 and "с" in normalized_text and "по" in normalized_text:
        return {"min_year": min(years), "max_year": max(years)}
    return {"exact_year": years[0]}


def _year_filter_text(year_filter):
    if not year_filter:
        return ""
    if "exact_year" in year_filter:
        return f"Фильтр по году: {year_filter['exact_year']}."

    min_year = year_filter.get("min_year")
    max_year = year_filter.get("max_year")
    if min_year is not None and max_year is not None:
        return f"Фильтр по годам: {min_year}-{max_year}."
    if min_year is not None:
        return f"Фильтр по годам: после {min_year}."
    if max_year is not None:
        return f"Фильтр по годам: до {max_year}."
    return ""


def _append_movie_description(lines, record):
    release_year = record.get("release_year")
    rating = _format_rating(record.get("rating"))
    summary_parts = []
    if isinstance(release_year, int):
        summary_parts.append(f"Год: {release_year}")
    if rating != "-":
        summary_parts.append(f"Рейтинг: {rating}")
    if summary_parts:
        lines.append(". ".join(summary_parts) + ".")

    tagline = str(record.get("tagline_display", "")).strip()
    if tagline:
        lines.append(f"Слоган: {tagline}")

    overview = str(record.get("overview_display", "")).strip()
    if overview:
        lines.append(f"Описание: {overview}")


def _build_title_match_response(search_result, year_filter):
    source = search_result["source"]
    matches = search_result["matches"]

    lines = [f"Похоже, вам нравится {source.get('display_full_title', source['full_title'])}."]
    _append_movie_description(lines, source)

    source_genres = source.get("genres_ru", [])
    if source_genres:
        lines.append(f"Жанры: {', '.join(source_genres)}.")

    year_filter_info = _year_filter_text(year_filter)
    if year_filter_info:
        lines.append(year_filter_info)

    if not matches:
        lines.append("По текущим условиям похожие фильмы не нашлись.")
        return "\n".join(lines)

    lines.append("Вот что можно посмотреть дальше:")
    for index, item in enumerate(matches, start=1):
        record = item["record"]
        candidate_genres = ", ".join(record.get("genres_ru", [])[:3])
        lines.append(
            f"{index}. {record.get('display_full_title', record['full_title'])} — "
            f"рейтинг {_format_rating(record.get('rating'))}, {candidate_genres}"
        )

    return "\n".join(lines)


def _build_hybrid_response(search_result, year_filter):
    matches = search_result["matches"]
    detected_genres = search_result.get("detected_genres", [])

    lines = ["Вот что нашел по вашему описанию:"]
    if detected_genres:
        lines.append(f"Под запрос хорошо подходят жанры: {', '.join(detected_genres)}.")

    year_filter_info = _year_filter_text(year_filter)
    if year_filter_info:
        lines.append(year_filter_info)

    for index, item in enumerate(matches, start=1):
        record = item["record"]
        candidate_genres = ", ".join(record.get("genres_ru", [])[:3])
        lines.append(
            f"{index}. {record.get('display_full_title', record['full_title'])} — "
            f"рейтинг {_format_rating(record.get('rating'))}, {candidate_genres}"
        )

    return "\n".join(lines)


def process_text_message(text, data_source=None):
    text = text if isinstance(text, str) else ""
    normalized_text = text.strip().lower()

    if not normalized_text:
        return "Напишите, какой фильм вам нравится, либо опишите желаемый сюжет."

    try:
        nlp_analysis = analyze_text(text)
    except ModuleNotFoundError as error:
        module_name = getattr(error, "name", "") or "unknown"
        return (
            "Ошибка NLP-окружения: отсутствует модуль "
            f"'{module_name}'. Установите зависимости командой "
            "'pip install -r requirements.txt' и перезапустите приложение."
        )
    except (RuntimeError, AttributeError) as error:
        return (
            "Ошибка совместимости NLP-стека. "
            f"Детали: {error}. Проверьте зависимости в venv и перезапустите приложение."
        )

    year_filter = _extract_year_filter(normalized_text, nlp_analysis)

    if "привет" in normalized_text:
        response = (
            "Привет! Я рекомендую фильмы по названию, жанрам и похожести запроса. "
            "Можно написать 'Мне нравится Toy Story', 'хочу триллер после 2010' "
            "или загрузить постер в чат."
        )
        return response

    search_result = search_movies_by_query(text, year_filter=year_filter, limit=5)
    mode = search_result.get("mode")

    if mode == "title_match":
        response = _build_title_match_response(search_result, year_filter)
        return response

    if mode == "hybrid_query" and search_result.get("matches"):
        response = _build_hybrid_response(search_result, year_filter)
        return response

    response = (
        "Не нашел подходящий фильм в текущем каталоге. "
        "Попробуйте точнее указать название, жанр или год."
    )
    return response
