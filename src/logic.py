import json
import os
import re
from functools import lru_cache

try:
    from .nlp_processor import analyze_text, build_nlp_summary
except ImportError:
    from nlp_processor import analyze_text, build_nlp_summary


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, "data", "raw", "rules.json")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
NODE_TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def check_rules(data):
    """Проверка сущности по продукционным правилам."""
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


def _append_nlp_summary(response, nlp_summary):
    if not nlp_summary:
        return response
    return f"{response}\n\n{nlp_summary}"


def _media_nodes(graph):
    return [node for node, attrs in graph.nodes(data=True) if attrs.get("type") == "media"]


def _format_rating(value):
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "-"


def _extract_candidate_terms(nlp_analysis):
    values = (
        nlp_analysis.get("people", [])
        + nlp_analysis.get("locations", [])
        + nlp_analysis.get("organizations", [])
        + nlp_analysis.get("lemmas", [])
    )
    result = []
    seen = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _node_tokens(text):
    return {token.lower() for token in NODE_TOKEN_PATTERN.findall(str(text))}


def _find_nodes_by_text_or_terms(graph, normalized_text, candidate_terms):
    normalized_nodes = {str(node).lower(): node for node in graph.nodes}
    fuzzy_stop_terms = {
        "фильм", "книга", "после", "раньше", "про", "хотеть",
        "нравиться", "посоветовать", "смотреть", "подобрать",
    }

    found_nodes = []
    for node_key, node in normalized_nodes.items():
        if node_key and node_key in normalized_text:
            found_nodes.append(node)

    for term in candidate_terms:
        if term in normalized_nodes:
            found_nodes.append(normalized_nodes[term])
            continue

        if len(term) < 4 or term in fuzzy_stop_terms:
            continue

        for node_key, node in normalized_nodes.items():
            node_terms = _node_tokens(node_key)
            token_match = any(
                token == term
                or (
                    min(len(token), len(term)) >= 4
                    and (token.startswith(term) or term.startswith(token))
                )
                for token in node_terms
            )
            if token_match:
                found_nodes.append(node)

    unique_nodes = []
    seen = set()
    for node in found_nodes:
        marker = str(node).lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique_nodes.append(node)

    return unique_nodes


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


def _passes_year_filter(media_attrs, year_filter):
    if not year_filter:
        return True

    year = media_attrs.get("release_year")
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


def _normalize_query_lemmas(nlp_analysis):
    stopwords = {
        "это", "как", "или", "что", "который", "мочь", "быть", "мой", "твой",
        "фильм", "книга", "посоветовать", "нравиться", "хотеть", "нужно",
    }

    normalized = set()
    for lemma in nlp_analysis.get("lemmas", []):
        item = lemma.strip().lower()
        if len(item) < 3 or item in stopwords:
            continue
        normalized.add(item)
    return normalized


@lru_cache(maxsize=256)
def _description_lemmas(description):
    if not isinstance(description, str) or not description.strip():
        return set()

    parsed = analyze_text(description)
    return {lemma.strip().lower() for lemma in parsed.get("lemmas", []) if lemma.strip()}


def _description_overlap(query_lemmas, description):
    if not query_lemmas:
        return []
    overlap = query_lemmas & _description_lemmas(description)
    return sorted(overlap)


def _recommend_similar_media(graph, source_media, query_lemmas, year_filter, limit=3):
    source_neighbors = set(graph.neighbors(source_media))
    recommendations = []

    for candidate in _media_nodes(graph):
        if candidate == source_media:
            continue

        candidate_attrs = graph.nodes[candidate]
        if not _passes_year_filter(candidate_attrs, year_filter):
            continue

        candidate_neighbors = set(graph.neighbors(candidate))
        shared_features = sorted(str(item) for item in source_neighbors & candidate_neighbors)
        overlap_terms = _description_overlap(query_lemmas, candidate_attrs.get("description", ""))

        if not shared_features and not overlap_terms:
            continue

        rating = float(candidate_attrs.get("rating", 0.0))
        score = len(shared_features) * 10.0 + len(overlap_terms) * 4.0 + rating
        recommendations.append(
            (
                score,
                rating,
                candidate,
                shared_features,
                overlap_terms,
                candidate_attrs.get("release_year"),
            )
        )

    recommendations.sort(reverse=True)
    return recommendations[:limit]


def _recommend_by_feature(graph, feature_node, query_lemmas, year_filter, limit=3):
    media_candidates = []
    for neighbor in graph.neighbors(feature_node):
        attrs = graph.nodes[neighbor]
        if attrs.get("type") != "media":
            continue
        if not _passes_year_filter(attrs, year_filter):
            continue

        overlap_terms = _description_overlap(query_lemmas, attrs.get("description", ""))
        rating = float(attrs.get("rating", 0.0))
        score = rating + len(overlap_terms) * 2.0
        media_candidates.append((score, rating, neighbor, overlap_terms, attrs.get("release_year")))

    media_candidates.sort(reverse=True)
    return media_candidates[:limit]


def _semantic_search_media(graph, query_lemmas, year_filter, limit=3):
    if not query_lemmas:
        return []

    results = []
    for media_node in _media_nodes(graph):
        attrs = graph.nodes[media_node]
        if not _passes_year_filter(attrs, year_filter):
            continue

        description = attrs.get("description", "")
        overlap_desc = _description_overlap(query_lemmas, description)
        overlap_title = sorted(query_lemmas & _node_tokens(media_node))
        overlap_terms = sorted(set(overlap_desc + overlap_title))
        if not overlap_terms:
            continue

        rating = float(attrs.get("rating", 0.0))
        score = len(overlap_terms) * 5.0 + rating
        results.append((score, rating, media_node, overlap_terms, attrs.get("release_year")))

    results.sort(reverse=True)
    return results[:limit]


def _build_media_response(graph, media_node, query_lemmas, year_filter):
    media_info = graph.nodes[media_node]
    description = media_info.get("description", "")
    release_year = media_info.get("release_year")

    year_text = f", {release_year}" if isinstance(release_year, int) else ""
    base = [f"Опорный фильм/книга: {media_node}{year_text} (рейтинг {_format_rating(media_info.get('rating'))})."]

    if isinstance(description, str) and description.strip():
        base.append(f"Описание: {description}")

    year_filter_info = _year_filter_text(year_filter)
    if year_filter_info:
        base.append(year_filter_info)

    recommendations = _recommend_similar_media(graph, media_node, query_lemmas, year_filter)
    if not recommendations:
        base.append("Не нашел похожие варианты под текущие условия запроса.")
        return "\n".join(base)

    base.append("Рекомендации:")
    for index, (_, rating, candidate, shared_features, overlap_terms, candidate_year) in enumerate(recommendations, start=1):
        reasons = []
        if shared_features:
            reasons.append(f"общие связи: {', '.join(shared_features)}")
        if overlap_terms:
            reasons.append(f"семантика запроса: {', '.join(overlap_terms)}")

        reason_text = "; ".join(reasons) if reasons else "схожесть по графу"
        year_suffix = f", {candidate_year}" if isinstance(candidate_year, int) else ""
        base.append(f"{index}. {candidate}{year_suffix} (рейтинг {_format_rating(rating)}; {reason_text})")

    return "\n".join(base)


def _build_feature_response(graph, feature_node, query_lemmas, year_filter):
    feature_type = graph.nodes[feature_node].get("type")
    title = "по жанру" if feature_type == "genre" else "по автору"

    recommendations = _recommend_by_feature(graph, feature_node, query_lemmas, year_filter)
    if not recommendations:
        return f"По узлу '{feature_node}' нет фильмов/книг под текущие условия запроса."

    lines = [f"Подборка {title} '{feature_node}':"]
    year_filter_info = _year_filter_text(year_filter)
    if year_filter_info:
        lines.append(year_filter_info)

    for index, (_, rating, media_node, overlap_terms, release_year) in enumerate(recommendations, start=1):
        semantic_text = f"; семантика: {', '.join(overlap_terms)}" if overlap_terms else ""
        year_suffix = f", {release_year}" if isinstance(release_year, int) else ""
        lines.append(f"{index}. {media_node}{year_suffix} (рейтинг {_format_rating(rating)}{semantic_text})")

    return "\n".join(lines)


def _build_semantic_response(matches, year_filter):
    lines = ["Нашел подходящие фильмы/книги по смыслу запроса:"]

    year_filter_info = _year_filter_text(year_filter)
    if year_filter_info:
        lines.append(year_filter_info)

    for index, (_, rating, media_node, overlap_terms, release_year) in enumerate(matches, start=1):
        year_suffix = f", {release_year}" if isinstance(release_year, int) else ""
        lines.append(
            f"{index}. {media_node}{year_suffix} (рейтинг {_format_rating(rating)}; "
            f"совпадения: {', '.join(overlap_terms)})"
        )

    return "\n".join(lines)


def process_text_message(text, data_source):
    """
    Обрабатывает текст пользователя для темы StoryGuide:
    определяет сущности (NLP) и предлагает фильмы/книги из графа.
    """
    text = text if isinstance(text, str) else ""
    normalized_text = text.strip().lower()

    if not normalized_text:
        return "Напишите запрос по теме фильмов/книг: название, жанр, автора или условие по году."

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

    nlp_summary = build_nlp_summary(nlp_analysis)
    query_lemmas = _normalize_query_lemmas(nlp_analysis)
    year_filter = _extract_year_filter(normalized_text, nlp_analysis)

    if "привет" in normalized_text:
        response = (
            "Привет! Я советчик StoryGuide. "
            "Напиши, что тебе нравится, и при желании добавь год: "
            "например 'Хочу триллер после 2010'."
        )
        return _append_nlp_summary(response, nlp_summary)

    if data_source is None or not hasattr(data_source, "nodes") or not hasattr(data_source, "neighbors"):
        response = "Граф рекомендаций недоступен. Невозможно подобрать фильмы/книги."
        return _append_nlp_summary(response, nlp_summary)

    candidate_terms = _extract_candidate_terms(nlp_analysis)
    mentioned_nodes = _find_nodes_by_text_or_terms(data_source, normalized_text, candidate_terms)

    media_node = next(
        (node for node in mentioned_nodes if data_source.nodes[node].get("type") == "media"),
        None,
    )
    feature_node = next(
        (
            node
            for node in mentioned_nodes
            if data_source.nodes[node].get("type") in {"genre", "creator"}
        ),
        None,
    )

    if media_node is not None:
        response = _build_media_response(data_source, media_node, query_lemmas, year_filter)
        return _append_nlp_summary(response, nlp_summary)

    if feature_node is not None:
        response = _build_feature_response(data_source, feature_node, query_lemmas, year_filter)
        return _append_nlp_summary(response, nlp_summary)

    semantic_matches = _semantic_search_media(data_source, query_lemmas, year_filter)
    if semantic_matches:
        response = _build_semantic_response(semantic_matches, year_filter)
        return _append_nlp_summary(response, nlp_summary)

    response = (
        "Не нашел подходящий вариант в графе. "
        "Попробуйте точнее: 'Мне нравится Начало', 'посоветуй фантастику после 2010', "
        "'хочу фильм про сон и подсознание'."
    )
    return _append_nlp_summary(response, nlp_summary)
