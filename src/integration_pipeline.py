import re
from pathlib import Path

from src.conversation_state import (
    build_anchor_payload,
    default_conversation_state,
    extract_genre_preferences,
    get_movie_by_imdb_id,
    has_active_filters,
    has_follow_up_markers,
    infer_year_filter_from_relative_follow_up,
    merge_filter_lists,
    merge_year_filters,
    normalize_conversation_state,
)
from src.dataset_loader import match_movie_title
from src.intent_router import classify_intent, extract_requested_limit
from src.logic import analyze_query, build_recommendation_response, describe_active_filters, load_rules
from src.movie_recommender import recommend_similar_movies, search_movies_by_query
from src.vision_processor import analyze_uploaded_image


IMAGE_TYPES = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
TEXT_TYPES = {"txt", "md", "json", "csv"}
MAX_TEXT_CHARS = 2400
QUOTED_TITLE_PATTERN = re.compile(r"[\"'«“](.+?)[\"'»”]")
TITLE_CAPTURE_PATTERNS = (
    re.compile(
        r"(?:^|\b)(?:мне\s+нрав(?:ится|ился|илась|илось)|люблю|обожаю|зацепил(?:о)?|"
        r"заш(?:е|ё)л|понрав(?:ился|илась|илось)|нравится)\s+(?P<title>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\b)(?:я\s+)?(?:недавно\s+|только\s+что\s+|вчера\s+)?"
        r"(?:посмотрел|посмотрела|смотрел|смотрела|глянул|глянула|видел|видела)\s+(?P<title>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\b)(?:дай|дайте|выдай|выдайте|подбери|посоветуй|найди)\s+(?:мне\s+)?"
        r"(?:список\s+)?(?:\d+\s+)?(?:фильмов?\s+)?(?:похож(?:их|ие|ий|ее)|подобн(?:ых|ые|ый|ое)|как)\s+"
        r"(?:на\s+)?(?P<title>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\b)(?:хочу\s+)?(?:что[-\s]?нибудь\s+)?(?:похож(?:ее|ий|ие|их)|подобн(?:ое|ый|ые|ых))\s+"
        r"(?:на\s+)?(?P<title>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\b)(?:наподобие)\s+(?P<title>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\b)(?:\d+\s+)?(?:фильмы|фильм|кино)\s+(?:похож(?:ие|их|ий|ее)|как)\s+(?:на\s+)?(?P<title>.+)$",
        flags=re.IGNORECASE,
    ),
)
TITLE_NOISE_PREFIXES = (
    "фильм ",
    "кино ",
    "список фильмов ",
    "список ",
)
TITLE_NOISE_SUFFIX_PATTERN = re.compile(
    r"\s+(?:и\s+)?(?:хочу|хочется|выдай|дай|подбери|посоветуй|найди|покажи|список|"
    r"похожие|похожее|похожий|похожих|что|ещ[её])\b.*$",
    flags=re.IGNORECASE,
)
ANCHOR_REFERENCE_PATTERN = re.compile(
    r"\b(?:этот|этого|этому|этом|этим|такой|такого|такому|таком|него|неё|нее)\b",
    flags=re.IGNORECASE,
)
ANCHOR_REFERENCE_ONLY_PATTERN = re.compile(
    r"^(?:этот|этого|этому|этом|этим|такой|такого|такому|таком|него|неё|нее)(?:\s+фильм)?$",
    flags=re.IGNORECASE,
)
SIMILAR_REQUEST_PATTERN = re.compile(
    r"\b(?:похож|подоб|список|выдай|подбери|посоветуй|что\s+посмотреть|что\s+еще)\b",
    flags=re.IGNORECASE,
)


def file_extension(file_name):
    return Path(file_name).suffix.lower().lstrip(".")


def extract_text_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Файл '{uploaded_file.name}' не удалось прочитать как UTF-8 текст."
        ) from error

    return content.strip()


def _truncate(text, limit=MAX_TEXT_CHARS):
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _merge_query_parts(parts):
    result = []
    seen = set()
    for part in parts:
        normalized = str(part or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return ". ".join(result)


def _filter_matches_by_rules(search_result):
    rules = load_rules()
    thresholds = rules.get("thresholds", {})
    min_year = thresholds.get("min_year")
    max_year = thresholds.get("max_year")
    blacklist = {item.strip().lower() for item in rules.get("lists", {}).get("blacklist", []) if item}

    kept_matches = []
    removed_out_of_range = 0
    removed_blacklist = 0

    for item in search_result.get("matches", []):
        record = item["record"]
        year = record.get("release_year")
        genres = {genre.strip().lower() for genre in record.get("genres_ru", []) if genre}

        out_of_range = (
            (isinstance(min_year, int) and isinstance(year, int) and year < min_year)
            or (isinstance(max_year, int) and isinstance(year, int) and year > max_year)
        )
        blacklisted = bool(genres & blacklist)

        if out_of_range:
            removed_out_of_range += 1
            continue
        if blacklisted:
            removed_blacklist += 1
            continue
        kept_matches.append(item)

    filtered = dict(search_result)
    filtered["matches"] = kept_matches
    return filtered, {
        "catalog_min_year": min_year,
        "catalog_max_year": max_year,
        "removed_out_of_range": removed_out_of_range,
        "removed_blacklist": removed_blacklist,
    }


def _summarize_nlp(query_analysis):
    details = []
    nlp_analysis = query_analysis["nlp_analysis"]
    if query_analysis["nlp_warning"]:
        details.append(query_analysis["nlp_warning"])
    if nlp_analysis.get("dates"):
        details.append(f"обнаружены даты: {', '.join(nlp_analysis['dates'])}")
    if nlp_analysis.get("people"):
        details.append(f"персоны: {', '.join(nlp_analysis['people'][:3])}")
    if nlp_analysis.get("locations"):
        details.append(f"локации: {', '.join(nlp_analysis['locations'][:3])}")
    if query_analysis["year_filter"]:
        details.append(f"выделен фильтр по году: {query_analysis['year_filter']}")
    return "; ".join(details) if details else "явные сущности и ограничения не найдены"


def _summarize_rules(rule_report):
    parts = [
        f"базовый каталог ограничен {rule_report['catalog_min_year']}-{rule_report['catalog_max_year']}",
        f"исключено вне диапазона: {rule_report['removed_out_of_range']}",
        f"исключено по blacklist-жанрам: {rule_report['removed_blacklist']}",
    ]
    return "; ".join(parts)


def _build_pipeline_header(input_summary, signal_lines, combined_query, query_analysis, rule_report):
    lines = [
        "**Пайплайн 9-10**",
        f"- Вход: {input_summary}",
    ]
    for signal_line in signal_lines:
        lines.append(f"- Сигнал: {signal_line}")
    lines.append(f"- NLP: {_summarize_nlp(query_analysis)}")
    lines.append(f"- Правила: {_summarize_rules(rule_report)}")
    lines.append(f"- Итоговый запрос: {combined_query}")
    return "\n".join(lines)


def _extract_title_candidates(text, include_full_text=True):
    normalized = str(text or "").strip()
    if not normalized:
        return []

    candidates = []
    candidates.extend(match.group(1).strip() for match in QUOTED_TITLE_PATTERN.finditer(normalized))
    if include_full_text:
        candidates.append(normalized)

    for pattern in TITLE_CAPTURE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            candidates.append(match.group("title").strip())

    cleaned_candidates = []
    seen = set()
    for candidate in candidates:
        value = str(candidate or "").strip(" .,!?:;")
        value = re.split(r"[,.!?;]", value, maxsplit=1)[0].strip()
        value = TITLE_NOISE_SUFFIX_PATTERN.sub("", value).strip(" .,!?:;")
        lowered = value.lower()
        for prefix in TITLE_NOISE_PREFIXES:
            if lowered.startswith(prefix):
                value = value[len(prefix):].strip()
                lowered = value.lower()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_candidates.append(value)

    return cleaned_candidates


def _extract_title_hint(text):
    cleaned_candidates = _extract_title_candidates(text, include_full_text=True)

    best_title = ""
    best_score = 0.0
    for candidate in cleaned_candidates:
        title_match = match_movie_title(candidate)
        if title_match is None:
            continue
        if title_match["score"] > best_score:
            best_title = candidate
            best_score = title_match["score"]

    return best_title


def _references_current_anchor(text):
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return bool(ANCHOR_REFERENCE_PATTERN.search(normalized) and SIMILAR_REQUEST_PATTERN.search(normalized))


def _is_filter_only_follow_up(text, state):
    if not state.get("anchor_movie"):
        return False

    normalized_text = str(text or "").strip().lower()
    if not normalized_text:
        return False

    if has_follow_up_markers(normalized_text):
        return True

    return False


def _resolve_explicit_anchor(cleaned_user_text, combined_query, signal_records, allow_text_anchor=True):
    for record in signal_records:
        if record is not None:
            return record

    if not allow_text_anchor:
        return None

    title_hint = _extract_title_hint(cleaned_user_text) or _extract_title_hint(combined_query)
    if not title_hint:
        return None

    title_match = match_movie_title(title_hint)
    if title_match is None:
        return None
    return title_match["record"]


def _extract_explicit_title_request(cleaned_user_text, combined_query):
    candidates = _extract_title_candidates(cleaned_user_text, include_full_text=False)
    if not candidates:
        candidates = _extract_title_candidates(combined_query, include_full_text=False)
    for candidate in candidates:
        if not ANCHOR_REFERENCE_ONLY_PATTERN.fullmatch(candidate.strip().lower()):
            return candidate
    return ""


def _build_search_result_from_anchor(anchor_record, filters, limit=5):
    return {
        "mode": "title_match",
        "source": anchor_record,
        "match_score": 1.0,
        "matches": recommend_similar_movies(
            anchor_record,
            year_filter=filters["year_filter"],
            include_genres=filters["include_genres"],
            exclude_genres=filters["exclude_genres"],
            limit=limit,
        ),
    }


def _build_recommendation_rows(search_result):
    rows = []
    for index, item in enumerate(search_result.get("matches", []), start=1):
        record = item["record"]
        rows.append(
            {
                "rank": index,
                "title": record.get("display_full_title", record.get("full_title", record.get("title", ""))),
                "year": record.get("release_year"),
                "rating": record.get("rating"),
                "genres": ", ".join(record.get("genres_ru", [])[:4]),
                "score": item.get("score"),
            }
        )
    return rows


def _build_ui_payload(
    input_summary,
    input_kinds,
    signal_lines,
    combined_query,
    intent_report,
    query_analysis,
    requested_limit,
    active_filters_text,
    search_result,
    filtered_search_result,
    rule_report,
    state,
):
    source_record = filtered_search_result.get("source") or search_result.get("source")
    source_movie = None
    if source_record is not None:
        source_movie = {
            "title": source_record.get("display_full_title", source_record.get("full_title", source_record.get("title", ""))),
            "year": source_record.get("release_year"),
            "rating": source_record.get("rating"),
            "genres": ", ".join(source_record.get("genres_ru", [])[:4]),
        }

    recommendation_rows = _build_recommendation_rows(filtered_search_result)
    return {
        "input_summary": input_summary,
        "input_kinds": list(input_kinds),
        "signal_lines": list(signal_lines),
        "combined_query": combined_query,
        "intent": intent_report.get("intent", "unknown"),
        "intent_score": intent_report.get("score", 0.0),
        "requested_limit": requested_limit,
        "active_filters_text": active_filters_text,
        "search_mode": filtered_search_result.get("mode", search_result.get("mode", "unknown")),
        "source_movie": source_movie,
        "recommendation_rows": recommendation_rows,
        "metrics": {
            "recommendation_count": len(recommendation_rows),
            "requested_limit": requested_limit,
            "signal_count": len(signal_lines),
            "turn_count": state.get("turn_count", 0),
            "has_anchor": bool(state.get("anchor_movie")),
        },
        "rule_report": dict(rule_report),
        "nlp": {
            "year_filter": dict(query_analysis.get("year_filter") or {}),
            "warning": query_analysis.get("nlp_warning", ""),
            "summary": _summarize_nlp(query_analysis),
        },
    }


def _compose_filter_only_reply():
    return (
        "Я увидел уточняющий запрос, но пока не знаю, от какого фильма отталкиваться. "
        "Сначала назовите фильм или загрузите постер, а затем можно уточнять год и жанры."
    )


def run_integrated_pipeline(
    user_text="",
    uploaded_files=None,
    conversation_state=None,
    include_trace=False,
):
    uploaded_files = list(uploaded_files or [])
    cleaned_user_text = str(user_text or "").strip()
    state = normalize_conversation_state(conversation_state or default_conversation_state())

    if not cleaned_user_text and not uploaded_files:
        return {
            "response": "Напишите, какой фильм вам нравится, либо опишите желаемый сюжет.",
            "conversation_state": state,
        }

    signal_lines = []
    query_parts = [cleaned_user_text]
    input_kinds = []
    signal_anchor_records = []

    for uploaded_file in uploaded_files:
        extension = file_extension(uploaded_file.name)

        if extension in IMAGE_TYPES:
            image_analysis = analyze_uploaded_image(uploaded_file.getvalue(), uploaded_file.name)
            extracted_data = image_analysis["extracted_data"]
            signal_lines.append(image_analysis["summary"])
            recognition_label = extracted_data.get("recognition_label", "")
            if recognition_label:
                signal_lines.append(recognition_label)
            if extracted_data.get("detected_genres"):
                signal_lines.append(
                    f"жанровые подсказки: {', '.join(extracted_data['detected_genres'])}"
                )
            query_parts.append(image_analysis["storyguide_query"])
            input_kinds.append(f"изображение '{uploaded_file.name}'")
            signal_anchor_records.append(get_movie_by_imdb_id(extracted_data.get("matched_imdb_id")))
            continue

        if extension in TEXT_TYPES:
            extracted_text = extract_text_file(uploaded_file)
            if not extracted_text:
                signal_lines.append(f"файл '{uploaded_file.name}' пустой")
            else:
                preview = _truncate(extracted_text, limit=220)
                signal_lines.append(f"из файла '{uploaded_file.name}' извлечен текст: {preview}")
                query_parts.append(_truncate(extracted_text))
            input_kinds.append(f"текстовый файл '{uploaded_file.name}'")
            continue

        signal_lines.append(
            f"файл '{uploaded_file.name}' пропущен: неподдерживаемый тип '{extension}'"
        )
        input_kinds.append(f"неподдерживаемый файл '{uploaded_file.name}'")

    if cleaned_user_text:
        input_kinds.insert(0, "текстовый запрос")

    combined_query = _merge_query_parts(query_parts)
    intent_report = classify_intent(combined_query)
    query_analysis = analyze_query(combined_query)
    genre_preferences = extract_genre_preferences(combined_query)
    follow_up_marked = has_follow_up_markers(cleaned_user_text)
    anchor_reference_follow_up = _references_current_anchor(cleaned_user_text)
    requested_limit = extract_requested_limit(combined_query, default_limit=state.get("result_limit", 5))
    state_anchor_record = get_movie_by_imdb_id((state.get("anchor_movie") or {}).get("imdb_id"))
    inferred_year_filter = infer_year_filter_from_relative_follow_up(
        cleaned_user_text,
        state_anchor_record,
        current_year_filter=query_analysis["year_filter"],
    )
    explicit_title_request = _extract_explicit_title_request(cleaned_user_text, combined_query)
    explicit_anchor = _resolve_explicit_anchor(
        cleaned_user_text,
        combined_query,
        signal_anchor_records,
        allow_text_anchor=True,
    )
    filter_only_follow_up = (
        (_is_filter_only_follow_up(cleaned_user_text, state) or anchor_reference_follow_up)
        and not explicit_title_request
    )
    similar_intent = intent_report["intent"] == "recommend_similar"

    if explicit_anchor is not None:
        state = default_conversation_state()
        state["anchor_movie"] = build_anchor_payload(explicit_anchor)
        state["filters"]["year_filter"] = dict(query_analysis["year_filter"])
        state["filters"]["include_genres"] = list(genre_preferences["include_genres"])
        state["filters"]["exclude_genres"] = list(genre_preferences["exclude_genres"])
        state["result_limit"] = requested_limit
        anchor_record = explicit_anchor
    elif explicit_title_request:
        state = default_conversation_state()
        state["filters"]["year_filter"] = dict(query_analysis["year_filter"])
        state["filters"]["include_genres"] = list(genre_preferences["include_genres"])
        state["filters"]["exclude_genres"] = list(genre_preferences["exclude_genres"])
        state["result_limit"] = requested_limit
        anchor_record = None
    else:
        state["filters"]["year_filter"] = merge_year_filters(
            state["filters"]["year_filter"],
            query_analysis["year_filter"] or inferred_year_filter,
        )
        state["filters"]["include_genres"] = merge_filter_lists(
            state["filters"]["include_genres"],
            genre_preferences["include_genres"],
        )
        state["filters"]["exclude_genres"] = merge_filter_lists(
            state["filters"]["exclude_genres"],
            genre_preferences["exclude_genres"],
        )
        state["result_limit"] = requested_limit
        anchor_record = state_anchor_record

    if filter_only_follow_up and anchor_record is None:
        response = _compose_filter_only_reply()
        state["last_query"] = combined_query
        state["turn_count"] += 1
        return {
            "response": response,
            "conversation_state": state,
            "ui_payload": {
                "input_summary": ", ".join(input_kinds) if input_kinds else "текстовый запрос",
                "input_kinds": list(input_kinds),
                "signal_lines": list(signal_lines),
                "combined_query": combined_query,
                "intent": intent_report.get("intent", "unknown"),
                "intent_score": intent_report.get("score", 0.0),
                "requested_limit": requested_limit,
                "active_filters_text": "",
                "search_mode": "filter_only",
                "source_movie": None,
                "recommendation_rows": [],
                "metrics": {
                    "recommendation_count": 0,
                    "requested_limit": requested_limit,
                    "signal_count": len(signal_lines),
                    "turn_count": state.get("turn_count", 0),
                    "has_anchor": bool(state.get("anchor_movie")),
                },
                "rule_report": {
                    "catalog_min_year": None,
                    "catalog_max_year": None,
                    "removed_out_of_range": 0,
                    "removed_blacklist": 0,
                },
                "nlp": {
                    "year_filter": dict(query_analysis.get("year_filter") or {}),
                    "warning": query_analysis.get("nlp_warning", ""),
                    "summary": _summarize_nlp(query_analysis),
                },
            },
        }

    filters = state["filters"]
    active_filters_text = describe_active_filters(
        year_filter=filters["year_filter"],
        include_genres=filters["include_genres"],
        exclude_genres=filters["exclude_genres"],
    )

    if anchor_record is not None and (explicit_anchor is not None or filter_only_follow_up or similar_intent):
        search_result = _build_search_result_from_anchor(anchor_record, filters, limit=requested_limit)
    else:
        search_query = explicit_title_request if explicit_title_request and explicit_anchor is None else combined_query
        search_result = search_movies_by_query(
            search_query,
            year_filter=filters["year_filter"],
            include_genres=filters["include_genres"],
            exclude_genres=filters["exclude_genres"],
            limit=requested_limit,
        )
        if search_result.get("mode") == "title_match" and search_result.get("source") is not None:
            state["anchor_movie"] = build_anchor_payload(search_result["source"])
            anchor_record = search_result["source"]

    filtered_search_result, rule_report = _filter_matches_by_rules(search_result)
    recommendation = build_recommendation_response(
        combined_query,
        filtered_search_result,
        year_filter=filters["year_filter"],
        sentiment_summary=query_analysis["sentiment_summary"],
        active_filters_text=active_filters_text,
    )

    state["last_query"] = combined_query
    state["turn_count"] += 1

    input_summary = ", ".join(input_kinds) if input_kinds else "текстовый запрос"
    response = recommendation
    if include_trace:
        pipeline_header = _build_pipeline_header(
            input_summary,
            signal_lines,
            combined_query,
            query_analysis,
            rule_report,
        )
        response = f"{pipeline_header}\n\n{recommendation}"

    return {
        "response": response,
        "conversation_state": state,
        "ui_payload": _build_ui_payload(
            input_summary,
            input_kinds,
            signal_lines,
            combined_query,
            intent_report,
            query_analysis,
            requested_limit,
            active_filters_text,
            search_result,
            filtered_search_result,
            rule_report,
            state,
        ),
    }
