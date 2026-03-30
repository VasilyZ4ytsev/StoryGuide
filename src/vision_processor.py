import re
import warnings
from functools import lru_cache

import cv2
import numpy as np

from src.dataset_loader import match_movie_title
from src.poster_matcher import match_movie_poster, poster_index_stats


TYPE_LABELS = {
    "movie_poster": "Постер фильма",
    "text_page": "Страница с текстом о фильме",
}

GENRE_KEYWORDS = {
    "фантастика": ("фантастика", "sci-fi", "science fiction"),
    "драма": ("драма", "drama"),
    "триллер": ("триллер", "thriller"),
    "фэнтези": ("фэнтези", "fantasy"),
    "ужасы": ("ужасы", "horror"),
    "детектив": ("детектив", "detective"),
    "боевик": ("боевик", "action"),
    "комедия": ("комедия", "comedy"),
    "криминал": ("криминал", "crime"),
    "анимация": ("анимация", "animation"),
}
MOVIE_HINTS = (
    "film",
    "movie",
    "режиссер",
    "director",
    "starring",
    "screenplay",
    "кино",
    "фильм",
)
WORD_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
PIN_MEMORY_WARNING_PATTERN = re.escape(
    "'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used."
)


def _require_easyocr():
    warnings.filterwarnings(
        "ignore",
        message=PIN_MEMORY_WARNING_PATTERN,
        category=UserWarning,
        module=r"torch\.utils\.data\.dataloader",
    )
    try:
        import easyocr
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "EasyOCR не установлен. Установите зависимости командой "
            "'pip install -r requirements.txt'."
        ) from error

    return easyocr


@lru_cache(maxsize=1)
def get_ocr_reader():
    easyocr = _require_easyocr()

    try:
        return easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    except Exception as error:
        raise RuntimeError(f"Не удалось инициализировать EasyOCR: {error}") from error


def decode_image(image_bytes):
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Файл не удалось декодировать как изображение.")
    return image


def preprocess_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return binary


def _normalize_text(value):
    return " ".join(str(value).split())


def _box_sort_key(box):
    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]
    return min(y_values), min(x_values)


def run_ocr(image_bgr):
    reader = get_ocr_reader()
    prepared_images = [
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        preprocess_for_ocr(image_bgr),
    ]

    merged_detections = {}
    for prepared_image in prepared_images:
        detections = reader.readtext(prepared_image, detail=1, paragraph=False)
        for box, text, confidence in detections:
            normalized_text = _normalize_text(text)
            if not normalized_text:
                continue

            key = normalized_text.lower()
            current = merged_detections.get(key)
            if current is None or confidence > current["confidence"]:
                merged_detections[key] = {
                    "box": box,
                    "text": normalized_text,
                    "confidence": float(confidence),
                }

    ordered = sorted(merged_detections.values(), key=lambda item: _box_sort_key(item["box"]))
    full_text = "\n".join(item["text"] for item in ordered)
    average_confidence = (
        sum(item["confidence"] for item in ordered) / len(ordered) if ordered else 0.0
    )

    return {
        "text": full_text,
        "detections": ordered,
        "count": len(ordered),
        "average_confidence": round(float(average_confidence), 3),
    }


def _compute_colorfulness(image_bgr):
    blue, green, red = cv2.split(image_bgr.astype("float32"))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    rg_std, rg_mean = np.std(rg), np.mean(rg)
    yb_std, yb_mean = np.std(yb), np.mean(yb)
    return float(np.sqrt(rg_std ** 2 + yb_std ** 2) + 0.3 * np.sqrt(rg_mean ** 2 + yb_mean ** 2))


def _largest_quadrilateral_ratio(gray_image):
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray_image.shape[0] * gray_image.shape[1]
    best_ratio = 0.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(contour)
        if len(approximation) == 4 and area > 0:
            best_ratio = max(best_ratio, area / image_area)

    return float(best_ratio)


def _extract_metrics(image_bgr, ocr_result):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(40, int(min(gray.shape[:2]) * 0.2)),
        maxLineGap=15,
    )
    tokens = WORD_PATTERN.findall(ocr_result["text"])

    return {
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "aspect_ratio": round(float(image_bgr.shape[0] / max(image_bgr.shape[1], 1)), 3),
        "mean_intensity": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "edge_density": round(float(np.count_nonzero(edges) / edges.size), 4),
        "line_count": int(0 if lines is None else len(lines)),
        "colorfulness": round(_compute_colorfulness(image_bgr), 2),
        "bright_ratio": round(float(np.mean(gray >= 200)), 4),
        "dark_ratio": round(float(np.mean(gray <= 60)), 4),
        "largest_quadrilateral_ratio": round(_largest_quadrilateral_ratio(gray), 4),
        "ocr_box_count": int(ocr_result["count"]),
        "ocr_token_count": int(len(tokens)),
        "ocr_average_confidence": ocr_result["average_confidence"],
    }


def _contains_hint(text, hints):
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def _score_image_type(metrics, ocr_text):
    scores = {"movie_poster": 0.0, "text_page": 0.0}
    reasons = {"movie_poster": [], "text_page": []}

    if metrics["ocr_box_count"] >= 8:
        scores["text_page"] += 3.0
        reasons["text_page"].append("много строк распознанного текста")
    if metrics["ocr_token_count"] >= 28:
        scores["text_page"] += 3.0
        reasons["text_page"].append("распознан длинный фрагмент описания")
    if metrics["bright_ratio"] >= 0.4:
        scores["text_page"] += 1.0
        reasons["text_page"].append("светлый фон страницы")
    if metrics["colorfulness"] <= 18:
        scores["text_page"] += 0.5
    if metrics["line_count"] >= 12:
        scores["text_page"] += 1.0

    if metrics["colorfulness"] >= 20:
        scores["movie_poster"] += 2.0
        reasons["movie_poster"].append("выраженная цветность постера")
    if metrics["contrast"] >= 42:
        scores["movie_poster"] += 1.5
        reasons["movie_poster"].append("контрастная афишная графика")
    if metrics["edge_density"] >= 0.035:
        scores["movie_poster"] += 1.0
        reasons["movie_poster"].append("заметная детализация изображения")
    if metrics["ocr_box_count"] <= 6:
        scores["movie_poster"] += 1.0
    if _contains_hint(ocr_text, MOVIE_HINTS):
        scores["movie_poster"] += 3.0
        reasons["movie_poster"].append("в тексте есть признаки фильма или режиссера")

    image_type = max(scores, key=scores.get)
    total_score = sum(scores.values()) or 1.0
    confidence = round(scores[image_type] / total_score, 3)
    return image_type, confidence, scores, reasons[image_type]


def _clean_candidate(line):
    compact = " ".join(line.split())
    if len(compact) < 3:
        return ""
    if sum(char.isdigit() for char in compact) > max(2, len(compact) // 3):
        return ""
    return compact


def _extract_title_candidate(ocr_result, image_height):
    best_score = -1.0
    best_line = ""

    for item in ocr_result["detections"]:
        text = _clean_candidate(item["text"])
        if not text:
            continue

        words = WORD_PATTERN.findall(text)
        if not words or len(words) > 8:
            continue

        top_y = min(point[1] for point in item["box"])
        bottom_y = max(point[1] for point in item["box"])
        left_x = min(point[0] for point in item["box"])
        right_x = max(point[0] for point in item["box"])
        height = max(bottom_y - top_y, 1.0)
        width = max(right_x - left_x, 1.0)

        prominence = height * width
        top_bonus = max(0.0, 1.2 - (top_y / max(image_height, 1.0)))
        alpha_bonus = sum(any(char.isalpha() for char in word) for word in words)
        score = prominence * item["confidence"] * (top_bonus + alpha_bonus)

        if score > best_score:
            best_score = score
            best_line = text

    return best_line


def _extract_secondary_name(lines, title_candidate):
    for line in lines:
        compact = _clean_candidate(line)
        if not compact or compact == title_candidate:
            continue
        words = WORD_PATTERN.findall(compact)
        if 1 <= len(words) <= 5 and any(char.isalpha() for char in compact):
            return compact
    return ""


def _extract_genres(ocr_text):
    lowered = ocr_text.lower()
    detected = []
    for genre, variants in GENRE_KEYWORDS.items():
        if any(variant in lowered for variant in variants):
            detected.append(genre)
    return detected


def _build_storyguide_query(title_candidate, image_type, genres):
    if title_candidate:
        return f"Мне нравится {title_candidate}"

    if genres:
        return f"Посоветуй фильм в жанре {genres[0]}"

    if image_type == "movie_poster":
        return "Посоветуй фильм, похожий на загруженный постер"

    return "Посоветуй фильм по загруженному изображению"


def _choose_movie_match(title_match, poster_match):
    poster_candidate = poster_match["best_match"] if poster_match else None

    if poster_candidate and poster_candidate["score"] >= 0.72:
        return poster_candidate["record"], "poster_visual"

    if title_match is not None and title_match["score"] >= 0.62:
        return title_match["record"], "ocr_title"

    if poster_candidate and poster_candidate["score"] >= 0.45:
        return poster_candidate["record"], "poster_visual"

    if title_match is not None:
        return title_match["record"], "ocr_title"

    return None, ""


def _recognition_details(matched_record, match_source, title_match, poster_match):
    if matched_record is None:
        return {
            "recognition_status": "not_identified",
            "recognition_label": "Не удалось определить фильм",
        }

    if match_source == "poster_visual":
        confidence = poster_match["best_match"]["score"] if poster_match else 0.0
        return {
            "recognition_status": "poster_visual",
            "recognition_label": (
                f"Найден по постеру: {matched_record.get('display_full_title', matched_record['full_title'])} "
                f"({round(float(confidence) * 100)}%)"
            ),
        }

    confidence = title_match["score"] if title_match else 0.0
    return {
        "recognition_status": "ocr_title",
        "recognition_label": (
            f"Найден по OCR: {matched_record.get('display_full_title', matched_record['full_title'])} "
            f"({round(float(confidence) * 100)}%)"
        ),
    }


def _build_extracted_data(image_bgr, image_type, ocr_result, metrics):
    lines = [line.strip() for line in ocr_result["text"].splitlines() if line.strip()]
    title_candidate = _extract_title_candidate(ocr_result, metrics["height"])
    secondary_name = _extract_secondary_name(lines, title_candidate)
    genres = _extract_genres(ocr_result["text"])
    title_match = match_movie_title(title_candidate or ocr_result["text"])
    poster_match = None
    poster_stats = None
    if image_type == "movie_poster":
        poster_match = match_movie_poster(image_bgr)
        poster_stats = poster_index_stats()

    matched_record, match_source = _choose_movie_match(title_match, poster_match)

    if matched_record:
        title_candidate = matched_record.get("display_full_title", matched_record["full_title"])
        genres = matched_record["genres_ru"] or genres

    trusted_title = matched_record.get("display_title", matched_record["title"]) if matched_record else ""
    storyguide_query = _build_storyguide_query(trusted_title, image_type, genres)

    payload = {
        "title_candidate": title_candidate,
        "detected_genres": genres,
        "top_lines": lines[:5],
        "storyguide_query": storyguide_query,
    }

    if matched_record:
        payload["matched_movie"] = matched_record.get("display_full_title", matched_record["full_title"])
        payload["matched_rating"] = matched_record["rating"]
        payload["matched_imdb_id"] = matched_record["imdb_id"]
        payload["match_source"] = match_source
        if title_match is not None:
            payload["title_match_score"] = title_match["score"]

    if poster_match is not None:
        payload["poster_match_score"] = poster_match["best_match"]["score"]
        payload["poster_match_candidates"] = [
            {
                "title": item["record"].get("display_full_title", item["record"]["full_title"]),
                "score": item["score"],
            }
            for item in poster_match["matches"]
        ]
    if poster_stats is not None:
        payload["poster_index_size"] = poster_stats["indexed_count"]
        payload["poster_catalog_size"] = poster_stats["catalog_count"]
        payload["poster_coverage_ratio"] = poster_stats["coverage_ratio"]

    payload.update(_recognition_details(matched_record, match_source, title_match, poster_match))

    if image_type == "movie_poster":
        payload["director_or_cast_candidate"] = secondary_name
    else:
        payload["description_preview"] = lines[:8]

    return payload


def build_summary(analysis_result):
    type_label = TYPE_LABELS[analysis_result["image_type"]]
    confidence = round(analysis_result["confidence"] * 100)
    reasons = ", ".join(analysis_result["classification_reasons"]) or "признаки распределены неоднозначно"
    title_candidate = analysis_result["extracted_data"].get("title_candidate")
    title_text = f" Вероятное название: {title_candidate}." if title_candidate else ""
    return f"{type_label} ({confidence}%): {reasons}.{title_text}"


def analyze_uploaded_image(image_bytes, file_name="uploaded_image"):
    if not image_bytes:
        raise ValueError("Пустой файл изображения.")

    image = decode_image(image_bytes)
    ocr_result = run_ocr(image)
    metrics = _extract_metrics(image, ocr_result)
    image_type, confidence, scores, reasons = _score_image_type(metrics, ocr_result["text"])
    extracted_data = _build_extracted_data(image, image_type, ocr_result, metrics)

    result = {
        "file_name": file_name,
        "image_type": image_type,
        "confidence": confidence,
        "classification_scores": {key: round(value, 3) for key, value in scores.items()},
        "classification_reasons": reasons,
        "ocr": ocr_result,
        "metrics": metrics,
        "extracted_data": extracted_data,
        "storyguide_query": extracted_data["storyguide_query"],
    }
    result["summary"] = build_summary(result)
    return result
