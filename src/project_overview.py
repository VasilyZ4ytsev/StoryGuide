import os
from functools import lru_cache

from src.dataset_loader import (
    MERGED_METADATA_CACHE_PATH,
    MOVIE_DATASET_PATH,
    MOVIE_LOCALIZATIONS_PATH,
    MOVIELENS_LINKS_PATH,
    MOVIELENS_MOVIES_PATH,
    MOVIELENS_RATINGS_PATH,
    MOVIELENS_TAGS_PATH,
    REVIEWS_DATASET_PATH,
    THE_MOVIES_CREDITS_PATH,
    THE_MOVIES_KEYWORDS_PATH,
    THE_MOVIES_METADATA_PATH,
    dataset_file_exists,
    load_movie_metadata,
)
from src.integration_pipeline import IMAGE_TYPES, TEXT_TYPES
from src.poster_matcher import FULL_POSTER_DIR, POSTER_INDEX_PATH, poster_index_stats
from src.text_search_index import TEXT_SEARCH_INDEX_PATH, load_text_search_index


def _file_size_mb(path):
    if not os.path.exists(path):
        return 0.0
    return round(os.path.getsize(path) / (1024 * 1024), 1)


def _catalog_metrics():
    records = load_movie_metadata()
    if not records:
        raise RuntimeError("Merged movie metadata is empty.")

    localized_titles = sum(1 for record in records if str(record.get("title_ru", "")).strip())
    described_movies = sum(1 for record in records if str(record.get("overview_display", "")).strip())
    rated_movies = sum(1 for record in records if float(record.get("movielens_rating", 0.0)) > 0.0)
    return {
        "movie_count": len(records),
        "localized_titles": localized_titles,
        "described_movies": described_movies,
        "rated_movies": rated_movies,
    }


def _text_index_metrics():
    index = load_text_search_index()
    document_count = len(index.get("record_ids", []))
    token_count = len(index.get("idf", {}))
    if document_count == 0 or token_count == 0:
        raise RuntimeError("Text search index is empty.")

    return {
        "document_count": document_count,
        "token_count": token_count,
        "file_size_mb": _file_size_mb(TEXT_SEARCH_INDEX_PATH),
    }


def _poster_metrics():
    if not FULL_POSTER_DIR.is_dir():
        raise FileNotFoundError(f"Poster directory is missing: {FULL_POSTER_DIR}")

    stats = poster_index_stats()
    if stats.get("indexed_count", 0) <= 0:
        raise RuntimeError("Poster index is empty.")

    return {
        "indexed_count": stats["indexed_count"],
        "catalog_count": stats["catalog_count"],
        "coverage_ratio": stats["coverage_ratio"],
        "file_size_mb": _file_size_mb(str(POSTER_INDEX_PATH)),
    }


def get_startup_commands():
    return [
        {
            "command": "python -m venv venv",
            "description": "Создать чистое окружение для воспроизводимого запуска.",
        },
        {
            "command": "venv\\Scripts\\activate",
            "description": "Активировать виртуальное окружение в Windows.",
        },
        {
            "command": "pip install -r requirements.txt",
            "description": "Установить зафиксированные зависимости.",
        },
        {
            "command": "python -m unittest discover -s tests -v",
            "description": "Прогнать unit-тесты backend-слоя, пайплайна и presenter-логики.",
        },
        {
            "command": "streamlit run app.py",
            "description": "Поднять официальный Streamlit entrypoint приложения.",
        },
        {
            "command": "python -m src.poster_downloader",
            "description": "Сервисный запуск для загрузки полного каталога постеров.",
        },
    ]


def _dataset_entry(name, path, purpose):
    return {
        "name": name,
        "available": dataset_file_exists(path),
        "path": path,
        "purpose": purpose,
    }


@lru_cache(maxsize=1)
def get_project_overview():
    catalog_metrics = _catalog_metrics()
    text_index_metrics = _text_index_metrics()
    poster_metrics = _poster_metrics()
    supported_types = sorted(IMAGE_TYPES | TEXT_TYPES)

    summary_metrics = [
        {
            "label": "Фильмов в каталоге",
            "value": catalog_metrics["movie_count"],
            "caption": "Объединенный каталог рекомендаций StoryGuide.",
        },
        {
            "label": "Текстовый индекс",
            "value": text_index_metrics["document_count"],
            "caption": "Документов доступно для semantic search.",
        },
        {
            "label": "Постеров в индексе",
            "value": poster_metrics["indexed_count"],
            "caption": "Покрытие visual matching по локальному каталогу постеров.",
        },
        {
            "label": "Поддерживаемые файлы",
            "value": len(supported_types),
            "caption": "Типы вложений, которые можно загрузить в интерфейс.",
        },
    ]

    return {
        "summary_metrics": summary_metrics,
        "supported_types": supported_types,
        "catalog_metrics": catalog_metrics,
        "text_index_metrics": text_index_metrics,
        "poster_metrics": poster_metrics,
        "capabilities": [
            "Рекомендации по названию фильма и контексту диалога.",
            "Гибридный поиск по описанию сюжета, ключевым словам и тегам.",
            "Фильтры по годам, включаемым и исключаемым жанрам.",
            "OCR по изображениям и подготовка запроса из постера или текстовой страницы.",
            "Visual poster matching по полному локальному индексу постеров.",
            "Структурированный payload для UI и диагностика пайплайна.",
        ],
        "architecture_layers": [
            {
                "name": "Presentation",
                "description": "Официальный entrypoint в `app.py`, UI-логика в `src/main.py`, presenter-функции в `src/ui_presenter.py`.",
            },
            {
                "name": "Backend Service",
                "description": "Сессионные операции, история чата и вызовы из UI в `src/app_service.py`.",
            },
            {
                "name": "Orchestration",
                "description": "Единый pipeline в `src/integration_pipeline.py`, который объединяет текст, файлы, OCR и правила.",
            },
            {
                "name": "Domain Logic",
                "description": "Intent routing, conversation state, рекомендации, NLP и sentiment.",
            },
            {
                "name": "Data Layer",
                "description": "Каталог фильмов, объединенный metadata cache, text index и poster index.",
            },
        ],
        "quality_gates": [
            "Unit-тесты покрывают chat-flow, payload для UI, presenter и сервисный слой.",
            "requirements.txt зафиксирован по версиям для воспроизводимого запуска.",
            "Сессии и состояние диалога сохраняются на диск в `data/processed`.",
            "При отсутствии обязательных данных или зависимостей приложение завершает запрос явной ошибкой.",
        ],
        "deployment_notes": [
            "Официальный UI-запуск: `streamlit run app.py`.",
            "Сервисные модули запускаются только как пакеты вида `python -m src.poster_downloader`.",
            "Кэши `merged_movie_metadata.pkl`, `text_search_index.pkl` и `poster_index.pkl` пересобираются из обязательных датасетов при несовпадении сигнатуры.",
        ],
        "datasets": [
            _dataset_entry("MovieGenre.csv", MOVIE_DATASET_PATH, "Базовый каталог фильмов, жанров и рейтингов."),
            _dataset_entry("movie_localizations.json", MOVIE_LOCALIZATIONS_PATH, "Русские локализации названий, описаний и алиасов."),
            _dataset_entry("movies_metadata.csv", THE_MOVIES_METADATA_PATH, "TMDB-метаданные, overview и tagline."),
            _dataset_entry("credits.csv", THE_MOVIES_CREDITS_PATH, "Актеры и режиссеры из The Movies Dataset."),
            _dataset_entry("keywords.csv", THE_MOVIES_KEYWORDS_PATH, "Ключевые слова из The Movies Dataset."),
            _dataset_entry("ml-25m/movies.csv", MOVIELENS_MOVIES_PATH, "Названия и жанры MovieLens 25M."),
            _dataset_entry("ml-25m/ratings.csv", MOVIELENS_RATINGS_PATH, "Пользовательские рейтинги MovieLens 25M."),
            _dataset_entry("ml-25m/tags.csv", MOVIELENS_TAGS_PATH, "Пользовательские теги MovieLens 25M."),
            _dataset_entry("ml-25m/links.csv", MOVIELENS_LINKS_PATH, "Связи MovieLens -> imdb/tmdb."),
            _dataset_entry("IMDB Dataset.csv", REVIEWS_DATASET_PATH, "Лексикон для анализа тональности англоязычных отзывов."),
            _dataset_entry("FullMoviePosters", str(FULL_POSTER_DIR), "Полный локальный каталог постеров для visual matching."),
        ],
        "artifacts": [
            {
                "name": "merged_movie_metadata.pkl",
                "size_mb": _file_size_mb(MERGED_METADATA_CACHE_PATH),
                "status": "ready" if os.path.exists(MERGED_METADATA_CACHE_PATH) else "missing",
            },
            {
                "name": "text_search_index.pkl",
                "size_mb": text_index_metrics["file_size_mb"],
                "status": "ready" if os.path.exists(TEXT_SEARCH_INDEX_PATH) else "missing",
            },
            {
                "name": "poster_index.pkl",
                "size_mb": poster_metrics["file_size_mb"],
                "status": "ready" if os.path.exists(str(POSTER_INDEX_PATH)) else "missing",
            },
        ],
        "startup_commands": get_startup_commands(),
    }
