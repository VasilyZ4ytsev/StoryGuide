from collections import Counter

import pandas as pd


TABLE_COLUMNS = {
    "rank": "№",
    "title": "Фильм",
    "year": "Год",
    "rating": "Рейтинг",
    "genres": "Жанры",
    "score": "Score",
}


def format_metric_value(value, default="-"):
    if value in (None, ""):
        return default
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_recommendation_dataframe(rows):
    frame = pd.DataFrame(list(rows or []))
    if frame.empty:
        return pd.DataFrame(columns=list(TABLE_COLUMNS.values()))

    for source_name in TABLE_COLUMNS:
        if source_name not in frame.columns:
            frame[source_name] = ""

    frame = frame[list(TABLE_COLUMNS.keys())].rename(columns=TABLE_COLUMNS)
    frame["Рейтинг"] = pd.to_numeric(frame["Рейтинг"], errors="coerce")
    frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
    return frame


def build_rating_chart_frame(rows):
    frame = build_recommendation_dataframe(rows)
    if frame.empty:
        return pd.DataFrame(columns=["Фильм", "Рейтинг"]).set_index("Фильм")

    chart_frame = frame[["Фильм", "Рейтинг"]].dropna()
    if chart_frame.empty:
        return pd.DataFrame(columns=["Рейтинг"])
    return chart_frame.set_index("Фильм")


def build_genre_distribution(rows):
    counts = Counter()
    for row in rows or []:
        genres_text = str(row.get("genres", "") or "")
        for genre in genres_text.split(","):
            normalized = genre.strip()
            if normalized:
                counts[normalized] += 1

    if not counts:
        return pd.DataFrame(columns=["Жанр", "Количество"])

    data = [
        {"Жанр": genre, "Количество": count}
        for genre, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return pd.DataFrame(data)


def build_year_distribution(rows):
    years = [
        int(row["year"])
        for row in rows or []
        if isinstance(row, dict) and isinstance(row.get("year"), int)
    ]
    if not years:
        return pd.DataFrame(columns=["Год", "Количество"])

    year_counts = Counter(years)
    data = [
        {"Год": year, "Количество": count}
        for year, count in sorted(year_counts.items())
    ]
    return pd.DataFrame(data)


def build_supported_types_caption(supported_types):
    extensions = [f".{extension}" for extension in supported_types or []]
    return ", ".join(extensions)
