import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, "data", "raw", "rules.json")


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def check_rules(data):
    """Проверка сущности по продукционным правилам."""
    rules = load_rules()

    # 1. Критическая проверка
    if rules["critical_rules"]["must_be_verified"] and not data["is_verified"]:
        return "Критическая ошибка: профиль не подтвержден."

    # 2. Проверка числового диапазона
    if data["release_year"] < rules["thresholds"]["min_year"]:
        return "Отказ: год релиза ниже минимального порога."

    if data["release_year"] > rules["thresholds"]["max_year"]:
        return "Отказ: год релиза выше максимального порога."

    # 3. Проверка черного списка
    for genre in data["genres"]:
        if genre in rules["lists"]["blacklist"]:
            return f"Предупреждение: найден запрещенный жанр ({genre})."

    return f"✅ Успех: объект соответствует сценарию '{rules['scenario_name']}'."


def process_text_message(text, data_source):
    """
    Принимает текст пользователя, ищет термин в графе/словаре
    и возвращает ответ.
    """
    normalized_text = text.strip().lower()

    if not normalized_text:
        return "Я не знаю такого термина"

    if "привет" in normalized_text:
        return "Привет! Я готов помочь по теме фильмов и книг. Напиши точное название, жанр или автора."

    if data_source is None:
        return "Я не знаю такого термина"

    # Поиск по графу знаний (Lab 3)
    if hasattr(data_source, "nodes") and hasattr(data_source, "neighbors"):
        normalized_nodes = {str(node).lower(): node for node in data_source.nodes}
        if normalized_text in normalized_nodes:
            node = normalized_nodes[normalized_text]
            neighbors = sorted(str(neighbor) for neighbor in data_source.neighbors(node))
            if neighbors:
                return f"Я нашел '{node}' в базе! С этим связано: {', '.join(neighbors)}"
            return f"Я нашел '{node}' в базе, но связанных объектов нет."

    # Поиск по словарям/правилам (Lab 2)
    if isinstance(data_source, dict):
        def iter_terms(obj):
            if isinstance(obj, dict):
                for nested_value in obj.values():
                    yield from iter_terms(nested_value)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    yield from iter_terms(item)
            elif isinstance(obj, str):
                yield obj

        for term in iter_terms(data_source):
            if term.lower() == normalized_text:
                return f"Я нашел '{term}' в правилах."

    return "Я не знаю такого термина"
