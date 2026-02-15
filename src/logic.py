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
        return "⛔ Критическая ошибка: профиль не подтвержден."

    # 2. Проверка числового диапазона
    if data["release_year"] < rules["thresholds"]["min_year"]:
        return "❌ Отказ: год релиза ниже минимального порога."

    if data["release_year"] > rules["thresholds"]["max_year"]:
        return "❌ Отказ: год релиза выше максимального порога."

    # 3. Проверка черного списка
    for genre in data["genres"]:
        if genre in rules["lists"]["blacklist"]:
            return f"⚠ Предупреждение: найден запрещенный жанр ({genre})."

    return f"✅ Успех: объект соответствует сценарию '{rules['scenario_name']}'."
