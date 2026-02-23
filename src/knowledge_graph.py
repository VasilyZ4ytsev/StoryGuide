import networkx as nx

try:
    from .models import MediaEntity
except ImportError:
    from models import MediaEntity


def create_graph():
    """Создает граф знаний для советчика фильмов/книг."""
    graph = nx.Graph()

    media_items = [
        MediaEntity("Интерстеллар", ["фантастика", "драма", "Кристофер Нолан"], 8.7),
        MediaEntity("Начало", ["фантастика", "триллер", "Кристофер Нолан"], 8.8),
        MediaEntity("Дюна", ["фантастика", "драма", "Дени Вильнёв"], 8.0),
        MediaEntity("Остров проклятых", ["триллер", "драма", "Мартин Скорсезе"], 8.2),
    ]
    descriptions = {
        "Интерстеллар": "Команда исследователей отправляется через червоточину, чтобы найти новый дом для человечества.",
        "Начало": "Профессиональный вор проникает в сны людей и получает задание внедрить идею в подсознание цели.",
        "Дюна": "Наследник знатного дома оказывается в центре борьбы за пустынную планету Арракис и ее ресурсы.",
        "Остров проклятых": "Маршал расследует исчезновение пациентки психиатрической клиники на изолированном острове.",
    }
    release_years = {
        "Интерстеллар": 2014,
        "Начало": 2010,
        "Дюна": 2021,
        "Остров проклятых": 2010,
    }

    genres = ["фантастика", "триллер", "драма"]
    creators = ["Кристофер Нолан", "Дени Вильнёв", "Мартин Скорсезе"]

    for item in media_items:
        graph.add_node(
            item.name,
            type="media",
            rating=item.value,
            description=descriptions.get(item.name, ""),
            release_year=release_years.get(item.name),
        )

    graph.add_nodes_from(genres, type="genre")
    graph.add_nodes_from(creators, type="creator")

    relationships = [
        ("Интерстеллар", "Кристофер Нолан"),
        ("Интерстеллар", "фантастика"),
        ("Интерстеллар", "драма"),
        ("Начало", "Кристофер Нолан"),
        ("Начало", "фантастика"),
        ("Начало", "триллер"),
        ("Дюна", "Дени Вильнёв"),
        ("Дюна", "фантастика"),
        ("Дюна", "драма"),
        ("Остров проклятых", "Мартин Скорсезе"),
        ("Остров проклятых", "триллер"),
        ("Остров проклятых", "драма"),
    ]
    graph.add_edges_from(relationships)

    return graph


def load_graph():
    """Возвращает граф знаний для чат-интерфейса."""
    return create_graph()


def find_related_entities(graph, start_node):
    """Находит все узлы, связанные с выбранным узлом."""
    if start_node not in graph:
        return []

    return list(graph.neighbors(start_node))
