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

    genres = ["фантастика", "триллер", "драма"]
    creators = ["Кристофер Нолан", "Дени Вильнёв", "Мартин Скорсезе"]

    for item in media_items:
        graph.add_node(item.name, type="media", rating=item.value)

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


def find_related_entities(graph, start_node):
    """Находит все узлы, связанные с выбранным узлом."""
    if start_node not in graph:
        return []

    return list(graph.neighbors(start_node))
