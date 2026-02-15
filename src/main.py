import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from knowledge_graph import create_graph, find_related_entities
from logic import check_rules, load_rules
from mock_data import test_entity as default_data


def render_lab2():
    st.header("Лабораторная 2: Продукционная модель")
    st.write("### Настройка входных данных")

    rules = load_rules()
    genre_options = sorted(
        set(default_data["genres"] + rules["lists"]["whitelist"] + rules["lists"]["blacklist"])
    )

    user_year = st.sidebar.number_input(
        "Год релиза:",
        value=int(default_data["release_year"]),
        key="lab2_year",
    )
    user_genres = st.sidebar.multiselect(
        "Жанры:",
        options=genre_options,
        default=default_data["genres"],
        key="lab2_genres",
    )
    user_verified = st.sidebar.checkbox(
        "Профиль подтвержден (True/False):",
        value=default_data["is_verified"],
        key="lab2_verified",
    )

    if st.button("Запустить проверку", key="lab2_button"):
        current_test_data = {
            "release_year": user_year,
            "is_verified": user_verified,
            "favorite_title": default_data["favorite_title"],
            "genres": user_genres,
        }

        result = check_rules(current_test_data)

        if "✅" in result:
            st.success(result)
        elif "⛔" in result:
            st.error(result)
        else:
            st.warning(result)

        st.write("Текущий тестовый объект:")
        st.json(current_test_data)


def render_lab3():
    st.header("Лабораторная 3: Объектная модель и граф знаний")

    graph = create_graph()
    all_nodes = sorted(list(graph.nodes()))
    selected_node = st.selectbox(
        "Выберите объект для поиска связей:",
        all_nodes,
        key="lab3_node",
    )

    if st.button("Найти связи", key="lab3_button"):
        results = find_related_entities(graph, selected_node)
        if results:
            st.success(f"Объект '{selected_node}' связан с: {', '.join(results)}")
        else:
            st.warning("Связи не найдены")

    st.write("### Визуализация структуры")
    figure, axis = plt.subplots(figsize=(10, 7))
    positions = nx.spring_layout(graph, seed=42)

    node_colors = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get("type", "unknown")
        if node_type == "media":
            node_colors.append("lightblue")
        elif node_type == "genre":
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightcoral")

    nx.draw(
        graph,
        positions,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=2200,
        font_size=9,
        ax=axis,
    )

    st.pyplot(figure)


st.title("StoryGuide: Лабораторные работы")
mode = st.sidebar.radio(
    "Выберите режим:",
    ("Лаба 2: Rule-Based", "Лаба 3: Knowledge Graph"),
)

if mode == "Лаба 2: Rule-Based":
    render_lab2()
else:
    render_lab3()
