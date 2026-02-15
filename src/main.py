import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from knowledge_graph import create_graph, find_related_entities

st.title("Исследователь графа знаний")

# 1. Загружаем граф
graph = create_graph()

# 2. Выбор узла для анализа
all_nodes = sorted(list(graph.nodes()))
selected_node = st.selectbox("Выберите объект для поиска связей:", all_nodes)

# 3. Кнопка поиска
if st.button("Найти связи"):
    results = find_related_entities(graph, selected_node)
    if results:
        st.success(f"Объект '{selected_node}' связан с: {', '.join(results)}")
    else:
        st.warning("Связи не найдены")

# 4. Визуализация графа
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
