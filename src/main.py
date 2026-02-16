import streamlit as st

from knowledge_graph import load_graph
from logic import process_text_message


st.title("StoryGuide AI Assistant")

# 1. Инициализация истории чата (память)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Отрисовка истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Поле ввода
if user_input := st.chat_input("Введите ваш запрос..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    graph = load_graph()
    bot_response = process_text_message(user_input, graph)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
