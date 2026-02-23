import json
import os
import uuid

import streamlit as st

from knowledge_graph import load_graph
from logic import process_text_message


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_SESSIONS_DIR = os.path.join(BASE_DIR, "data", "processed", "chat_sessions")


def get_session_id_from_query_params():
    if hasattr(st, "query_params"):
        raw_session_id = st.query_params.get("sid", "")

        if isinstance(raw_session_id, list):
            raw_session_id = raw_session_id[0] if raw_session_id else ""

        session_id = str(raw_session_id).strip()
        if session_id:
            return session_id

        session_id = uuid.uuid4().hex
        st.query_params["sid"] = session_id
        return session_id

    raw_query_params = st.experimental_get_query_params().get("sid", [""])
    raw_session_id = raw_query_params[0] if raw_query_params else ""
    session_id = str(raw_session_id).strip()
    if session_id:
        return session_id

    session_id = uuid.uuid4().hex
    st.experimental_set_query_params(sid=session_id)
    return session_id


def normalize_session_id(session_id):
    safe_session_id = "".join(char for char in session_id if char.isalnum() or char in ("-", "_"))
    return safe_session_id or uuid.uuid4().hex


def get_history_path(session_id):
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
    safe_session_id = normalize_session_id(session_id)
    return os.path.join(CHAT_SESSIONS_DIR, f"{safe_session_id}.json")


def load_messages(history_path):
    if not os.path.exists(history_path):
        return []

    try:
        with open(history_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    messages = []
    for item in data:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    return messages


def save_messages(history_path, messages):
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=2)


st.title("StoryGuide AI Assistant")

session_id = get_session_id_from_query_params()
history_path = get_history_path(session_id)

# При полном refresh Streamlit стартует новую server-side сессию,
# поэтому восстанавливаем историю из файла по sid в URL.
if st.session_state.get("history_path") != history_path:
    st.session_state.history_path = history_path
    st.session_state.messages = load_messages(history_path)
elif "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Введите ваш запрос..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_messages(st.session_state.history_path, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_input)

    graph = load_graph()
    bot_response = process_text_message(user_input, graph)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    save_messages(st.session_state.history_path, st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(bot_response)
