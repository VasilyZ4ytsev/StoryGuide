import json
import os
import uuid
from pathlib import Path

import streamlit as st

from logic import process_text_message
from vision_processor import analyze_uploaded_image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_SESSIONS_DIR = os.path.join(BASE_DIR, "data", "processed", "chat_sessions")
IMAGE_TYPES = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
TEXT_TYPES = {"txt", "md", "json", "csv"}
CHAT_FILE_TYPES = sorted(IMAGE_TYPES | TEXT_TYPES)


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


def append_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    save_messages(st.session_state.history_path, st.session_state.messages)


def render_assistant_message(content):
    append_message("assistant", content)
    with st.chat_message("assistant"):
        st.markdown(content)


def render_user_message(content, uploaded_files=None):
    append_message("user", content)
    with st.chat_message("user"):
        st.markdown(content)
        for uploaded_file in uploaded_files or []:
            render_uploaded_file_preview(uploaded_file)


def render_uploaded_file_preview(uploaded_file):
    extension = file_extension(uploaded_file.name)
    if extension in IMAGE_TYPES:
        st.image(uploaded_file.getvalue(), caption=uploaded_file.name, width=320)
    else:
        st.caption(f"Файл: {uploaded_file.name}")


def file_extension(file_name):
    return Path(file_name).suffix.lower().lstrip(".")


def build_user_message(text, uploaded_files):
    lines = []
    cleaned_text = text.strip()
    if cleaned_text:
        lines.append(cleaned_text)

    if uploaded_files:
        lines.append("Вложения:")
        for uploaded_file in uploaded_files:
            lines.append(f"- {uploaded_file.name}")

    return "\n".join(lines).strip() or "Пустое сообщение"


def build_combined_query(user_text, generated_query):
    cleaned_user_text = user_text.strip()
    cleaned_generated_query = generated_query.strip()

    if cleaned_user_text and cleaned_generated_query:
        lowered = cleaned_user_text.lower()
        if cleaned_generated_query.lower() in lowered:
            return cleaned_user_text
        return f"{cleaned_generated_query}. {cleaned_user_text}"

    return cleaned_user_text or cleaned_generated_query


def extract_text_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Файл '{uploaded_file.name}' не удалось прочитать как UTF-8 текст."
        ) from error

    return content.strip()


def process_uploaded_image(uploaded_file, user_text):
    image_analysis = analyze_uploaded_image(uploaded_file.getvalue(), uploaded_file.name)
    extracted_data = image_analysis["extracted_data"]
    combined_query = build_combined_query(user_text, image_analysis["storyguide_query"])
    recommendation = process_text_message(combined_query)

    matched_movie = extracted_data.get("matched_movie")
    if matched_movie:
        return recommendation

    return recommendation


def process_uploaded_text_file(uploaded_file, user_text):
    extracted_text = extract_text_file(uploaded_file)
    if not extracted_text:
        return f"Файл '{uploaded_file.name}' пустой. Нечего анализировать."

    combined_query = build_combined_query(user_text, extracted_text)
    recommendation = process_text_message(combined_query)
    return f"Я прочитал текст из файла и попробовал подобрать фильмы.\n\n{recommendation}"


def handle_chat_submission(user_text, uploaded_files):
    if not user_text.strip() and not uploaded_files:
        return

    visible_prompt = build_user_message(user_text, uploaded_files)
    render_user_message(visible_prompt, uploaded_files)

    if not uploaded_files:
        bot_response = process_text_message(user_text)
        render_assistant_message(bot_response)
        return

    responses = []
    for uploaded_file in uploaded_files:
        extension = file_extension(uploaded_file.name)
        try:
            if extension in IMAGE_TYPES:
                responses.append(process_uploaded_image(uploaded_file, user_text))
            elif extension in TEXT_TYPES:
                responses.append(process_uploaded_text_file(uploaded_file, user_text))
            else:
                responses.append(
                    f"Файл '{uploaded_file.name}' не поддерживается. "
                    f"Допустимые типы: {', '.join(CHAT_FILE_TYPES)}."
                )
        except Exception as error:
            responses.append(f"Ошибка при обработке файла '{uploaded_file.name}': {error}")

    render_assistant_message("\n\n---\n\n".join(responses))


def process_chat_submission(chat_value):
    if hasattr(chat_value, "text"):
        user_text = chat_value.text or ""
        uploaded_files = list(getattr(chat_value, "files", []) or [])
    else:
        user_text = str(chat_value or "")
        uploaded_files = []
    handle_chat_submission(user_text, uploaded_files)


st.set_page_config(page_title="StoryGuide AI Assistant", layout="wide")
st.title("StoryGuide AI Assistant")
st.caption("Рекомендации по фильмам. Постеры и текстовые файлы можно отправлять прямо в чат.")

session_id = get_session_id_from_query_params()
history_path = get_history_path(session_id)

if st.session_state.get("history_path") != history_path:
    st.session_state.history_path = history_path
    st.session_state.messages = load_messages(history_path)
elif "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_value = st.chat_input(
    "Введите ваш запрос по фильмам...",
    accept_file="multiple",
    file_type=CHAT_FILE_TYPES,
)
if chat_value:
    process_chat_submission(chat_value)
