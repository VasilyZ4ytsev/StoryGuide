import json
import os
import uuid

from src.conversation_state import (
    default_conversation_state,
    get_state_path,
    load_conversation_state,
    save_conversation_state,
)
from src.integration_pipeline import IMAGE_TYPES, TEXT_TYPES, run_integrated_pipeline
from src.project_overview import get_project_overview


CHAT_SESSIONS_SUBDIR = os.path.join("data", "processed", "chat_sessions")


def get_supported_chat_file_types():
    return sorted(IMAGE_TYPES | TEXT_TYPES)


def load_project_overview():
    return get_project_overview()


def normalize_session_id(session_id):
    safe_session_id = "".join(char for char in str(session_id or "") if char.isalnum() or char in ("-", "_"))
    return safe_session_id or uuid.uuid4().hex


def get_history_path(base_dir, session_id):
    history_dir = os.path.join(base_dir, CHAT_SESSIONS_SUBDIR)
    os.makedirs(history_dir, exist_ok=True)
    return os.path.join(history_dir, f"{normalize_session_id(session_id)}.json")


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
        json.dump(list(messages or []), file, ensure_ascii=False, indent=2)


def append_message(messages, role, content):
    updated_messages = list(messages or [])
    updated_messages.append({"role": role, "content": str(content or "")})
    return updated_messages


def load_chat_session(base_dir, session_id):
    safe_session_id = normalize_session_id(session_id)
    history_path = get_history_path(base_dir, safe_session_id)
    state_path = get_state_path(base_dir, safe_session_id)
    return {
        "session_id": safe_session_id,
        "history_path": history_path,
        "state_path": state_path,
        "messages": load_messages(history_path),
        "conversation_state": load_conversation_state(state_path),
    }


def reset_chat_session(base_dir, session_id, clear_messages=False):
    session = load_chat_session(base_dir, session_id)
    session["conversation_state"] = default_conversation_state()
    save_conversation_state(session["state_path"], session["conversation_state"])
    if clear_messages:
        session["messages"] = []
        save_messages(session["history_path"], session["messages"])
    return session


def reset_filters_only(base_dir, session_id):
    session = load_chat_session(base_dir, session_id)
    session["conversation_state"]["filters"] = {
        "year_filter": {},
        "include_genres": [],
        "exclude_genres": [],
    }
    save_conversation_state(session["state_path"], session["conversation_state"])
    return session


def build_user_message(text, uploaded_files):
    lines = []
    cleaned_text = str(text or "").strip()
    if cleaned_text:
        lines.append(cleaned_text)

    for uploaded_file in uploaded_files or []:
        lines.append(f"[Файл] {uploaded_file.name}")

    return "\n".join(lines).strip() or "Пустое сообщение"


def process_chat_turn(base_dir, session_id, user_text="", uploaded_files=None):
    uploaded_files = list(uploaded_files or [])
    session = load_chat_session(base_dir, session_id)
    visible_prompt = build_user_message(user_text, uploaded_files)
    session["messages"] = append_message(session["messages"], "user", visible_prompt)

    pipeline_result = run_integrated_pipeline(
        user_text=user_text,
        uploaded_files=uploaded_files,
        conversation_state=session["conversation_state"],
    )
    session["conversation_state"] = pipeline_result["conversation_state"]
    session["messages"] = append_message(session["messages"], "assistant", pipeline_result["response"])

    save_messages(session["history_path"], session["messages"])
    save_conversation_state(session["state_path"], session["conversation_state"])

    return {
        "session_id": session["session_id"],
        "visible_prompt": visible_prompt,
        "response": pipeline_result["response"],
        "messages": session["messages"],
        "conversation_state": session["conversation_state"],
        "ui_payload": pipeline_result.get("ui_payload", {}),
        "history_path": session["history_path"],
        "state_path": session["state_path"],
    }
