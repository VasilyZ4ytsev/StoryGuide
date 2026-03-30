import os
import sys
import uuid
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from src.app_service import (
    get_supported_chat_file_types,
    load_chat_session,
    process_chat_turn,
    reset_chat_session,
)
from src.conversation_state import build_context_summary
from src.ui_presenter import (
    build_genre_distribution,
    build_rating_chart_frame,
    build_recommendation_dataframe,
    build_supported_types_caption,
    build_year_distribution,
    format_metric_value,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_FILE_TYPES = get_supported_chat_file_types()

SEARCH_MODE_LABELS = {
    "title_match": "По названию",
    "hybrid_query": "Гибридный",
    "filter_only": "Фильтры",
    "empty": "Пусто",
    "unknown": "Неизвестно",
}


def get_bare_mode_message():
    return (
        "StoryGuide is a Streamlit app.\n"
        "Run it with one of these commands:\n"
        "  streamlit run app.py\n"
        "  streamlit run src/main.py"
    )


def inject_app_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 24rem),
                radial-gradient(circle at bottom right, rgba(180, 83, 9, 0.10), transparent 22rem),
                #f7f3ea;
        }

        .st-key-result_window {
            position: relative;
        }

        @media (min-width: 1100px) {
            .st-key-result_window {
                position: sticky;
                top: 1rem;
            }
        }

        .st-key-result_window [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(249, 246, 239, 0.98));
            border: 1px solid rgba(31, 41, 55, 0.12);
            border-radius: 26px;
            box-shadow: 0 24px 50px rgba(31, 41, 55, 0.10);
            padding: 0.6rem 0.75rem 0.8rem;
        }

        .st-key-result_window [data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }

        .st-key-result_window button[data-baseweb="tab"] {
            border-radius: 999px;
            border: 1px solid rgba(31, 41, 55, 0.08);
            background: rgba(255, 255, 255, 0.75);
            padding: 0.35rem 0.9rem;
        }

        .st-key-result_window button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(15, 118, 110, 0.12);
            border-color: rgba(15, 118, 110, 0.22);
        }

        .sg-window-head {
            padding: 0.25rem 0.15rem 0.85rem;
            border-bottom: 1px solid rgba(31, 41, 55, 0.08);
            margin-bottom: 0.9rem;
        }

        .sg-window-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #5f6b7a;
            margin-bottom: 0.45rem;
        }

        .sg-window-dots {
            display: inline-flex;
            gap: 0.25rem;
        }

        .sg-window-dots span {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 999px;
            display: inline-block;
        }

        .sg-window-dots span:nth-child(1) { background: #ef4444; }
        .sg-window-dots span:nth-child(2) { background: #f59e0b; }
        .sg-window-dots span:nth-child(3) { background: #10b981; }

        .sg-window-title {
            margin: 0;
            color: #102a43;
            font-size: clamp(1.8rem, 1.25rem + 1.2vw, 2.55rem);
            line-height: 1.05;
        }

        .sg-window-subtitle {
            margin: 0.45rem 0 0;
            color: #5f6b7a;
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .sg-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.85rem;
        }

        @media (max-width: 640px) {
            .sg-metric-grid {
                grid-template-columns: 1fr;
            }
        }

        .sg-metric-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(31, 41, 55, 0.09);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 7.25rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
        }

        .sg-metric-label {
            color: #6b7280;
            font-size: 0.86rem;
            line-height: 1.35;
        }

        .sg-metric-value {
            color: #102a43;
            font-size: clamp(1.8rem, 1.4rem + 0.9vw, 2.4rem);
            line-height: 1;
            margin: 0.6rem 0 0.15rem;
            font-weight: 700;
        }

        .sg-metric-caption {
            color: #7b8794;
            font-size: 0.84rem;
            line-height: 1.4;
        }

        .sg-note-card {
            background: linear-gradient(180deg, rgba(15, 118, 110, 0.07), rgba(15, 118, 110, 0.02));
            border: 1px solid rgba(15, 118, 110, 0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
        }

        .sg-note-label {
            display: block;
            color: #0f766e;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .sg-note-body {
            color: #304250;
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0;
        }

        .sg-empty-state {
            border: 1px dashed rgba(31, 41, 55, 0.16);
            border-radius: 20px;
            padding: 1.3rem 1.1rem;
            color: #5f6b7a;
            background: rgba(255, 255, 255, 0.65);
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_session_id_from_query_params():
    if hasattr(st, "query_params"):
        raw_session_id = st.query_params.get("sid", "")
        if isinstance(raw_session_id, list):
            raw_session_id = raw_session_id[0] if raw_session_id else ""
        session_id = str(raw_session_id or "").strip()
        if session_id:
            return session_id

        session_id = uuid.uuid4().hex
        st.query_params["sid"] = session_id
        return session_id

    raw_query_params = st.experimental_get_query_params().get("sid", [""])
    session_id = str(raw_query_params[0] if raw_query_params else "").strip()
    if session_id:
        return session_id

    session_id = uuid.uuid4().hex
    st.experimental_set_query_params(sid=session_id)
    return session_id


def initialize_streamlit_session():
    session_id = get_session_id_from_query_params()
    session_snapshot = load_chat_session(BASE_DIR, session_id)

    if st.session_state.get("session_id") != session_snapshot["session_id"]:
        st.session_state.session_id = session_snapshot["session_id"]
        st.session_state.history_path = session_snapshot["history_path"]
        st.session_state.state_path = session_snapshot["state_path"]
        st.session_state.messages = session_snapshot["messages"]
        st.session_state.conversation_state = session_snapshot["conversation_state"]
        st.session_state.last_result = None

    st.session_state.setdefault("history_path", session_snapshot["history_path"])
    st.session_state.setdefault("state_path", session_snapshot["state_path"])
    st.session_state.setdefault("messages", session_snapshot["messages"])
    st.session_state.setdefault("conversation_state", session_snapshot["conversation_state"])
    st.session_state.setdefault("last_result", None)


def handle_reset(clear_messages):
    session_snapshot = reset_chat_session(BASE_DIR, st.session_state.session_id, clear_messages=clear_messages)
    st.session_state.messages = session_snapshot["messages"]
    st.session_state.conversation_state = session_snapshot["conversation_state"]
    st.session_state.last_result = None
    st.rerun()


def render_sidebar():
    with st.sidebar:
        st.subheader("Текущий контекст")
        st.caption(f"SID: {st.session_state.session_id}")
        for line in build_context_summary(st.session_state.conversation_state):
            st.caption(line)

        st.divider()
        st.subheader("Управление")
        if st.button("Новая тема", use_container_width=True):
            handle_reset(clear_messages=True)

        st.divider()
        st.subheader("Поддерживаемые файлы")
        st.caption(build_supported_types_caption(CHAT_FILE_TYPES))


def render_message_history():
    if not st.session_state.messages:
        st.info(
            "Напишите, какой фильм вам нравится, опишите сюжет или прикрепите постер/текстовый файл."
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def format_search_mode(search_mode):
    normalized = str(search_mode or "unknown").strip().lower()
    return SEARCH_MODE_LABELS.get(normalized, normalized.replace("_", " ").title())


def render_window_header():
    st.html(
        """
        <div class="sg-window-head">
          <div class="sg-window-eyebrow">
            <span class="sg-window-dots">
              <span></span><span></span><span></span>
            </span>
            Result Window
          </div>
          <h2 class="sg-window-title">Последний результат</h2>
          <p class="sg-window-subtitle">
            Отдельная панель с итогом последнего запроса: рекомендации, графики и диагностика.
          </p>
        </div>
        """
    )


def render_metric_cards(payload):
    metrics = payload.get("metrics", {})
    cards = [
        {
            "label": "Рекомендаций",
            "value": str(metrics.get("recommendation_count", 0)),
            "caption": "Сколько фильмов попало в текущий ответ.",
        },
        {
            "label": "Лимит",
            "value": str(payload.get("requested_limit", 0)),
            "caption": "Сколько рекомендаций запросил пользователь.",
        },
        {
            "label": "Режим поиска",
            "value": format_search_mode(payload.get("search_mode", "unknown")),
            "caption": "Как StoryGuide нашёл текущую подборку.",
        },
        {
            "label": "Ходов в теме",
            "value": str(metrics.get("turn_count", 0)),
            "caption": "Сколько сообщений уже было в этой теме.",
        },
    ]

    cards_html = "".join(
        f"""
        <article class="sg-metric-card">
          <span class="sg-metric-label">{escape(item['label'])}</span>
          <strong class="sg-metric-value">{escape(item['value'])}</strong>
          <span class="sg-metric-caption">{escape(item['caption'])}</span>
        </article>
        """
        for item in cards
    )
    st.html(f'<div class="sg-metric-grid">{cards_html}</div>')


def render_note_card(label, body):
    if not str(body or "").strip():
        return

    st.html(
        f"""
        <div class="sg-note-card">
          <span class="sg-note-label">{escape(label)}</span>
          <p class="sg-note-body">{escape(str(body))}</p>
        </div>
        """
    )


def render_result_dashboard():
    payload = st.session_state.get("last_result")
    expander_label = "Последний результат"
    if payload:
        recommendation_count = payload.get("metrics", {}).get("recommendation_count", 0)
        expander_label = f"Последний результат: {recommendation_count} рекомендаций"

    with st.expander(expander_label, expanded=False):
        with st.container(key="result_window", border=True):
            render_window_header()

            if not payload:
                st.html(
                    """
                    <div class="sg-empty-state">
                      После первого запроса здесь появится отдельное окно с рекомендациями,
                      графиками распределения и технической диагностикой пайплайна.
                    </div>
                    """
                )
                return

            render_metric_cards(payload)

            source_movie = payload.get("source_movie")
            if source_movie:
                source_text = (
                    f"{source_movie['title']} | "
                    f"год: {format_metric_value(source_movie.get('year'))} | "
                    f"рейтинг: {format_metric_value(source_movie.get('rating'))}"
                )
                render_note_card("Фильм-источник", source_text)

            if payload.get("active_filters_text"):
                render_note_card("Активные фильтры", payload["active_filters_text"])

            recommendation_tab, charts_tab, diagnostics_tab = st.tabs(
                ["Рекомендации", "Графики", "Диагностика"]
            )

            with recommendation_tab:
                rows = payload.get("recommendation_rows", [])
                dataframe = build_recommendation_dataframe(rows)
                if dataframe.empty:
                    st.info("По текущему запросу рекомендации пока не сформированы.")
                else:
                    st.dataframe(dataframe, use_container_width=True, hide_index=True)

            with charts_tab:
                rows = payload.get("recommendation_rows", [])
                chart_available = False

                rating_frame = build_rating_chart_frame(rows)
                if not rating_frame.empty:
                    chart_available = True
                    st.caption("Рейтинги рекомендаций")
                    st.bar_chart(rating_frame)

                genre_frame = build_genre_distribution(rows)
                if not genre_frame.empty:
                    chart_available = True
                    st.caption("Жанровое распределение")
                    st.bar_chart(genre_frame.set_index("Жанр"))

                year_frame = build_year_distribution(rows)
                if not year_frame.empty:
                    chart_available = True
                    st.caption("Распределение по годам")
                    st.line_chart(year_frame.set_index("Год"))

                if not chart_available:
                    st.info("Графики появятся, когда в ответе будет достаточно данных для визуализации.")

            with diagnostics_tab:
                if payload.get("signal_lines"):
                    signal_frame = pd.DataFrame({"Сигнал": payload["signal_lines"]})
                    st.dataframe(signal_frame, use_container_width=True, hide_index=True)

                st.json(
                    {
                        "input_summary": payload.get("input_summary"),
                        "combined_query": payload.get("combined_query"),
                        "intent": payload.get("intent"),
                        "intent_score": payload.get("intent_score"),
                        "nlp": payload.get("nlp", {}),
                        "rule_report": payload.get("rule_report", {}),
                    }
                )


def process_chat_submission(chat_value):
    if hasattr(chat_value, "text"):
        user_text = chat_value.text or ""
        uploaded_files = list(getattr(chat_value, "files", []) or [])
    else:
        user_text = str(chat_value or "")
        uploaded_files = []

    if not user_text.strip() and not uploaded_files:
        return

    try:
        with st.spinner("Подбираю рекомендации и анализирую вложения..."):
            result = process_chat_turn(
                BASE_DIR,
                st.session_state.session_id,
                user_text=user_text,
                uploaded_files=uploaded_files,
            )
    except Exception as error:
        st.error(f"Не удалось обработать запрос: {error}")
        return

    st.session_state.messages = result["messages"]
    st.session_state.conversation_state = result["conversation_state"]
    st.session_state.last_result = result.get("ui_payload", {})
    st.rerun()


def main():
    if get_script_run_ctx(suppress_warning=True) is None:
        print(get_bare_mode_message())
        return

    st.set_page_config(page_title="StoryGuide", layout="wide")
    inject_app_styles()
    initialize_streamlit_session()
    render_sidebar()

    st.title("StoryGuide")
    st.caption("Кино-ассистент с рекомендациями, OCR и загрузкой файлов.")

    chat_column, dashboard_column = st.columns((1.28, 0.92), gap="large")
    with chat_column:
        render_message_history()
    with dashboard_column:
        render_result_dashboard()

    chat_value = st.chat_input(
        "Введите запрос по фильмам или прикрепите файл...",
        accept_file="multiple",
        file_type=CHAT_FILE_TYPES,
    )
    if chat_value:
        process_chat_submission(chat_value)


if __name__ == "__main__":
    main()
