import streamlit as st

from logic import check_rules, load_rules
from mock_data import test_entity as default_data

st.title("Отладчик продукционной системы")
st.write("### Настройка входных данных")

rules = load_rules()
genre_options = sorted(
    set(default_data["genres"] + rules["lists"]["whitelist"] + rules["lists"]["blacklist"])
)

user_year = st.sidebar.number_input(
    "Год релиза:",
    value=int(default_data["release_year"]),
)
user_genres = st.sidebar.multiselect(
    "Жанры:",
    options=genre_options,
    default=default_data["genres"],
)
user_verified = st.sidebar.checkbox(
    "Профиль подтвержден (True/False):",
    value=default_data["is_verified"],
)

if st.button("Запустить проверку"):
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
