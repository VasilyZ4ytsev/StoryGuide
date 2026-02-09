import streamlit as st

st.title("Привет, Проект!")
st.write("Если вы видите этот текст, значит Streamlit работает отлично.")

if st.button("Нажми меня"):
    st.success("Все системы в норме!")
