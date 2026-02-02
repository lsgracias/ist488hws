import streamlit as st

hw1 = st.Page("hws/HW1.py", title="HW 1", icon="📄")
hw2 = st.Page("hws/HW2.py", title="HW 2 - URL Summarizer", icon="🌐", default=True)

pg = st.navigation([hw1, hw2])

st.set_page_config(page_title="HW Manager", page_icon="📚")

pg.run()