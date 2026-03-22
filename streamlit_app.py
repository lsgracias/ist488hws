import streamlit as st

hw1 = st.Page("hws/HW1.py", title="HW 1", icon="📄")
hw2 = st.Page("hws/HW2.py", title="HW 2 - URL Summarizer", icon="🌐")
hw3 = st.Page("hws/HW3.py", title="HW 3 - URL Chatbot", icon=":material/description:")
hw4 = st.Page("hws/HW4.py", title ="HW 4 - iSchool Org Chatbot", icon=":material/description:")
hw5 = st.Page("hws/HW5.py", title ="HW 5 - SU Org Memory Chatbot", icon=":material/description:")
hw7 = st.Page("hws/HW7.py", title = "HW 7 - News Information Bot", icon=":material/description:", default= True)

pg = st.navigation([hw1, hw2, hw3, hw4, hw5, hw7])

st.set_page_config(page_title="HW Manager", page_icon="📚")

pg.run()