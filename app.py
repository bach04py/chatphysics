import streamlit as st
from chatbot import build_chatbot

# Load chatbot 1 lần
@st.cache_resource
def load_bot():
    return build_chatbot()

bot = load_bot()

st.set_page_config(page_title="NLP Chatbot", layout="wide")

st.title("📘 NLP RAG Chatbot")

# Lưu lịch sử chat
if "history" not in st.session_state:
    st.session_state.history = []

# Input
user_input = st.text_input("Ask a nlp question:")

if user_input:
    result = bot.invoke({"question": user_input})
    answer = result["answer"]

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", answer))

# Hiển thị chat
for role, text in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {text}")
    else:
        st.markdown(f"**🤖 {role}:** {text}")