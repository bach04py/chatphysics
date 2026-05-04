import streamlit as st
from chatbot import build_chatbot
import uuid

st.set_page_config(page_title="Physics AI", layout="wide")

# =========================
# CACHE BOT
# =========================
@st.cache_resource
def load_bot():
    return build_chatbot()

bot = load_bot()

# =========================
# SESSION STATE INIT
# =========================
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

# =========================
# SIDEBAR (CHAT LIST)
# =========================
with st.sidebar:
    st.title("💬 Chats")

    # New chat
    if st.button("➕ New Chat"):
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat = chat_id
        st.session_state.chats[chat_id] = []
        st.rerun()

    # Search
    search = st.text_input("🔍 Search chat")

    # List chats
    for chat_id, messages in st.session_state.chats.items():

        title = "New Chat"
        if messages:
            title = messages[0]["content"][:30]

        if search:
            if not any(search.lower() in m["content"].lower() for m in messages):
                continue

        if st.button(title, key=chat_id):
            st.session_state.current_chat = chat_id
            st.rerun()

# =========================
# MAIN CHAT AREA
# =========================
st.title("⚡ Physics AI Chatbot")

messages = st.session_state.chats[st.session_state.current_chat]

# display messages
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# INPUT
# =========================
if prompt := st.chat_input("Ask something..."):

    # save user message
    messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # =========================
    # BOT RESPONSE (STREAM)
    # =========================
    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_text = ""

        try:
            stream_fn = getattr(bot, "stream", None)
            if stream_fn is None:
                stream_fn = bot.stream_answer

            for chunk in stream_fn(prompt):
                if not chunk:
                    continue

                full_text += str(chunk)
                placeholder.markdown(full_text)

        except Exception as e:
            full_text = f"⚠️ Error: {e}"
            placeholder.markdown(full_text)

        # save assistant response
        messages.append({
            "role": "assistant",
            "content": full_text
        })