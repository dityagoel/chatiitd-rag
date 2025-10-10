import streamlit as st  
from streamlit.runtime.scriptrunner import get_script_run_ctx
import os

st.set_page_config(page_title="CHATIITD", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.block-container { max-width: 700px; margin: auto; padding-top: 2rem; }
[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
    padding: 12px 16px;
    display: inline-block;
    max-width: 80%;
    word-wrap: break-word;
}
[data-testid="stChatMessage"][data-testid="assistant"] p {
    background: #E8EAF6;
    border-radius: 18px 18px 18px 0;
}
[data-testid="stChatMessage"][data-testid="user"] p {
    background: #DCF8C6;
    border-radius: 18px 18px 0 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 CHATIITD") 
st.markdown( """ Your personal IIT Delhi academic guide 🤖 Ask me about courses or institute rules. Type 'quit' to exit. """ )

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to continue.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key   

import agent as a 
ctx = get_script_run_ctx()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "quit" not in st.session_state:
    st.session_state.quit = False

for message in st.session_state.messages:
    role = "assistant" if message["role"] == "bot" else message["role"]
    avatar = "🎓" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        content = message["content"]
        if isinstance(content, dict):
            st.markdown(content.get("output", ""))
        else:
            st.markdown(content)

if not st.session_state.quit:
    user_input = st.chat_input("Type your message...")
    if user_input:
        if user_input.lower() == "quit":
            st.session_state.messages.append({
                "role": "bot",
                "content": {"output": "Chat ended. Refresh to start again."}
            })
            st.session_state.quit = True
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            session_id = ctx.session_id if ctx else "default_session"
            bot_response = a.invoke_memory_agent({"input": user_input}, session_id=session_id)
            st.session_state.messages.append({"role": "bot", "content": bot_response})

        st.rerun()

