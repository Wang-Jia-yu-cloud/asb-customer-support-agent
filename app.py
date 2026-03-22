import os
import streamlit as st
from dotenv import load_dotenv
from src.agent import run_crew

load_dotenv()

st.set_page_config(
    page_title="ASB Virtual Support",
    page_icon="🏦",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #e8e8e8; }

    .asb-header {
        background-color: #FFCC00;
        padding: 36px 48px;
        display: flex;
        align-items: center;
        gap: 28px;
        border-radius: 0 0 20px 20px;
        margin-bottom: 40px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .asb-header-title {
        font-size: 52px;
        font-weight: 700;
        color: #000;
        margin: 0;
    }
    .asb-header-subtitle {
        font-size: 32px;
        color: #333;
        margin-top: 6px;
    }
    .asb-avatar {
        background-color: #000;
        color: #FFCC00;
        border-radius: 50%;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 38px;
        flex-shrink: 0;
    }
    .asb-avatar-sm {
        background-color: #000;
        color: #FFCC00;
        border-radius: 50%;
        width: 72px;
        height: 72px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 24px;
        flex-shrink: 0;
        margin-top: 4px;
    }
    .status-dot {
        width: 16px;
        height: 16px;
        background-color: #22c55e;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
    }
    .secure-badge {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
        border-radius: 20px;
        padding: 10px 22px;
        font-size: 26px;
        white-space: nowrap;
        font-weight: 500;
    }

    .bubble-row-user {
        display: flex;
        justify-content: flex-end;
        margin: 20px 0;
    }
    .bubble-row-assistant {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 20px;
        margin: 20px 0;
    }
    .user-bubble {
        background-color: #1a1a1a;
        color: white;
        padding: 24px 34px;
        border-radius: 32px 32px 4px 32px;
        max-width: 72%;
        font-size: 32px;
        line-height: 1.75;
    }
    .assistant-bubble {
        background-color: white;
        color: #1a1a1a;
        padding: 24px 34px;
        border-radius: 32px 32px 32px 4px;
        max-width: 72%;
        font-size: 32px;
        line-height: 1.75;
        border: 1px solid #ddd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .assistant-bubble p,
    .assistant-bubble li,
    .assistant-bubble ol,
    .assistant-bubble ul,
    .assistant-bubble strong,
    .assistant-bubble a {
        font-size: 32px !important;
        line-height: 1.75 !important;
    }
    .bubble-name {
        font-size: 24px;
        color: #999;
        margin-bottom: 8px;
        font-weight: 500;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button p,
    div[data-testid="stButton"] > button,
    .stButton > button {
        background-color: white !important;
        border: 2.5px solid #FFCC00 !important;
        border-radius: 32px !important;
        color: #1a1a1a !important;
        font-size: 26px !important;
        padding: 20px 28px !important;
        font-weight: 500 !important;
        transition: background-color 0.2s;
        width: 100%;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #FFCC00 !important;
    }

    [data-testid="stChatInput"] textarea {
        caret-color: #1a1a1a !important;
        font-size: 30px !important;
        border-radius: 36px !important;
        padding: 24px 32px !important;
        background-color: white !important;
        color: #1a1a1a !important;
        min-height: 80px !important;
    }
    [data-testid="stBottom"] {
        background-color: #e8e8e8 !important;
        padding: 24px 0;
    }
    [data-testid="stBottom"] > div {
        background-color: #e8e8e8 !important;
    }

    section[data-testid="stMain"] > div > div > div > div {
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="asb-header">
    <div class="asb-avatar">KM</div>
    <div style="flex:1">
        <div class="asb-header-title">Kiri Mahana</div>
        <div class="asb-header-subtitle">
            <span class="status-dot"></span>
            <span style="color:#166534;font-weight:600;">Available now</span>
            &nbsp;·&nbsp; ASB Virtual Support Specialist
        </div>
    </div>
    <span class="secure-badge">🔒 Secure session</span>
</div>
""", unsafe_allow_html=True)

SUGGESTIONS = [
    "How do I reset my password?",
    "What are the daily payment limits?",
    "How do I set up alerts?",
    "I want to speak to someone",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "state" not in st.session_state:
    st.session_state.state = {}


def render_messages():
    import markdown
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="bubble-row-user">
                <div class="user-bubble">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            html_content = markdown.markdown(msg["content"])
            st.markdown(f"""
            <div class="bubble-row-assistant">
                <div class="asb-avatar-sm">KM</div>
                <div style="max-width:72%">
                    <div class="bubble-name">Kiri · ASB Virtual Support</div>
                    <div class="assistant-bubble">{html_content}</div>
                </div>
            </div>""", unsafe_allow_html=True)


render_messages()

if not st.session_state.messages:
    st.markdown("""
    <div class="bubble-row-assistant">
        <div class="asb-avatar-sm">KM</div>
        <div>
            <div class="bubble-name">Kiri · ASB Virtual Support</div>
            <div class="assistant-bubble">
                Kia ora! I'm Kiri, your ASB Virtual Support Specialist.<br><br>
                How can I help you today?
            </div>
        </div>
    </div>
    <br>""", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, key=f"sug_{i}"):
            st.session_state.pending = suggestion
            st.rerun()

TYPING_BUBBLE = """
<div class="bubble-row-assistant">
    <div class="asb-avatar-sm">KM</div>
    <div>
        <div class="bubble-name">Kiri · ASB Virtual Support</div>
        <div class="assistant-bubble" style="color:#999;font-style:italic;">
            Kiri is typing...
        </div>
    </div>
</div>"""

def user_bubble(text):
    return f"""
    <div class="bubble-row-user">
        <div class="user-bubble">{text}</div>
    </div>"""

if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(user_bubble(prompt), unsafe_allow_html=True)
    st.markdown(TYPING_BUBBLE, unsafe_allow_html=True)
    response = run_crew(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if prompt := st.chat_input("Ask Kiri anything about your banking..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(user_bubble(prompt), unsafe_allow_html=True)
    st.markdown(TYPING_BUBBLE, unsafe_allow_html=True)
    response = run_crew(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
