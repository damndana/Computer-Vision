"""Shared mobile-first styles and navigation for Streamlit pages."""

import streamlit as st


MOBILE_CSS = """
<style>
  html, body, [data-testid="stAppViewContainer"] {
    overflow-x: hidden;
  }
  .block-container {
    max-width: 28rem !important;
    padding-top: 0.75rem !important;
    padding-bottom: 3rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
  @media (min-width: 640px) {
    .block-container { max-width: 36rem !important; }
  }
  .stApp {
    background: #f7f7f5 !important;
    color: #1a1a1a !important;
  }
  .stApp, .stApp p, .stApp span, .stApp label {
    color: #1a1a1a !important;
  }
  h1, h2, h3 { color: #111 !important; font-weight: 600 !important; letter-spacing: -0.02em; }
  .stButton > button {
    width: 100%;
    min-height: 3.25rem;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    border: none !important;
    background: #1a1a1a !important;
    color: #fff !important;
    transition: transform 0.15s ease, opacity 0.15s ease;
  }
  .stButton > button:hover { opacity: 0.92; transform: scale(1.01); }
  div[data-testid="stFileUploader"] section {
    padding: 1.25rem !important;
    border-radius: 16px !important;
    border: 1px dashed #c8c8c4 !important;
    background: #fff !important;
  }
  .stTextInput input, .stNumberInput input {
    border-radius: 12px !important;
    min-height: 2.75rem !important;
    font-size: 1rem !important;
  }
  .card {
    background: #fff;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #e8e8e4;
  }
  .card-best { border-color: #2d6a4f; box-shadow: 0 0 0 2px rgba(45,106,79,0.15); }
  .muted { color: #5c5c58 !important; font-size: 0.9rem; }
  .badge-ok { display:inline-block; padding:0.35rem 0.65rem; border-radius:999px; background:#d8f3dc; color:#1b4332; font-weight:600; font-size:0.9rem; }
  .badge-bad { display:inline-block; padding:0.35rem 0.65rem; border-radius:999px; background:#ffccd5; color:#7f1d1d; font-weight:600; font-size:0.9rem; }
  section[data-testid="stSidebar"] {
    background: #fafaf8 !important;
    border-right: 1px solid #e8e8e4 !important;
  }
  div[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
  .nav-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #888 !important; margin-bottom: 0.5rem; }
</style>
"""


def inject_theme():
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)


def render_sidebar_nav():
    with st.sidebar:
        st.markdown('<p class="nav-title">Меню</p>', unsafe_allow_html=True)
        st.page_link("app.py", label="Проверка блюда", icon="🍽️")
        st.page_link("pages/2_User_Meals.py", label="Мои приёмы пищи", icon="📋")
        st.divider()
