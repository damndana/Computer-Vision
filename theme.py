"""Shared mobile-first styles and navigation for Streamlit pages."""

import streamlit as st


def image_wide(image, **kwargs):
    """st.image full width: works on Streamlit with use_container_width or older use_column_width."""
    try:
        st.image(image, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image, use_column_width=True, **kwargs)


MOBILE_CSS = """
<style>
  :root {
    --bg: #f4f3f0;
    --surface: #ffffff;
    --text: #1c1917;
    --text-muted: #57534e;
    --border: #e7e5e4;
    --accent: #0f766e;
    --accent-hover: #0d9488;
    --accent-text: #ffffff;
    --radius: 14px;
    --shadow: 0 1px 2px rgba(28, 25, 23, 0.06);
  }

  html, body, [data-testid="stAppViewContainer"] {
    overflow-x: hidden;
  }

  .stApp {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
  }

  .block-container {
    max-width: 28rem !important;
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
  }
  @media (min-width: 640px) {
    .block-container { max-width: 36rem !important; }
  }

  /* Main text — only outside interactive controls (fixes invisible button labels) */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
  }
  .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
  }
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  .stMarkdown p {
    color: var(--text) !important;
  }

  /* ----- Buttons: force label contrast on ALL nested nodes ----- */
  .stButton > button {
    width: 100%;
    min-height: 3.1rem;
    font-family: inherit !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
    transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
  }

  /* Primary */
  .stButton > button[kind="primary"],
  .stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, var(--accent-hover) 0%, var(--accent) 100%) !important;
    border: none !important;
    color: var(--accent-text) !important;
  }
  .stButton > button[kind="primary"] *,
  .stButton > button[data-testid="baseButton-primary"] * {
    color: var(--accent-text) !important;
  }
  .stButton > button[kind="primary"]:hover,
  .stButton > button[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 4px 14px rgba(15, 118, 110, 0.35) !important;
    transform: translateY(-1px);
  }

  /* Secondary / default — light surface, dark readable text */
  .stButton > button[kind="secondary"],
  .stButton > button[data-testid="baseButton-secondary"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text) !important;
    box-shadow: var(--shadow) !important;
  }
  .stButton > button[kind="secondary"] *,
  .stButton > button[data-testid="baseButton-secondary"] * {
    color: var(--text) !important;
  }
  .stButton > button[kind="secondary"]:hover,
  .stButton > button[data-testid="baseButton-secondary"]:hover {
    border-color: #d6d3d1 !important;
    background: #fafaf9 !important;
  }

  /* Fallback if kind attribute missing */
  .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text) !important;
  }
  .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) * {
    color: var(--text) !important;
  }

  /* Labels for inputs */
  .stTextInput label, .stNumberInput label,
  .stCameraInput label, .stFileUploader label,
  label[data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
    font-weight: 500 !important;
  }

  .stTextInput input, .stNumberInput input {
    border-radius: 12px !important;
    min-height: 2.85rem !important;
    font-size: 1rem !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
  }

  div[data-testid="stFileUploader"] section {
    padding: 1.25rem !important;
    border-radius: var(--radius) !important;
    border: 2px dashed #d6d3d1 !important;
    background: var(--surface) !important;
  }
  div[data-testid="stFileUploader"] section small,
  div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
    color: var(--text-muted) !important;
  }

  [data-testid="stCameraInput"] > div {
    border-radius: var(--radius) !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
  }

  /* Sidebar nav links */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebarNavLink"] {
    border-radius: 12px !important;
    color: var(--text) !important;
  }
  [data-testid="stSidebarNavLink"] span {
    color: var(--text) !important;
  }
  [data-testid="stSidebarNavLink"]:hover {
    background: var(--bg) !important;
  }

  div[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
  .nav-title {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem !important;
  }

  .card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1rem 1.15rem;
    margin: 0.5rem 0;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
  }
  .card-best {
    border-color: #0f766e;
    box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.12);
  }
  .muted { color: var(--text-muted) !important; font-size: 0.9rem; }
  .badge-ok {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: #ccfbf1;
    color: #115e59;
    font-weight: 600;
    font-size: 0.9rem;
  }
  .badge-bad {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: #ffe4e6;
    color: #9f1239;
    font-weight: 600;
    font-size: 0.9rem;
  }

  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.25rem 0;
  }

  /* Dividers Streamlit */
  [data-testid="stHorizontalBlock"] + hr,
  .stDivider {
    background-color: var(--border) !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--surface) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
  }
  .streamlit-expanderHeader svg {
    fill: var(--text) !important;
  }

  /* Spinner */
  [data-testid="stSpinner"] {
    color: var(--accent) !important;
  }
</style>
"""


def inject_theme():
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)


def render_sidebar_nav() -> None:
    with st.sidebar:
        st.markdown('<p class="nav-title">Меню</p>', unsafe_allow_html=True)
        st.page_link("app.py", label="Проверка блюда", icon="🍽️")
        st.page_link("pages/2_User_Meals.py", label="Мои приёмы пищи", icon="📋")
        st.divider()
