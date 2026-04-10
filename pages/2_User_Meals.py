import io
import os

import streamlit as st
from PIL import Image

from database import fetch_all_results
from theme import inject_theme, render_sidebar_nav

st.set_page_config(
    page_title="Мои приёмы пищи",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_theme()
render_sidebar_nav()

st.title("Мои приёмы пищи")
st.caption("История анализов (сначала новые).")

if not os.environ.get("DATABASE_URL"):
    st.info("История хранится в PostgreSQL. Укажите DATABASE_URL в окружении сервера.")
    st.stop()

rows = fetch_all_results(limit=200)
if not rows:
    st.write("Пока нет записей. Сначала сделайте анализ на главной странице.")
    st.stop()

for row in rows:
    created = row.get("created_at")
    if hasattr(created, "strftime"):
        ts = created.strftime("%Y-%m-%d %H:%M")
    else:
        ts = str(created or "")

    ok = row.get("verification_status")
    badge = "✅ Подтверждено" if ok else "❌ Не подтверждено"
    img_bytes = row.get("image_jpeg")

    with st.container():
        st.markdown(f"#### {row.get('user_name', '')} · {ts}")
        st.markdown(f"**{badge}**")

        if img_bytes:
            try:
                st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
            except Exception:
                st.caption("(Изображение недоступно)")

        st.markdown(
            '<div class="card">'
            f"<strong>Вы</strong><br/>{row.get('user_dish_name') or '—'} · "
            f"{row.get('user_portion') or 0:.0f} г"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card">'
            f"<strong>ИИ</strong><br/>{row.get('gemini_dish_name') or '—'} · "
            f"{row.get('gemini_portion') or 0:.0f} г"
            "</div>",
            unsafe_allow_html=True,
        )
        en = row.get("matched_db_name_en") or ""
        db_line = row.get("matched_db_dish") or "—"
        if en:
            db_line = f"{db_line} ({en})"
        st.markdown(
            f'<div class="card"><strong>База</strong><br/>{db_line}</div>',
            unsafe_allow_html=True,
        )
        st.divider()
