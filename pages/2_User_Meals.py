import io
import json
import os
from typing import Dict, List

import streamlit as st
from PIL import Image

from database import fetch_all_results
from theme import image_wide, inject_theme, render_sidebar_nav

st.set_page_config(
    page_title="Мои приёмы пищи",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_theme()
render_sidebar_nav()

st.title("Мои приёмы пищи")
st.caption("История анализов (сначала новые). Калории — по строке справочника, на вашу граммовку и на порцию ИИ.")

if not os.environ.get("DATABASE_URL"):
    st.info("История хранится в PostgreSQL. Укажите DATABASE_URL в окружении сервера.")
    st.stop()

rows = fetch_all_results(limit=200)
if not rows:
    st.write("Пока нет записей. Сначала сделайте анализ на главной странице.")
    st.stop()

def _algo_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("algorithm_results")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


for row in rows:
    created = row.get("created_at")
    if hasattr(created, "strftime"):
        ts = created.strftime("%Y-%m-%d %H:%M")
    else:
        ts = str(created or "")

    ok = row.get("verification_status")
    badge = "✅ Подтверждено" if ok else "❌ Не подтверждено"
    img_bytes = row.get("image_jpeg")
    algo = _algo_dict(row)
    meal_items: List[Dict[str, Any]] = algo.get("meal_items") or []
    multi_saved = bool(algo.get("multi_plate")) and isinstance(meal_items, list) and len(meal_items) >= 2

    with st.container():
        st.markdown(f"#### {row.get('user_name', '')} · {ts}")
        st.markdown(f"**{badge}**")

        if img_bytes:
            try:
                image_wide(Image.open(io.BytesIO(img_bytes)))
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
        if multi_saved:
            with st.expander("Несколько блюд на снимке (сохранённый разбор)"):
                for i, it in enumerate(meal_items):
                    if not isinstance(it, dict):
                        continue
                    role = str(it.get("role") or "—")
                    gnm = str(it.get("gemini_name") or "—")
                    gp = float(it.get("gemini_portion") or 0)
                    ua = float(it.get("user_portion_allocated") or 0)
                    vr = "✅" if it.get("verified") else "❌"
                    st.markdown(
                        f"**{i + 1}.** {role} · {gnm} · ИИ ~{gp:.0f} г · "
                        f"ваша доля ~{ua:.0f} г {vr}"
                    )
                    rs = str(it.get("reasoning") or "").strip()
                    if rs:
                        st.caption(rs[:280] + ("…" if len(rs) > 280 else ""))
        en = row.get("matched_db_name_en") or ""
        db_line = row.get("matched_db_dish") or "—"
        if en:
            db_line = f"{db_line} ({en})"
        st.markdown(
            '<div class="card">'
            f"<strong>База</strong><br/>{db_line}"
            "</div>",
            unsafe_allow_html=True,
        )

        # Калории — отдельный заметный блок (главное на этой странице)
        st.markdown("##### Калории из базы данных")
        ku = row.get("kcal_user_portion")
        kg = row.get("kcal_gemini_portion")
        up = float(row.get("user_portion") or 0)
        gp = float(row.get("gemini_portion") or 0)

        k1, k2 = st.columns(2)
        with k1:
            if ku is not None:
                st.metric(
                    f"На вашу порцию ({up:.0f} г)",
                    f"{float(ku):.0f} ккал",
                )
            else:
                st.metric("На вашу порцию", "—")
                st.caption("нет данных в записи")
        with k2:
            if kg is not None and gp > 0:
                st.metric(
                    f"На порцию ИИ ({gp:.0f} г)",
                    f"{float(kg):.0f} ккал",
                )
            elif kg is not None:
                st.metric("На порцию ИИ", f"{float(kg):.0f} ккал")
            else:
                st.metric("На порцию ИИ", "—")
                st.caption("нет данных в записи")

        if ku is None and kg is None:
            st.info(
                "Для этой записи калории не сохранены. "
                "Так бывает у старых записей до обновления приложения или если не было совпадения в справочнике. "
                "Сделайте новый анализ на главной — ккал появятся здесь."
            )

        st.divider()
