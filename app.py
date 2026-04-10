import io
import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from PIL import Image

from database import load_database_csv, load_database_pg, save_result_row
from theme import image_wide, inject_theme, render_sidebar_nav

load_dotenv()

st.set_page_config(
    page_title="Nutristeppe — проверка блюда",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_theme()
render_sidebar_nav()

# -----------------------------------------------------------------------------
# Text normalization & synonyms (RU / EN)
# -----------------------------------------------------------------------------

DISH_SYNONYMS = {
    "картошка": "картофель",
    "картошки": "картофель",
    "помидор": "томат",
    "помидоры": "томат",
    "огурец": "огурцы",
    "макароны": "макаронные изделия",
    "паста": "макаронные изделия",
    "ризотто": "рис",
    "ceasar": "цезарь",
    "caesar": "цезарь",
    "pizza": "пицца",
    "burger": "бургер",
    "salad": "салат",
}


def normalize_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = t.replace("ё", "е")
    t = re.sub(r"\s+", " ", t)
    return t


def apply_synonyms(text_norm: str) -> str:
    out = text_norm
    for a, b in DISH_SYNONYMS.items():
        if a in out:
            out = out.replace(a, b)
    return out


def get_gemini_api_key() -> Optional[str]:
    for k in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return None


def compress_image_bytes(image: Image.Image, max_side: int = 1280, quality: int = 82) -> bytes:
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _meal_photo_bytes_set(img: Image.Image) -> None:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    st.session_state["meal_photo_bytes"] = buf.getvalue()


def _meal_photo_image_get() -> Optional[Image.Image]:
    raw = st.session_state.get("meal_photo_bytes")
    if not raw:
        return None
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _clear_meal_photo() -> None:
    st.session_state.pop("meal_photo_bytes", None)
    st.session_state.pop("meal_file_uploader", None)


# -----------------------------------------------------------------------------
# Search engine (fuzzy, token, Levenshtein-style, hybrid; RU + EN)
# -----------------------------------------------------------------------------


class DishSearchEngine:
    def __init__(self, database: pd.DataFrame):
        self.database = database

    @staticmethod
    def _row_names(row: pd.Series) -> Tuple[str, str]:
        ru = normalize_text(str(row.get("name", "") or ""))
        en = normalize_text(str(row.get("name_en", "") or ""))
        return ru, en

    def _score_query_vs_names(self, q: str, ru: str, en: str) -> float:
        scores: List[float] = []
        for name in (ru, en):
            if not name:
                continue
            scores.extend(
                [
                    float(fuzz.ratio(q, name)),
                    float(fuzz.partial_ratio(q, name)),
                    float(fuzz.token_sort_ratio(q, name)),
                    float(fuzz.token_set_ratio(q, name)),
                ]
            )
        return max(scores) if scores else 0.0

    def _invalid_query(self, query: str) -> bool:
        n = normalize_text(query)
        return not n or n in ("неизвестно", "unknown")

    def search_by_fuzzy_matching(self, query: str, threshold: float = 52.0) -> pd.DataFrame:
        if self._invalid_query(query):
            return pd.DataFrame()
        q = apply_synonyms(normalize_text(query))
        rows_out: List[Dict[str, Any]] = []
        for idx, row in self.database.iterrows():
            ru, en = self._row_names(row)
            s = self._score_query_vs_names(q, ru, en)
            if s >= threshold:
                d = row.to_dict()
                d["score"] = s
                d["index"] = idx
                rows_out.append(d)
        df = pd.DataFrame(rows_out)
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        return df

    def search_by_token_matching(self, query: str, min_jaccard: float = 0.28) -> pd.DataFrame:
        if self._invalid_query(query):
            return pd.DataFrame()
        q = apply_synonyms(normalize_text(query))
        q_tokens = set(q.split())
        rows_out: List[Dict[str, Any]] = []
        for idx, row in self.database.iterrows():
            ru, en = self._row_names(row)
            best = 0.0
            for name in (ru, en):
                if not name:
                    continue
                nt = set(name.split())
                if not q_tokens or not nt:
                    continue
                inter = len(q_tokens & nt)
                union = len(q_tokens | nt)
                j = inter / union if union else 0.0
                best = max(best, j)
            if best >= min_jaccard:
                d = row.to_dict()
                d["score"] = best * 100.0
                d["index"] = idx
                rows_out.append(d)
        df = pd.DataFrame(rows_out)
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        return df

    def search_by_levenshtein(self, query: str, threshold: float = 48.0) -> pd.DataFrame:
        if self._invalid_query(query):
            return pd.DataFrame()
        q = apply_synonyms(normalize_text(query))
        rows_out: List[Dict[str, Any]] = []
        for idx, row in self.database.iterrows():
            ru, en = self._row_names(row)
            s = self._score_query_vs_names(q, ru, en)
            if s >= threshold:
                d = row.to_dict()
                d["score"] = s
                d["index"] = idx
                rows_out.append(d)
        df = pd.DataFrame(rows_out)
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        return df

    def hybrid_search(self, query: str, top_n: int = 5) -> pd.DataFrame:
        parts = [
            self.search_by_fuzzy_matching(query, threshold=48),
            self.search_by_token_matching(query, min_jaccard=0.22),
            self.search_by_levenshtein(query, threshold=45),
        ]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True)
        key = "bls_code" if "bls_code" in combined.columns else "name"
        if key in combined.columns:
            combined = combined.sort_values("score", ascending=False).drop_duplicates(subset=[key], keep="first")
        else:
            combined = combined.sort_values("score", ascending=False).drop_duplicates(subset=["name"], keep="first")
        return combined.sort_values("score", ascending=False).head(top_n)

    def algorithm_bundle(self, query: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Primary hybrid table + JSON-safe summaries for all methods."""
        hybrid = self.hybrid_search(query, top_n=5)
        fuzzy = self.search_by_fuzzy_matching(query).head(5)
        token = self.search_by_token_matching(query).head(5)
        lev = self.search_by_levenshtein(query).head(5)

        def pack(df: pd.DataFrame) -> List[Dict[str, Any]]:
            if df.empty:
                return []
            out = []
            for _, r in df.iterrows():
                out.append(
                    {
                        "name": str(r.get("name", "")),
                        "name_en": str(r.get("name_en", "") or ""),
                        "score": float(r.get("score", 0)),
                    }
                )
            return out

        bundle = {
            "hybrid_top5": pack(hybrid),
            "fuzzy_top5": pack(fuzzy),
            "token_top5": pack(token),
            "levenshtein_top5": pack(lev),
        }
        return hybrid, bundle


# -----------------------------------------------------------------------------
# Verification (similarity thresholds, not strict equality)
# -----------------------------------------------------------------------------

USER_GEMINI_MIN = float(os.environ.get("VERIFY_USER_GEMINI_MIN", "72"))
GEMINI_DB_MIN = float(os.environ.get("VERIFY_GEMINI_DB_MIN", "68"))


def verification_scores(
    user_dish: str,
    gemini_dish: str,
    db_name: str,
    db_name_en: str,
) -> Dict[str, float]:
    u = apply_synonyms(normalize_text(user_dish))
    g = apply_synonyms(normalize_text(gemini_dish))
    db_ru = normalize_text(db_name)
    db_en = normalize_text(db_name_en or "")
    user_gemini = max(
        float(fuzz.token_sort_ratio(u, g)),
        float(fuzz.partial_ratio(u, g)),
    )
    gemini_db = 0.0
    for dn in (db_ru, db_en):
        if dn:
            gemini_db = max(
                gemini_db,
                float(fuzz.token_sort_ratio(g, dn)),
                float(fuzz.partial_ratio(g, dn)),
            )
    user_db = 0.0
    for dn in (db_ru, db_en):
        if dn:
            user_db = max(
                user_db,
                float(fuzz.token_sort_ratio(u, dn)),
                float(fuzz.partial_ratio(u, dn)),
            )
    return {
        "user_vs_gemini": user_gemini,
        "gemini_vs_db": gemini_db,
        "user_vs_db": user_db,
    }


def is_verified(user_dish: str, gemini_dish: str, db_name: str, db_name_en: str) -> Tuple[bool, Dict[str, Any]]:
    s = verification_scores(user_dish, gemini_dish, db_name, db_name_en)
    ok = s["user_vs_gemini"] >= USER_GEMINI_MIN and s["gemini_vs_db"] >= GEMINI_DB_MIN
    detail = {
        "thresholds": {"user_vs_gemini_min": USER_GEMINI_MIN, "gemini_vs_db_min": GEMINI_DB_MIN},
        "scores": s,
        "verified": ok,
    }
    return ok, detail


# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------


class GeminiMealAgent:
    def __init__(self, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def analyze_meal_image_jpeg(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        try:
            prompt = """
            Ты распознаёшь еду на фото. Верни до 4 блюд с порцией в граммах и уверенностью 0–1.
            Названия — на русском (можно коротко).
            Только JSON-массив:
            [{"dish_name": "...", "portion_grams": 200, "confidence": 0.9}]
            Если не уверен: {"dish_name": "неизвестно", "portion_grams": 0, "confidence": 0}
            """
            response = self.model.generate_content(
                [
                    prompt,
                    {"mime_type": "image/jpeg", "data": jpeg_bytes},
                ]
            )
            text = response.text or ""
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
            else:
                om = re.search(r"\{.*\}", text, re.DOTALL)
                parsed = [json.loads(om.group())] if om else []

            if not isinstance(parsed, list):
                parsed = [parsed]

            dishes: List[Dict[str, Any]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                dishes.append(
                    {
                        "dish_name": str(item.get("dish_name", "неизвестно")),
                        "portion_grams": float(item.get("portion_grams", 0) or 0),
                        "confidence": float(item.get("confidence", 0) or 0),
                    }
                )
            if not dishes:
                dishes = [{"dish_name": "неизвестно", "portion_grams": 0, "confidence": 0}]
            return {"dishes": dishes}
        except Exception as e:
            return {"dishes": [], "error": str(e)}


# -----------------------------------------------------------------------------
# Nutrition (optional display)
# -----------------------------------------------------------------------------


def nutrition_for_row(row: pd.Series, portion_g: float) -> Dict[str, float]:
    base = row.get("serving_size_g", 100)
    if pd.isna(base) or float(base) == 0:
        base = 100.0
    ratio = float(portion_g) / float(base)
    return {
        "kilocalories": float(row.get("kilocalories", 0) or 0) * ratio,
        "protein_g": float(row.get("protein", 0) or 0) * ratio / 1000.0,
        "fat_g": float(row.get("fat", 0) or 0) * ratio / 1000.0,
        "carb_g": float(row.get("carbohydrate", 0) or 0) * ratio / 1000.0,
    }


@st.cache_data(ttl=120)
def load_database() -> pd.DataFrame:
    base = pathlib.Path(__file__).parent
    df = load_database_pg()
    if df is not None and not df.empty:
        return df
    return load_database_csv(base)


def ensure_session():
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False


def render_match_cards(records: List[Dict[str, Any]]):
    if not records:
        st.warning("Нет близких совпадений в базе.")
        return
    for i, r in enumerate(records):
        name = str(r.get("name", ""))
        en = str(r.get("name_en", "") or "")
        sc = float(r.get("score", 0))
        extra = f" · {en}" if en else ""
        cls = "card card-best" if i == 0 else "card"
        label = "Лучшее совпадение" if i == 0 else "Другой вариант"
        st.markdown(
            f'<div class="{cls}"><strong>{label}</strong><br/>{name}{extra}<br/>'
            f'<span class="muted">Уверенность {sc:.0f}%</span></div>',
            unsafe_allow_html=True,
        )


def render_meal_result(payload: Dict[str, Any]):
    verified = payload["verified"]
    user_dish = payload["user_dish"]
    user_portion = payload["user_portion"]
    gemini_name = payload["gemini_name"]
    gemini_portion = payload["gemini_portion"]
    hybrid_records: List[Dict[str, Any]] = payload["hybrid_records"]
    dishes = payload["dishes"]
    primary = payload["primary"]
    algo_json = payload.get("algo_json") or {}

    st.divider()
    st.subheader("Итог")
    badge = (
        '<span class="badge-ok">Подтверждено</span>'
        if verified
        else '<span class="badge-bad">Не подтверждено</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)
    st.caption(
        "Подтверждение: ваше название близко к ответу ИИ, а ответ ИИ согласуется с выбранной строкой базы "
        f"(пороги схожести ≈ {USER_GEMINI_MIN:.0f}% / {GEMINI_DB_MIN:.0f}%)."
    )

    st.markdown("**Вы указали**")
    st.markdown(f'<div class="card">{user_dish} · {user_portion:.0f} г</div>', unsafe_allow_html=True)
    st.markdown("**ИИ**")
    st.markdown(
        f'<div class="card">{gemini_name} · {gemini_portion:.0f} г</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Совпадения в базе**")
    render_match_cards(hybrid_records)

    if hybrid_records:
        nut = nutrition_for_row(pd.Series(hybrid_records[0]), user_portion)
        st.markdown(
            f'<div class="card muted">Оценка КБЖУ на вашу порцию: '
            f'{nut["kilocalories"]:.0f} ккал · Б {nut["protein_g"]:.1f} г · '
            f'Ж {nut["fat_g"]:.1f} г · У {nut["carb_g"]:.1f} г</div>',
            unsafe_allow_html=True,
        )

    if len(dishes) > 1:
        with st.expander("Другие объекты на фото"):
            for d in dishes:
                if d is primary:
                    continue
                st.write(f"- {d['dish_name']} (~{d['portion_grams']:.0f} g)")

    if algo_json:
        with st.expander("Детали подбора (алгоритмы)"):
            st.json(algo_json)


def main():
    ensure_session()

    st.title("Проверка блюда")
    st.caption("Фото, что вы реально съели — сравнение с ИИ и базой Nutristeppe.")

    if not st.session_state.user_name.strip():
        name = st.text_input("Ваше имя", placeholder="Как подписывать записи", key="name_input")
        if st.button("Далее", type="primary"):
            if name.strip():
                st.session_state.user_name = name.strip()
                st.rerun()
        return

    st.markdown(
        f'<p class="muted">Вы: <strong>{st.session_state.user_name}</strong></p>',
        unsafe_allow_html=True,
    )
    if st.button("Сменить имя", type="secondary", key="chg"):
        st.session_state.user_name = ""
        st.session_state.pop("meal_result", None)
        st.session_state.show_camera = False
        _clear_meal_photo()
        st.rerun()

    st.subheader("Фото")
    st.caption("Камера не включается сама — только после кнопки ниже (меньше расход батареи и приватность).")

    up = st.file_uploader(
        "Загрузить файл",
        type=["jpg", "jpeg", "png"],
        key="meal_file_uploader",
        help="Фото из галереи или файлов",
    )

    if st.button("Сделать фото камерой", type="secondary", key="btn_open_cam"):
        st.session_state.show_camera = True
        st.rerun()

    if st.session_state.show_camera:
        st.info("Разрешите доступ к камере в браузере. Превью только здесь — после снимка окно камеры закроется.")
        cam = st.camera_input(
            "Камера",
            key="widget_camera_input",
            help="Нажмите «Take Photo» / «Снять», когда готовы.",
        )
        if st.button("Отмена — закрыть камеру", type="secondary", key="btn_close_cam"):
            st.session_state.show_camera = False
            st.session_state.pop("widget_camera_input", None)
            st.rerun()
        if cam is not None:
            _meal_photo_bytes_set(Image.open(cam).convert("RGB"))
            st.session_state.show_camera = False
            st.session_state.pop("widget_camera_input", None)
            st.rerun()

    has_stored = bool(st.session_state.get("meal_photo_bytes"))
    has_file = up is not None
    if has_stored or has_file:
        if st.button("Удалить фото", type="secondary", key="btn_clear_photo"):
            _clear_meal_photo()
            st.session_state.show_camera = False
            st.session_state.pop("widget_camera_input", None)
            st.rerun()

    img: Optional[Image.Image] = None
    if up is not None:
        img = Image.open(up).convert("RGB")
        st.session_state.pop("meal_photo_bytes", None)
    else:
        img = _meal_photo_image_get()

    if img is not None:
        image_wide(img)

    st.subheader("Что вы съели")
    user_dish = st.text_input("Название блюда", placeholder="как в реальности", key="udish")
    user_portion = st.number_input("Порция, г", min_value=0.0, max_value=5000.0, value=200.0, step=10.0)

    analyze = st.button("Анализировать", type="primary", key="go")

    if analyze:
        if img is None:
            st.error("Сначала добавьте фото (камера или файл).")
        elif not user_dish.strip():
            st.error("Введите название блюда.")
        else:
            api_key = get_gemini_api_key()
            if not api_key:
                st.error("Анализ недоступен: на сервере не задан ключ API.")
            else:
                db = load_database()
                if db.empty:
                    st.error("База блюд пуста или недоступна.")
                else:
                    jpeg = compress_image_bytes(img)
                    with st.spinner("Анализ фото…"):
                        agent = GeminiMealAgent(api_key)
                        result = agent.analyze_meal_image_jpeg(jpeg)

                    if result.get("error"):
                        st.error(f"Ошибка модели: {result['error']}")
                    else:
                        dishes = result.get("dishes") or []
                        if not dishes or normalize_text(dishes[0]["dish_name"]) == normalize_text("неизвестно"):
                            st.error("Не удалось распознать блюдо. Попробуйте другой ракурс или свет.")
                        else:
                            primary = max(dishes, key=lambda d: float(d.get("confidence", 0)))
                            gemini_name = primary["dish_name"]
                            gemini_portion = float(primary["portion_grams"] or 0)

                            engine = DishSearchEngine(db)
                            hybrid_df, algo_json = engine.algorithm_bundle(gemini_name)

                            matched_name = ""
                            matched_en = ""
                            if not hybrid_df.empty:
                                br = hybrid_df.iloc[0]
                                matched_name = str(br.get("name", ""))
                                matched_en = str(br.get("name_en", "") or "")

                            verified, vdetail = is_verified(
                                user_dish, gemini_name, matched_name, matched_en
                            )

                            saved = save_result_row(
                                user_name=st.session_state.user_name,
                                user_dish_name=user_dish.strip(),
                                user_portion=float(user_portion),
                                gemini_dish_name=gemini_name,
                                gemini_portion=gemini_portion,
                                matched_db_dish=matched_name,
                                matched_db_name_en=matched_en,
                                algorithm_results=algo_json,
                                verification_status=verified,
                                verification_detail=vdetail,
                                image_jpeg=jpeg,
                            )
                            if saved:
                                st.caption("Сохранено в истории.")
                            elif os.environ.get("DATABASE_URL"):
                                st.caption("Не удалось сохранить в историю (ошибка БД).")

                            hybrid_records = (
                                hybrid_df.to_dict("records") if not hybrid_df.empty else []
                            )
                            st.session_state.meal_result = {
                                "verified": verified,
                                "user_dish": user_dish.strip(),
                                "user_portion": float(user_portion),
                                "gemini_name": gemini_name,
                                "gemini_portion": gemini_portion,
                                "hybrid_records": hybrid_records,
                                "dishes": dishes,
                                "primary": primary,
                                "algo_json": algo_json,
                            }

    res = st.session_state.get("meal_result")
    if res is not None:
        render_meal_result(res)
        if st.button("Скрыть результат", type="secondary", key="dismiss_res"):
            st.session_state.pop("meal_result", None)
            st.rerun()


if __name__ == "__main__":
    main()
