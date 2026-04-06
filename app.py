import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
import re
from typing import Tuple, Dict, Optional, List
import plotly.express as px
import plotly.graph_objects as go
import os
import pathlib

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Meal Recognition Agent",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme
st.markdown("""
<style>
    /* Main dark theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff !important;
    }
    
    /* All text white by default */
    .stApp, .stApp * {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .main-header h1, .main-header p {
        color: white !important;
    }
    
    /* Nutrition cards */
    .nutrition-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea !important;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #cccccc !important;
    }
    
    /* Match highlight */
    .match-highlight {
        background: rgba(46, 204, 113, 0.2);
        border-left: 4px solid #2ecc71;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background: rgba(0,0,0,0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 0.5rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255,255,255,0.05);
        border-radius: 0.5rem;
    }
    
    /* Metric boxes */
    .stMetric {
        background: rgba(255,255,255,0.05);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background: rgba(0,0,0,0.5);
        border-radius: 0.5rem;
    }
    
    /* Select boxes */
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.05);
        border-radius: 0.5rem;
    }
    
    /* Slider */
    .stSlider {
        background: rgba(255,255,255,0.05);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(255,255,255,0.05);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white !important;
    }
    
    /* Plotly charts text */
    .plotly .main-svg {
        background: rgba(0,0,0,0.3) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Recipe container */
    .recipe-container {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE SETUP - LOAD FROM CSV
# ============================================================================

@st.cache_data
def load_database() -> pd.DataFrame:
    """
    Load the dish database from CSV file.
    """
    try:
        base_path = pathlib.Path(__file__).parent
        csv_path = base_path / '2April.csv'
        
        if not csv_path.exists():
            st.error(f"❌ CSV file not found at: {csv_path}")
            return pd.DataFrame()

        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns properly
        numeric_cols = ['kilocalories', 'protein', 'fat', 'carbohydrate', 
                       'fiber', 'sugar_mg', 'salt_total_mg', 'saturated_fat_mg',
                       'serving_size_g', 'kilocalories_portion', 'calculated_kcal']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create a searchable text column with dish name (Russian)
        df['search_text'] = df['name'].fillna('').str.lower()
        
        st.success(f"✅ Database loaded successfully! Total dishes: {len(df)}")
        return df
        
    except FileNotFoundError:
        st.error("❌ CSV file '30March.csv' not found! Please make sure the file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading database: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# GEMINI AI AGENT SETUP - FIXED FOR RUSSIAN OUTPUT
# ============================================================================

class GeminiMealAgent:
    def __init__(self, api_key: str):
        """Initialize the Gemini AI agent."""
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def analyze_meal_image(self, image: Image.Image) -> Dict:
        """
        Analyze meal image to detect one or multiple dishes and estimate portions.
        Returns: {'dishes': [{'dish_name': str, 'portion_grams': float, 'confidence': float}, ...]}
        """
        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create the prompt for Gemini - FORCED RUSSIAN OUTPUT
            prompt = """
            Ты профессиональный ИИ для распознавания еды. Проанализируй изображение с едой и предоставь:
            
            1. Список до 6 блюд, которые видны на изображении.
            2. Предполагаемый размер порции в граммах для каждого блюда.
            3. Уровень уверенности для каждого блюда (0-1).
            
            ВАЖНО: ВСЕ названия блюд должны быть ТОЛЬКО на РУССКОМ языке!
            Например: "пицца" вместо "pizza", "бургер" вместо "burger", "салат" вместо "salad"
            
            Верни ТОЛЬКО валидный JSON в одном из этих форматов:
            [
                {"dish_name": "название блюда на русском", "portion_grams": 350, "confidence": 0.95},
                ...
            ]
            
            Если видно только одно блюдо, верни массив с одним элементом.
            Если не можешь определить блюдо, используй {"dish_name": "неизвестно", "portion_grams": 0, "confidence": 0}.
            
            ОБЯЗАТЕЛЬНО используй русские названия для ВСЕХ блюд, независимо от кухни!
            """
            
            # Send to Gemini
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_byte_arr}
            ])
            
            # Parse the response
            response_text = response.text
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                obj_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if obj_match:
                    parsed = [json.loads(obj_match.group())]
                else:
                    parsed = []

            if not isinstance(parsed, list):
                parsed = [parsed]

            dishes = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                dish_name = item.get('dish_name', 'unknown')
                # Convert any English names to Russian approximation if needed
                dish_name = self._ensure_russian_name(dish_name)
                dishes.append({
                    'dish_name': dish_name,
                    'portion_grams': float(item.get('portion_grams', 0) or 0),
                    'confidence': float(item.get('confidence', 0) or 0)
                })

            if not dishes:
                dishes = [{'dish_name': 'неизвестно', 'portion_grams': 0, 'confidence': 0}]

            return {'dishes': dishes}
            
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return {'dishes': [{'dish_name': 'неизвестно', 'portion_grams': 0, 'confidence': 0}]}
    
    def _ensure_russian_name(self, name: str) -> str:
        """Convert common English dish names to Russian."""
        english_to_russian = {
            'pizza': 'пицца',
            'burger': 'бургер',
            'salad': 'салат',
            'soup': 'суп',
            'pasta': 'паста',
            'rice': 'рис',
            'chicken': 'курица',
            'beef': 'говядина',
            'fish': 'рыба',
            'ceasar': 'цезарь',
            'caesar': 'цезарь',
            'salami': 'салями',
            'cheese': 'сыр',
            'sandwich': 'сэндвич',
            'fries': 'картофель фри',
            'potato': 'картофель',
            'steak': 'стейк',
            'sushi': 'суши',
            'roll': 'ролл',
            'noodle': 'лапша',
            'curry': 'карри',
            'taco': 'тако',
            'wrap': 'ролл',
            'bowl': 'миска',
            'plate': 'тарелка',
            'with': 'с',
            'and': 'и'
        }
        
        name_lower = name.lower()
        for eng, rus in english_to_russian.items():
            if eng in name_lower:
                name = name_lower.replace(eng, rus)
        
        # Capitalize first letter
        if name and len(name) > 0:
            name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()
        
        return name

# ============================================================================
# SEARCH ALGORITHMS
# ============================================================================

class DishSearchEngine:
    def __init__(self, database: pd.DataFrame):
        self.database = database
        
    def search_by_fuzzy_matching(self, query: str, threshold: int = 60) -> pd.DataFrame:
        """
        Search dishes using fuzzy string matching algorithm on Russian names only.
        """
        if query.lower() == 'неизвестно' or query.lower() == 'unknown' or not query:
            return pd.DataFrame()
        
        results = []
        query_lower = query.lower()
        
        for idx, row in self.database.iterrows():
            # Get the dish name (Russian)
            dish_name = str(row['name']).lower()
            
            # Calculate match scores using various fuzzy methods
            name_score = fuzz.ratio(query_lower, dish_name)
            partial_score = fuzz.partial_ratio(query_lower, dish_name)
            token_score = fuzz.token_sort_ratio(query_lower, dish_name)
            
            # Final score is the best of all methods
            final_score = max(name_score, partial_score, token_score)
            
            if final_score >= threshold:
                results.append({
                    'score': final_score,
                    'index': idx,
                    **row.to_dict()
                })
        
        # Sort by score and return
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('score', ascending=False)
        
        return results_df
    
    def search_by_token_matching(self, query: str) -> pd.DataFrame:
        """
        Search using token-based matching (more precise for multi-word queries).
        """
        if query.lower() == 'неизвестно' or query.lower() == 'unknown' or not query:
            return pd.DataFrame()
        
        query_tokens = set(query.lower().split())
        results = []
        
        for idx, row in self.database.iterrows():
            name_tokens = set(str(row['name']).lower().split())
            
            # Calculate Jaccard similarity
            if len(query_tokens) == 0:
                jaccard_score = 0
            else:
                intersection = len(query_tokens & name_tokens)
                union = len(query_tokens | name_tokens)
                jaccard_score = intersection / union if union > 0 else 0
            
            if jaccard_score > 0.3:  # Threshold
                results.append({
                    'score': jaccard_score * 100,
                    'index': idx,
                    **row.to_dict()
                })
        
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('score', ascending=False)
        
        return results_df
    
    def search_by_levenshtein(self, query: str) -> pd.DataFrame:
        """
        Search using Levenshtein distance for close matches.
        Using fuzz.ratio which is based on Levenshtein distance.
        """
        if query.lower() == 'неизвестно' or query.lower() == 'unknown' or not query:
            return pd.DataFrame()
        
        query_lower = query.lower()
        results = []
        
        for idx, row in self.database.iterrows():
            dish_name = str(row['name']).lower()
            
            # Use fuzz.ratio which is based on Levenshtein distance
            similarity = fuzz.ratio(query_lower, dish_name)
            
            if similarity >= 50:
                results.append({
                    'score': similarity,
                    'index': idx,
                    **row.to_dict()
                })
        
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('score', ascending=False)
        
        return results_df
    
    def hybrid_search(self, query: str) -> pd.DataFrame:
        """
        Hybrid search combining fuzzy, token, and levenshtein matching for best results.
        """
        fuzzy_results = self.search_by_fuzzy_matching(query, threshold=50)
        token_results = self.search_by_token_matching(query)
        levenshtein_results = self.search_by_levenshtein(query)
        
        results_list = []
        if not fuzzy_results.empty:
            results_list.append(fuzzy_results)
        if not token_results.empty:
            results_list.append(token_results)
        if not levenshtein_results.empty:
            results_list.append(levenshtein_results)
        
        if not results_list:
            return pd.DataFrame()
        
        # Combine all results
        combined = pd.concat(results_list, ignore_index=True)
        
        # Deduplicate by bls_code or name
        if 'bls_code' in combined.columns:
            combined = combined.drop_duplicates(subset=['bls_code'], keep='first')
        else:
            combined = combined.drop_duplicates(subset=['name'], keep='first')
        
        # Sort by score
        combined = combined.sort_values('score', ascending=False)
        
        return combined.head(5)

# ============================================================================
# NUTRITION CALCULATOR
# ============================================================================

class NutritionCalculator:
    @staticmethod
    def calculate_nutrition(dish: pd.Series, portion_grams: float) -> Dict:
        """
        Calculate nutritional values based on detected portion.
        """
        # Get base serving size (default to 100g if not specified)
        base_serving = dish.get('serving_size_g', 100)
        if pd.isna(base_serving) or base_serving == 0:
            base_serving = 100
        
        # Calculate ratio
        ratio = portion_grams / base_serving
        
        # Calculate values
        nutrition = {
            'dish_name': dish.get('name', 'Unknown'),
            'portion_grams': portion_grams,
            'kilocalories': dish.get('kilocalories', 0) * ratio,
            'protein': dish.get('protein', 0) * ratio / 1000,  # Convert to grams
            'fat': dish.get('fat', 0) * ratio / 1000,
            'carbohydrate': dish.get('carbohydrate', 0) * ratio / 1000,
            'fiber': dish.get('fiber', 0) * ratio / 1000 if 'fiber' in dish else 0,
            'sugar_mg': dish.get('sugar_mg', 0) * ratio,
            'salt_total_mg': dish.get('salt_total_mg', 0) * ratio,
            'saturated_fat_mg': dish.get('saturated_fat_mg', 0) * ratio,
            'match_score': dish.get('score', 0) if 'score' in dish else 100,
            'recipe': dish.get('steps', 'Нет рецепта'),
            'ingredients': dish.get('ingredients', 'Нет списка ингредиентов'),
            'health_index': dish.get('index_health', 'N/A')
        }
        
        return nutrition

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_nutrition_card(nutrition: Dict):
    """Display nutrition information in a beautiful card layout."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="nutrition-card">
            <div class="metric-value">{nutrition['kilocalories']:.0f}</div>
            <div class="metric-label">🔥 Калории (ккал)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="nutrition-card">
            <div class="metric-value">{nutrition['protein']:.1f}г</div>
            <div class="metric-label">💪 Белки</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="nutrition-card">
            <div class="metric-value">{nutrition['fat']:.1f}г</div>
            <div class="metric-label">🥑 Жиры</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="nutrition-card">
            <div class="metric-value">{nutrition['carbohydrate']:.1f}г</div>
            <div class="metric-label">🍚 Углеводы</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional macros
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Пищевые волокна", f"{nutrition['fiber']:.1f}г" if nutrition['fiber'] > 0 else "Н/Д")
    
    with col6:
        sugar_g = nutrition['sugar_mg'] / 1000
        st.metric("Сахар", f"{sugar_g:.1f}г")
    
    with col7:
        salt_g = nutrition['salt_total_mg'] / 1000
        st.metric("Соль", f"{salt_g:.1f}г")

def display_macros_chart(nutrition: Dict):
    """Display a pie chart of macronutrient distribution."""
    
    macros = {
        'Белки': nutrition['protein'],
        'Жиры': nutrition['fat'],
        'Углеводы': nutrition['carbohydrate']
    }
    
    fig = px.pie(
        values=list(macros.values()),
        names=list(macros.keys()),
        title="Распределение макронутриентов",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Update chart for dark theme
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white')
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_recipe_section(nutrition: Dict):
    """Display recipe and ingredients in a separate section (not inside expander)."""
    
    st.markdown('<div class="recipe-container">', unsafe_allow_html=True)
    st.subheader("📖 Рецепт и ингредиенты")
    
    # Ingredients
    st.markdown("**🥘 Ингредиенты:**")
    ingredients = nutrition['ingredients']
    if isinstance(ingredients, str) and ingredients and ingredients != 'Нет списка ингредиентов':
        if ingredients.startswith('['):
            try:
                import ast
                ingredients_list = ast.literal_eval(ingredients)
                for ing in ingredients_list:
                    st.markdown(f"- {ing}")
            except:
                st.write(ingredients)
        else:
            st.write(ingredients)
    else:
        st.write("Информация об ингредиентах отсутствует")
    
    st.markdown("---")
    
    # Preparation Steps
    st.markdown("**📋 Этапы приготовления:**")
    recipe = nutrition['recipe']
    if recipe and recipe != 'Нет рецепта':
        st.write(recipe)
    else:
        st.write("Информация о приготовлении отсутствует")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ AI Агент распознавания блюд</h1>
        <p>Работает на Google Gemini Flash | Интеллектуальное распознавание еды и анализ питания</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/google-gemini.png", width=80)
        st.title("⚙️ Настройки")
        
        # API Key input
        api_key = st.text_input("🔑 Ключ Google Gemini API", type="password", 
                                help="Введите ваш ключ Google Gemini API")
        
        if api_key:
            st.success("✅ Ключ API настроен")
        else:
            st.warning("⚠️ Пожалуйста, введите ключ Gemini API")
        
        st.divider()
        
        st.subheader("ℹ️ О приложении")
        st.info("""
        **Как это работает:**
        1. 📸 Загрузите фото еды
        2. 🤖 ИИ распознает блюдо и размер порции
        3. 🔍 Ищет в базе данных используя нечеткое сравнение
        4. 📊 Рассчитывает точные значения питательных веществ
        5. 📖 Показывает рецепт и ингредиенты
        """)
        
        st.divider()
        
        st.caption("Сделано с ❤️ используя Gemini AI")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Загрузите фото блюда")
        uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное блюдо", use_column_width=True)
    
    with col2:
        st.subheader("⚙️ Параметры распознавания")
        search_method = st.selectbox(
            "Алгоритм поиска",
            ["Гибридный (рекомендуется)", "Нечеткое совпадение", "Токенное совпадение", "Расстояние Левенштейна"]
        )
        
        portion_adjustment = st.slider(
            "Коэффициент корректировки порции",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Отрегулируйте размер порции вручную"
        )
    
    # Process button
    if uploaded_file is not None and api_key:
        if st.button("🔍 Проанализировать блюдо", type="primary", use_container_width=True):
            with st.spinner("🤖 ИИ агент анализирует ваше блюдо..."):
                
                # Step 1: Initialize AI Agent
                gemini_agent = GeminiMealAgent(api_key)
                
                # Step 2: Analyze image
                detection_result = gemini_agent.analyze_meal_image(image)
                dishes = detection_result.get('dishes', [])
                
                if not dishes or dishes[0]['dish_name'] == 'неизвестно':
                    st.error("❌ Блюда не обнаружены. Пожалуйста, попробуйте другое изображение.")
                    return
                
                st.subheader("🎯 Результаты распознавания")
                summary_rows = []
                for item in dishes:
                    summary_rows.append({
                        'Блюдо': item['dish_name'].title(),
                        'Порция (г)': int(item['portion_grams']),
                        'Доверие': f"{item['confidence']*100:.0f}%"
                    })
                st.table(pd.DataFrame(summary_rows))
                
                # Step 3: Load database
                database = load_database()
                if database.empty:
                    st.error("❌ Не удалось загрузить базу данных. Проверьте файл CSV.")
                    return
                
                # Step 4: Search database for each detected dish
                search_engine = DishSearchEngine(database)
                calculator = NutritionCalculator()
                
                # Use tabs instead of nested expanders
                if len(dishes) > 1:
                    tabs = st.tabs([f"Блюдо {i+1}: {item['dish_name'].title()}" for i, item in enumerate(dishes)])
                else:
                    tabs = [None]  # No tabs for single dish
                
                for idx, item in enumerate(dishes):
                    if len(dishes) > 1:
                        current_tab = tabs[idx]
                        with current_tab:
                            process_single_dish(item, idx, search_method, portion_adjustment, search_engine, calculator)
                    else:
                        # Single dish - no tabs needed
                        process_single_dish(item, idx, search_method, portion_adjustment, search_engine, calculator)
                            
    elif uploaded_file is None:
        st.info("👈 Загрузите фото блюда, чтобы начать анализ")
    elif not api_key:
        st.info("🔑 Введите ключ Google Gemini API в боковой панели")

def process_single_dish(item, idx, search_method, portion_adjustment, search_engine, calculator):
    """Process a single dish and display results."""
    dish_name = item['dish_name']
    adjusted_portion = item['portion_grams'] * portion_adjustment if item['portion_grams'] > 0 else 200
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Определенное блюдо", dish_name.title())
    with col_b:
        st.metric("Порция", f"{int(adjusted_portion)} г")
    with col_c:
        st.metric("Доверие", f"{item['confidence']*100:.0f}%")
    
    # Translate search method name for function call
    method_mapping = {
        "Гибридный (рекомендуется)": "hybrid",
        "Нечеткое совпадение": "fuzzy",
        "Токенное совпадение": "token",
        "Расстояние Левенштейна": "levenshtein"
    }
    
    method = method_mapping.get(search_method, "hybrid")
    
    # Select search method
    if method == "hybrid":
        search_results = search_engine.hybrid_search(dish_name)
    elif method == "fuzzy":
        search_results = search_engine.search_by_fuzzy_matching(dish_name)
    elif method == "token":
        search_results = search_engine.search_by_token_matching(dish_name)
    else:
        search_results = search_engine.search_by_levenshtein(dish_name)
    
    if not search_results.empty:
        best_match = search_results.iloc[0]
        
        st.markdown(f"""
        <div class="match-highlight">
            ✅ <strong>Лучшее совпадение:</strong> {best_match['name']}<br>
            🎯 <strong>Уровень совпадения:</strong> {best_match['score']:.1f}%
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate nutrition
        nutrition = calculator.calculate_nutrition(best_match, adjusted_portion)
        
        st.subheader("📊 Пищевая ценность")
        display_nutrition_card(nutrition)
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            display_macros_chart(nutrition)
        
        with col_chart2:
            st.subheader("💚 Показатели здоровья")
            health_score = nutrition['health_index']
            if health_score != 'N/A' and pd.notna(health_score):
                try:
                    health_val = float(health_score)
                    st.metric("Индекс здоровья", f"{health_val}/5")
                    
                    # Health gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=health_val,
                        title={'text': "Оценка здоровья"},
                        gauge={'axis': {'range': [0, 5]},
                               'bar': {'color': "#2ecc71"},
                               'steps': [
                                   {'range': [0, 2], 'color': "#e74c3c"},
                                   {'range': [2, 3.5], 'color': "#f39c12"},
                                   {'range': [3.5, 5], 'color': "#2ecc71"}
                               ]}
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.write(f"Индекс здоровья: {health_score}")
        
        # Display recipe section (not inside expander)
        display_recipe_section(nutrition)
        
        # Alternative matches (using expander for this is fine)
        if len(search_results) > 1:
            with st.expander("🔄 Альтернативные совпадения"):
                for i, match in search_results.iloc[1:].iterrows():
                    st.markdown(f"""
                    - **{match['name']}** (Совпадение: {match['score']:.0f}%)
                    """)
    else:
        st.warning(f"⚠️ Не найдено совпадений для '{dish_name}'. Попробуйте другое изображение или измените метод поиска.")

if __name__ == "__main__":
    main()