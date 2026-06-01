"""
Movie Review Sentiment Analysis System
A professional Streamlit application for sentiment analysis using SimpleRNN, LSTM, and GRU models.
"""

import streamlit as st
import numpy as np
import pickle
import os
import time
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --primary: #D4AF37;          /* Classic Gold */
        --primary-dark: #AA8C2C;     /* Muted Gold */
        --secondary: #A0A0A5;       /* Classic Gray */
        --accent: #D4AF37;          /* Gold Accent */
        --bg-dark: #000000;         /* Pitch Black */
        --bg-card: #0F0F10;         /* Matte Black Card */
        --bg-card-hover: #161618;   /* Hover State Card */
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0B5;
        --text-muted: #55555A;
        --border: #222222;          /* Subtle Border */
        --border-gold: rgba(212, 175, 55, 0.25);
        --positive: #10B981;        /* Emerald Green */
        --negative: #EF4444;        /* Crimson Red */
    }

    /* Global styles */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #121215 0%, #050505 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0A0A0B !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    /* Main content padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header styles */
    .hero-header {
        text-align: center;
        padding: 3rem 2rem 2.5rem;
        background: linear-gradient(180deg, #0A0A0C 0%, #050505 100%);
        border-radius: 16px;
        border: 1px solid var(--border);
        margin-bottom: 2rem;
        position: relative;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #D4AF37 50%, #A0A0A5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        margin-top: 0.75rem;
        font-weight: 400;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }

    .hero-badge {
        background: rgba(212, 175, 55, 0.05);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 30px;
        padding: 0.35rem 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Card styles */
    .glass-card {
        background: #0F0F10;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: var(--border-gold);
        box-shadow: 0 8px 30px rgba(212, 175, 55, 0.03);
    }

    .card-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Sentiment result styles */
    .result-card {
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .result-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.02) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .result-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.02) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .result-neutral {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.08) 0%, rgba(212, 175, 55, 0.02) 100%);
        border: 1px solid rgba(212, 175, 55, 0.3);
    }

    .sentiment-label {
        font-size: 2rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }

    .sentiment-positive { color: var(--positive); }
    .sentiment-negative { color: var(--negative); }

    .confidence-value {
        font-size: 3.5rem;
        font-weight: 900;
        font-family: 'Outfit', sans-serif;
        line-height: 1;
        margin-top: 0.5rem;
    }

    .confidence-label {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* Model card styles */
    .model-card {
        background: #0F0F10;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .model-card:hover {
        border-color: var(--border-gold);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    }

    .model-name {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .model-rnn { color: #A0A0A5; }
    .model-lstm { color: #E5C158; }
    .model-gru { color: #D4AF37; }

    /* Progress bar styles */
    .prob-bar-container {
        margin: 0.5rem 0;
    }

    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
        color: var(--text-secondary);
    }

    .prob-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.05);
        overflow: hidden;
    }

    .prob-bar-fill-pos {
        height: 100%;
        background: linear-gradient(90deg, #10B981, #34D399);
        border-radius: 4px;
        transition: width 0.8s ease;
    }

    .prob-bar-fill-neg {
        height: 100%;
        background: linear-gradient(90deg, #EF4444, #F87171);
        border-radius: 4px;
        transition: width 0.8s ease;
    }

    /* Textarea styling */
    .stTextArea textarea {
        background: #0A0A0C !important;
        border: 1px solid #222222 !important;
        border-radius: 8px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.15) !important;
    }

    .stTextArea textarea::placeholder {
        color: rgba(160, 160, 165, 0.4) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #AA8C2C 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.15) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #E5C158 0%, #C3A23A 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Radio buttons */
    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* Sidebar label */
    .sidebar-section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--primary);
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }

    /* Sidebar logo area */
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }

    .sidebar-logo-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-logo-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }

    .sidebar-logo-subtitle {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* Metrics */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }

    .metric-box {
        flex: 1;
        background: rgba(212, 175, 55, 0.05);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--primary);
        font-family: 'Outfit', sans-serif;
    }

    .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* Best model badge */
    .best-badge {
        display: inline-block;
        background: linear-gradient(135deg, #D4AF37, #AA8C2C);
        color: #000000;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    /* Comparison table */
    .comparison-header {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(212, 175, 55, 0.08);
        border-radius: 8px 8px 0 0;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--primary);
    }

    .comparison-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border);
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }

    .comparison-row:hover {
        background: rgba(212, 175, 55, 0.03);
    }

    .comparison-row:last-child {
        border-bottom: none;
    }

    /* Status indicator */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.4rem;
    }

    .status-loaded { background: var(--positive); }
    .status-error { background: var(--negative); }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 1.5rem 0;
    }

    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Warning box */
    .warning-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 8px;
        padding: 1rem;
        color: var(--negative);
        font-size: 0.9rem;
    }

    /* Success box */
    .success-box {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
        padding: 1rem;
        color: var(--positive);
        font-size: 0.9rem;
    }

    /* Info box */
    .info-box {
        background: rgba(212, 175, 55, 0.05);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 8px;
        padding: 1rem;
        color: var(--primary);
        font-size: 0.9rem;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #0F0F10 !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Hide default streamlit elements styling */
    .stMarkdown p {
        color: var(--text-secondary);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--primary) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-muted);
        font-size: 0.8rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading Functions ───────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all three models and preprocessing artifacts from the models directory."""
    models = {}
    errors = {}

    models_dir = 'models'

    model_configs = {
        'SimpleRNN': os.path.join(models_dir, 'simplernn_imdb_model.keras'),
        'LSTM': os.path.join(models_dir, 'lstm_imdb_model.keras'),
        'GRU': os.path.join(models_dir, 'gru_imdb_model.keras')
    }

    try:
        import tensorflow as tf
    except ImportError:
        for name in model_configs:
            errors[name] = "TensorFlow not installed"
        return models, errors, None, None

    for name, path in model_configs.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
            except Exception as e:
                errors[name] = str(e)
        else:
            errors[name] = f"File not found: {path}"

    # Load word index
    word_index = None
    word_index_path = os.path.join(models_dir, 'imdb_word_index.pkl')
    if os.path.exists(word_index_path):
        try:
            with open(word_index_path, 'rb') as f:
                word_index = pickle.load(f)
        except Exception as e:
            errors['word_index'] = str(e)
    else:
        errors['word_index'] = f"imdb_word_index.pkl not found at {word_index_path}"

    # Load maxlen
    maxlen = None
    maxlen_path = os.path.join(models_dir, 'imdb_maxlen.pkl')
    if os.path.exists(maxlen_path):
        try:
            with open(maxlen_path, 'rb') as f:
                maxlen = pickle.load(f)
        except Exception as e:
            errors['maxlen'] = str(e)
    else:
        errors['maxlen'] = f"imdb_maxlen.pkl not found at {maxlen_path}"

    return models, errors, word_index, maxlen


def preprocess_text(review_text, word_index, maxlen):
    """Preprocess the review text for model prediction."""
    try:
        try:
            from keras.utils import pad_sequences          # Keras 3.x (TF 2.16+)
        except ImportError:
            try:
                from keras.preprocessing.sequence import pad_sequences
            except ImportError:
                from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Tokenize
        words = review_text.lower().split()
        # Map words to indices (offset by 3 for special tokens)
        sequence = []
        for word in words:
            if word in word_index:
                idx = word_index[word]
                # IMDB word index: values > 2 shift by 3 (reserved 0=pad, 1=start, 2=oov, 3=unused)
                if idx < 10000:  # typical vocab size limit
                    sequence.append(idx + 3)
                else:
                    sequence.append(2)  # OOV token
            else:
                sequence.append(2)  # OOV token

        # Pad sequence
        padded = pad_sequences([sequence], maxlen=maxlen, padding='pre', truncating='pre')
        return padded
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None


def predict_sentiment(model, padded_sequence):
    """Run prediction and return probability."""
    try:
        prob = float(model.predict(padded_sequence, verbose=0)[0][0])
        return prob
    except Exception as e:
        return None


def get_sentiment_info(prob):
    """Get sentiment label, confidence, and styling class."""
    if prob >= 0.5:
        return "Positive", prob * 100, "positive", "😊", "#10B981"
    else:
        return "Negative", (1 - prob) * 100, "negative", "😞", "#EF4444"


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🎬</div>
        <div class="sidebar-logo-title">SentimentAI</div>
        <div class="sidebar-logo-subtitle">Movie Review Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">⚙️ Analysis Mode</div>', unsafe_allow_html=True)

    analysis_mode = st.radio(
        "Select Mode",
        ["Single Model", "Compare All Models"],
        label_visibility="collapsed"
    )

    if analysis_mode == "Single Model":
        st.markdown('<div class="sidebar-section-label">🤖 Select Model</div>', unsafe_allow_html=True)
        selected_model = st.radio(
            "Model",
            ["SimpleRNN", "LSTM", "GRU"],
            label_visibility="collapsed"
        )

        model_descriptions = {
            "SimpleRNN": "Basic recurrent neural network. Simpler architecture, faster but may struggle with long-range dependencies.",
            "LSTM": "Long Short-Term Memory. Excellent at capturing long-range dependencies with memory gates.",
            "GRU": "Gated Recurrent Unit. Efficient architecture balancing performance and speed. 🏆 Best performer."
        }

        st.markdown(f"""
        <div style="
            background: rgba(212, 175, 55, 0.04);
            border: 1px solid rgba(212, 175, 55, 0.15);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.5;
        ">
            {model_descriptions[selected_model]}
        </div>
        """, unsafe_allow_html=True)
    else:
        selected_model = "All"

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1.5rem;">📊 Model Performance</div>', unsafe_allow_html=True)

    perf_data = {
        "SimpleRNN": {"acc": "~54%", "loss": "~0.68"},
        "LSTM": {"acc": "~84.5%", "loss": "~0.42"},
        "GRU": {"acc": "~86.6%", "loss": "~0.42"}
    }

    for model_name, perf in perf_data.items():
        color = {"SimpleRNN": "#A0A0A5", "LSTM": "#E5C158", "GRU": "#D4AF37"}[model_name]
        best = " 🏆" if model_name == "GRU" else ""
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0.6rem;
            margin: 0.3rem 0;
            background: rgba(255,255,255,0.02);
            border-radius: 8px;
            font-size: 0.8rem;
        ">
            <span style="color: {color}; font-weight: 600;">{model_name}{best}</span>
            <span style="color: #B0B0B5;">Acc: {perf['acc']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1.5rem;">💡 Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.78rem; color: #718096; line-height: 1.6;">
        • Write detailed reviews for better accuracy<br>
        • GRU model performs best on IMDB data<br>
        • Compare all models to see differences<br>
        • Confidence > 80% indicates strong sentiment
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🎬 Movie Review Sentiment Analysis</h1>
    <p class="hero-subtitle">Deep Learning Based Sentiment Classification</p>
    <div class="hero-badges">
        <span class="hero-badge">🧠 SimpleRNN</span>
        <span class="hero-badge">🔄 LSTM</span>
        <span class="hero-badge">⚡ GRU</span>
        <span class="hero-badge">🎯 IMDB Dataset</span>
        <span class="hero-badge">🤖 TensorFlow</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading AI models..."):
    models, errors, word_index, maxlen = load_models()

# Model status display
all_loaded = len(models) == 3 and word_index is not None and maxlen is not None

col_status1, col_status2, col_status3 = st.columns(3)
model_status_cols = {"SimpleRNN": col_status1, "LSTM": col_status2, "GRU": col_status3}
model_colors = {"SimpleRNN": "#A0A0A5", "LSTM": "#E5C158", "GRU": "#D4AF37"}

for model_name, col in model_status_cols.items():
    with col:
        is_loaded = model_name in models
        status_color = "#10B981" if is_loaded else "#EF4444"
        status_text = "Loaded ✓" if is_loaded else "Error ✗"
        st.markdown(f"""
        <div style="
            background: #0F0F10;
            border: 1px solid {model_colors[model_name]}30;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            text-align: center;
        ">
            <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">{model_name}</div>
            <div style="font-size: 0.85rem; color: {status_color}; font-weight: 600; margin-top: 0.25rem;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

if not all_loaded:
    missing_items = []
    for key, err in errors.items():
        missing_items.append(f"• **{key}**: {err}")

    st.markdown(f"""
    <div class="warning-box" style="margin-top: 1rem;">
        ⚠️ <strong>Some models or files could not be loaded.</strong> Please ensure the following files exist in the <code>models/</code> directory:<br><br>
        <code>models/simplernn_imdb_model.keras</code> &nbsp; <code>models/lstm_imdb_model.keras</code> &nbsp; <code>models/gru_imdb_model.keras</code><br>
        <code>models/imdb_word_index.pkl</code> &nbsp; <code>models/imdb_maxlen.pkl</code>
    </div>
    """, unsafe_allow_html=True)

# Handle sample review loading safely before the text area is instantiated
if "temp_review" in st.session_state:
    st.session_state.review_input = st.session_state.temp_review
    del st.session_state.temp_review

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── Input Area ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-title">
    ✍️ Enter Your Movie Review
</div>
""", unsafe_allow_html=True)

review_text = st.text_area(
    "Movie Review Input",
    placeholder="Enter your movie review here... (e.g., 'This film was absolutely stunning! The performances were incredible and the storyline kept me on the edge of my seat throughout...')",
    height=150,
    label_visibility="collapsed",
    key="review_input"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_btn = st.button("🔍 Analyze Review", key="analyze_btn", use_container_width=True)

# ─── Sample Reviews ───────────────────────────────────────────────────────────

with st.expander("💡 Try Sample Reviews", expanded=False):
    sample_positive = "This movie was absolutely phenomenal! The director created a masterpiece with stunning visuals, incredible performances, and a deeply moving storyline. I was completely captivated from the first frame to the last. The cinematography was breathtaking and the musical score perfectly complemented every scene. Highly recommended to everyone!"
    sample_negative = "What a terrible waste of time and money! The plot made absolutely no sense, the acting was wooden and unconvincing, and the special effects looked cheap and amateurish. The dialogue was cringeworthy and the pacing was painfully slow. I nearly fell asleep multiple times. Do yourself a favor and avoid this disaster."

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("😊 Load Positive Sample", use_container_width=True):
            st.session_state.temp_review = sample_positive
            st.rerun()
    with col_s2:
        if st.button("😞 Load Negative Sample", use_container_width=True):
            st.session_state.temp_review = sample_negative
            st.rerun()

# ─── Analysis Results ─────────────────────────────────────────────────────────

if analyze_btn and review_text.strip():
    if not all_loaded:
        st.error("❌ Cannot analyze: Models or preprocessing files not loaded. Please check the file paths.")
    else:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Preprocess text
        with st.spinner("🔄 Processing your review..."):
            padded = preprocess_text(review_text, word_index, maxlen)
            time.sleep(0.3)  # Brief pause for UX

        if padded is not None:
            if analysis_mode == "Single Model":
                # ─── Single Model Analysis ─────────────────────────────────
                model = models[selected_model]
                prob = predict_sentiment(model, padded)

                if prob is not None:
                    sentiment, confidence, sentiment_class, emoji, color = get_sentiment_info(prob)
                    pos_prob = prob * 100
                    neg_prob = (1 - prob) * 100

                    st.markdown(f"""
                    <div class="section-title">
                        📊 Analysis Results — {selected_model}
                    </div>
                    """, unsafe_allow_html=True)

                    col_main, col_chart = st.columns([1, 1])

                    with col_main:
                        st.markdown(f"""
                        <div class="result-card result-{sentiment_class}">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{emoji}</div>
                            <div class="sentiment-label sentiment-{sentiment_class}">
                                {sentiment}
                            </div>
                            <div class="confidence-value" style="color: {color};">
                                {confidence:.1f}%
                            </div>
                            <div class="confidence-label">Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_chart:
                        st.markdown("""
                        <div class="glass-card" style="margin-top: 0; padding-bottom: 0.5rem;">
                            <div class="card-header">📈 Sentiment Gauge</div>
                        """, unsafe_allow_html=True)

                        # Render a gorgeous Plotly gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = pos_prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            number = {'suffix': "%", 'font': {'size': 32, 'color': '#FFFFFF'}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888888"},
                                'bar': {'color': "#D4AF37"},  # Gold indicator bar
                                'bgcolor': "#050505",
                                'borderwidth': 1,
                                'bordercolor': "#222222",
                                'steps': [
                                    {'range': [0, 50], 'color': '#2C0E0E'},  # Deep red for negative
                                    {'range': [50, 100], 'color': '#0B2C1A'}  # Deep green for positive
                                ]
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#FFFFFF", 'family': "Inter"},
                            height=200,
                            margin=dict(l=25, r=25, t=30, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Review stats
                    word_count = len(review_text.split())
                    st.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-value">{word_count}</div>
                            <div class="metric-label">Words</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{len(review_text)}</div>
                            <div class="metric-label">Characters</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" style="color: {color};">{confidence:.0f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" style="font-size: 1rem;">{selected_model}</div>
                            <div class="metric-label">Model</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                # ─── Compare All Models ────────────────────────────────────
                st.markdown("""
                <div class="section-title">
                    🔬 Model Comparison Dashboard
                </div>
                """, unsafe_allow_html=True)

                results = {}
                model_order = ["SimpleRNN", "LSTM", "GRU"]
                model_emojis = {"SimpleRNN": "🧠", "LSTM": "🔄", "GRU": "⚡"}

                with st.spinner("🔄 Running all models..."):
                    for model_name in model_order:
                        if model_name in models:
                            prob = predict_sentiment(models[model_name], padded)
                            if prob is not None:
                                sentiment, confidence, sentiment_class, emoji, color = get_sentiment_info(prob)
                                results[model_name] = {
                                    "prob": prob,
                                    "sentiment": sentiment,
                                    "confidence": confidence,
                                    "sentiment_class": sentiment_class,
                                    "emoji": emoji,
                                    "color": color,
                                    "pos_prob": prob * 100,
                                    "neg_prob": (1 - prob) * 100
                                }

                if results:
                    # Find best model (highest confidence)
                    best_model = max(results, key=lambda k: results[k]["confidence"])

                    # Display individual model cards
                    cols = st.columns(len(results))
                    for idx, model_name in enumerate(model_order):
                        if model_name in results:
                            r = results[model_name]
                            model_color = model_colors[model_name]
                            is_best = model_name == best_model

                            with cols[idx]:
                                best_indicator = '<span class="best-badge">Best</span>' if is_best else ""
                                st.markdown(f"""
                                <div class="result-card result-{r['sentiment_class']}" style="
                                    {'border: 2px solid ' + model_color + ';' if is_best else ''}
                                    {'box-shadow: 0 0 20px ' + model_color + '30;' if is_best else ''}
                                ">
                                    <div style="font-size: 0.8rem; font-weight: 700; color: {model_color}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                                        {model_emojis[model_name]} {model_name}{best_indicator}
                                    </div>
                                    <div style="font-size: 1.8rem; margin: 0.3rem 0;">{r['emoji']}</div>
                                    <div class="sentiment-label sentiment-{r['sentiment_class']}" style="font-size: 1.4rem;">
                                        {r['sentiment']}
                                    </div>
                                    <div class="confidence-value" style="color: {r['color']}; font-size: 2.5rem;">
                                        {r['confidence']:.1f}%
                                    </div>
                                    <div class="confidence-label">Confidence</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Detailed comparison chart
                    st.markdown("""
                    <div class="glass-card" style="margin-top: 1.5rem; padding-bottom: 0.5rem;">
                        <div class="card-header">📊 Detailed Probability Comparison</div>
                    """, unsafe_allow_html=True)

                    models_list = [model_name for model_name in model_order if model_name in results]
                    pos_probs = [results[m]["pos_prob"] for m in models_list]
                    neg_probs = [results[m]["neg_prob"] for m in models_list]

                    fig = go.Figure(data=[
                        go.Bar(
                            name='😊 Positive',
                            y=models_list,
                            x=pos_probs,
                            orientation='h',
                            marker=dict(color='#10B981'),
                            text=[f"{p:.1f}%" for p in pos_probs],
                            textposition='inside',
                            textfont=dict(color='#FFFFFF')
                        ),
                        go.Bar(
                            name='😞 Negative',
                            y=models_list,
                            x=neg_probs,
                            orientation='h',
                            marker=dict(color='#EF4444'),
                            text=[f"{n:.1f}%" for n in neg_probs],
                            textposition='inside',
                            textfont=dict(color='#FFFFFF')
                        )
                    ])

                    fig.update_layout(
                        barmode='group',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#FFFFFF", 'family': "Inter"},
                        xaxis=dict(
                            gridcolor='#222222',
                            tickfont=dict(color='#A0A0A5'),
                            range=[0, 100],
                            ticksuffix='%',
                            title="Probability"
                        ),
                        yaxis=dict(
                            gridcolor='rgba(0,0,0,0)',
                            tickfont=dict(color='#FFFFFF', size=13),
                            autorange="reversed"
                        ),
                        legend=dict(
                            font=dict(color='#FFFFFF'),
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=250,
                        margin=dict(l=20, r=20, t=10, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Summary table
                    st.markdown("""
                    <div class="glass-card">
                        <div class="card-header">📋 Summary Comparison Table</div>
                        <div class="comparison-header">
                            <div>Model</div>
                            <div>Prediction</div>
                            <div>Confidence</div>
                            <div>Pos. Prob.</div>
                        </div>
                    """, unsafe_allow_html=True)

                    for model_name in model_order:
                        if model_name in results:
                            r = results[model_name]
                            model_color = model_colors[model_name]
                            is_best = model_name == best_model
                            best_tag = " 🏆" if is_best else ""

                            st.markdown(f"""
                            <div class="comparison-row">
                                <div style="color: {model_color}; font-weight: 600;">{model_emojis[model_name]} {model_name}{best_tag}</div>
                                <div style="color: {r['color']};">{r['emoji']} {r['sentiment']}</div>
                                <div style="color: {r['color']}; font-weight: 700;">{r['confidence']:.1f}%</div>
                                <div style="color: #10B981;">{r['pos_prob']:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Consensus
                    sentiments = [results[m]["sentiment"] for m in model_order if m in results]
                    positive_count = sentiments.count("Positive")
                    negative_count = sentiments.count("Negative")
                    consensus = "Positive" if positive_count > negative_count else "Negative" if negative_count > positive_count else "Mixed"
                    consensus_color = "#10B981" if consensus == "Positive" else "#EF4444" if consensus == "Negative" else "#D4AF37"
                    consensus_emoji = "😊" if consensus == "Positive" else "😞" if consensus == "Negative" else "🤔"

                    st.markdown(f"""
                    <div style="
                        background: #0F0F10;
                        border: 1px solid {consensus_color}30;
                        border-radius: 12px;
                        padding: 1.5rem;
                        text-align: center;
                        margin-top: 1rem;
                    ">
                        <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.5rem;">Overall Consensus</div>
                        <div style="font-size: 2.5rem; margin: 0.5rem 0;">{consensus_emoji}</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: {consensus_color}; font-family: 'Outfit', sans-serif;">{consensus}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                            {positive_count} model(s) → Positive &nbsp;|&nbsp; {negative_count} model(s) → Negative
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

elif analyze_btn and not review_text.strip():
    st.markdown("""
    <div class="warning-box" style="margin-top: 1rem; text-align: center;">
        ⚠️ Please enter a movie review before clicking Analyze.
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">🎬</div>
    <div style="color: #A0AEC0; font-weight: 600; margin-bottom: 0.25rem;">Movie Review Sentiment Analysis System</div>
    <div>Built with ❤️ using TensorFlow, Keras & Streamlit | IMDB Dataset | SimpleRNN · LSTM · GRU</div>
</div>
""", unsafe_allow_html=True)
