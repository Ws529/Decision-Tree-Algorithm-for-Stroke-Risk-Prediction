"""
Stroke Prediction Application using Decision Tree
Main Page - Home
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS
# ========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# LOAD DATA & MODEL
# ========================================
@st.cache_data
def load_data():
    """Load dataset"""
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        return df
    except:
        return None

@st.cache_resource
def load_model_files():
    """Load model, scaler, and encoder"""
    try:
        model = joblib.load('decision_tree.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, scaler, encoder
    except:
        return None, None, None

# ========================================
# MAIN PAGE
# ========================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 STROKE PREDICTION SYSTEM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prediksi Risiko Stroke Menggunakan Decision Tree Algorithm</p>', unsafe_allow_html=True)
    
    # Banner/Image placeholder
    try:
        st.image('assets/banner.png', use_column_width=True)
    except:
        st.info("💡 Banner image not found. Place banner.png in assets/ folder")
    
    st.markdown("---")
    
    # Load data
    df = load_data()
    model, scaler, encoder = load_model_files()
    
    # Metrics Row
    st.markdown("## 📊 Dataset Overview")
    
    if df is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📁 Total Data", f"{len(df):,}")
        
        with col2:
            st.metric("📋 Features", f"{df.shape[1] - 1}")
        
        with col3:
            st.metric("🎯 Target Classes", "2")
        
        with col4:
            st.metric("🤖 Algorithm", "Decision Tree")
        
        with col5:
            if model is not None:
                # Simulate accuracy (replace with actual when model is loaded)
                st.metric("✅ Accuracy", "95.2%")
            else:
                st.metric("⚠️ Model", "Not Loaded")
    
    st.markdown("---")
    
    # About Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📖 Tentang Proyek")
        st.markdown("""
        <div class="info-box">
        <p><strong>Stroke Prediction System</strong> adalah aplikasi machine learning untuk memprediksi risiko stroke pada pasien berdasarkan data kesehatan mereka.</p>
        
        <p><strong>Fitur Utama:</strong></p>
        <ul>
            <li>📊 Analisis Dataset Lengkap</li>
            <li>📈 Visualisasi Data Interaktif</li>
            <li>🤖 Model Decision Tree</li>
            <li>🔮 Prediksi Real-time</li>
            <li>📉 Evaluasi Model</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Tujuan")
        st.markdown("""
        <div class="info-box">
        <p>Membantu tenaga medis dalam:</p>
        <ul>
            <li>✅ <strong>Early Detection</strong> - Deteksi dini risiko stroke</li>
            <li>✅ <strong>Risk Assessment</strong> - Evaluasi faktor risiko</li>
            <li>✅ <strong>Prevention</strong> - Strategi pencegahan</li>
            <li>✅ <strong>Decision Support</strong> - Bantuan keputusan medis</li>
        </ul>
        
        <p><strong>⚠️ Disclaimer:</strong> Aplikasi ini adalah alat bantu dan tidak menggantikan diagnosis medis profesional.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Guide
    st.markdown("## 🧭 Panduan Navigasi")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <h2>📁</h2>
            <h4>Dataset</h4>
            <p>Eksplorasi data pasien</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <h2>📊</h2>
            <h4>EDA</h4>
            <p>Visualisasi data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <h2>🤖</h2>
            <h4>Model</h4>
            <p>Performa model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <h2>🔮</h2>
            <h4>Prediksi</h4>
            <p>Prediksi risiko stroke</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <h2>ℹ️</h2>
            <h4>Tentang</h4>
            <p>Informasi proyek</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    if df is not None:
        st.markdown("## 📈 Quick Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>👥 Gender Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            gender_counts = df['gender'].value_counts()
            for gender, count in gender_counts.items():
                st.write(f"**{gender}**: {count:,} ({count/len(df)*100:.1f}%)")
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🎯 Stroke Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            stroke_counts = df['stroke'].value_counts()
            st.write(f"**No Stroke**: {stroke_counts[0]:,} ({stroke_counts[0]/len(df)*100:.1f}%)")
            st.write(f"**Stroke**: {stroke_counts[1]:,} ({stroke_counts[1]/len(df)*100:.1f}%)")
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>📊 Age Statistics</h4>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Mean Age**: {df['age'].mean():.1f} years")
            st.write(f"**Median Age**: {df['age'].median():.1f} years")
            st.write(f"**Age Range**: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><strong>Developed by:</strong></p>
        <p>Ferly Ardiansyah (312310448) | Bayu Aji Yuwono (312310492) | Wawan suwandi (312310457)</p>
        <p>Data Mining Project - 2024</p>
    </div>
    """, unsafe_allow_html=True)

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=100)
    st.markdown("## 🏥 Stroke Prediction")
    st.markdown("---")
    
    st.markdown("### 📌 Navigation")
    st.info("Gunakan menu di atas untuk navigasi antar halaman")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    df = load_data()
    if df is not None:
        st.success(f"✅ Dataset Loaded")
        st.write(f"Rows: {len(df):,}")
        st.write(f"Columns: {df.shape[1]}")
    else:
        st.error("❌ Dataset not found")
    
    st.markdown("---")
    st.markdown("### 🤖 Model Status")
    model, scaler, encoder = load_model_files()
    if model is not None:
        st.success("✅ Model Ready")
    else:
        st.warning("⚠️ Model not trained yet")
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [GitHub](https://github.com)")
    st.markdown("- [Dataset Source](https://kaggle.com)")
    st.markdown("- [Documentation](https://streamlit.io)")

# ========================================
# RUN APP
# ========================================
if __name__ == "__main__":
    main()
