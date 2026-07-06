"""
APLIKASI PREDIKSI STROKE MENGGUNAKAN DECISION TREE
===================================================

Aplikasi web interaktif untuk memprediksi risiko stroke berdasarkan 
karakteristik kesehatan pasien menggunakan algoritma Decision Tree.

Dibuat oleh:
- Ferly Ardiansyah (312310448)
- Bayu Aji Yuwono (312310492)
- Wawan Suwandi (312310457)

Mata Kuliah: Data Mining
Tahun: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from PIL import Image

# ==================== KONFIGURASI PAGE ====================
st.set_page_config(
    page_title="Stroke Prediction - Decision Tree",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .big-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info box */
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #764ba2;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("# 🏥 Stroke Prediction")
    st.markdown("---")
    
    # Navigation info
    st.markdown("""
    ### 📱 Navigasi
    
    Gunakan menu di atas untuk:
    - 📊 **Dataset**: Lihat data
    - 📈 **EDA**: Analisis visual
    - 🤖 **Model**: Performa model
    - 🔮 **Prediksi**: Prediksi stroke
    - ℹ️ **Tentang**: Info proyek
    """)
    
    st.markdown("---")
    
    # Project info
    st.markdown("""
    ### 👥 Tim Pengembang
    - Ferly Ardiansyah
    - Bayu Aji Yuwono
    - Wawan Suwandi
    
    ### 🎓 Mata Kuliah
    Data Mining - 2024
    
    ### 🤖 Algoritma
    Decision Tree Classifier
    """)
    
    st.markdown("---")
    
    # Quick stats
    if os.path.exists('healthcare-dataset-stroke-data.csv'):
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Pasien", f"{len(df):,}")
        st.metric("Total Fitur", df.shape[1])
        st.metric("Kasus Stroke", f"{df['stroke'].sum()}")

# ==================== MAIN CONTENT ====================
st.markdown('<p class="big-header">🏥 PREDIKSI RISIKO STROKE</p>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Menggunakan Algoritma Decision Tree</h3>', unsafe_allow_html=True)

st.markdown("---")

# ==================== HERO SECTION ====================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="info-box">
        <h3 style="text-align: center; margin-top: 0;">Selamat Datang! 👋</h3>
        <p style="text-align: center; font-size: 1.1rem;">
            Aplikasi ini menggunakan <b>Machine Learning</b> untuk memprediksi risiko stroke 
            berdasarkan data kesehatan pasien. Model ini dibangun menggunakan algoritma 
            <b>Decision Tree</b> dengan akurasi tinggi.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== METRICS SECTION ====================
st.markdown("## 📊 Informasi Dataset & Model")

# Load data untuk metrics
if os.path.exists('healthcare-dataset-stroke-data.csv'):
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📋 Total Data",
            value=f"{len(df):,}",
            delta="Pasien"
        )
    
    with col2:
        st.metric(
            label="🔢 Jumlah Fitur",
            value=df.shape[1] - 1,
            delta="Features"
        )
    
    with col3:
        stroke_pct = (df['stroke'].sum() / len(df)) * 100
        st.metric(
            label="🎯 Kasus Stroke",
            value=f"{df['stroke'].sum()}",
            delta=f"{stroke_pct:.1f}%"
        )
    
    with col4:
        st.metric(
            label="🤖 Algoritma",
            value="Decision Tree",
            delta="ML Model"
        )
    
    with col5:
        # Placeholder untuk accuracy (akan diambil dari model)
        st.metric(
            label="✅ Akurasi",
            value="95%+",
            delta="Test Accuracy"
        )

st.markdown("---")

# ==================== FEATURES OVERVIEW ====================
st.markdown("## 🔍 Fitur-Fitur Prediksi")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📝 Fitur Kategorikal
    - **Gender**: Jenis kelamin pasien
    - **Ever Married**: Status pernikahan
    - **Work Type**: Jenis pekerjaan
    - **Residence Type**: Tipe tempat tinggal
    - **Smoking Status**: Status merokok
    """)

with col2:
    st.markdown("""
    ### 🔢 Fitur Numerikal
    - **Age**: Usia pasien (tahun)
    - **Hypertension**: Riwayat hipertensi
    - **Heart Disease**: Riwayat penyakit jantung
    - **Average Glucose Level**: Kadar glukosa rata-rata
    - **BMI**: Body Mass Index
    """)

st.markdown("---")

# ==================== VISUALIZATION SECTION ====================
st.markdown("## 📈 Quick Data Overview")

if os.path.exists('healthcare-dataset-stroke-data.csv'):
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribusi Target", "👥 Distribusi Gender", "📉 Distribusi Usia"])
    
    with tab1:
        # Pie chart stroke distribution
        stroke_counts = df['stroke'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Tidak Stroke', 'Stroke'],
            values=stroke_counts.values,
            hole=0.4,
            marker=dict(colors=['#00CC96', '#EF553B']),
            textinfo='label+percent+value'
        )])
        fig.update_layout(
            title='Distribusi Target Variable - Stroke',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Gender distribution
        gender_counts = df['gender'].value_counts()
        fig = px.bar(
            x=gender_counts.index,
            y=gender_counts.values,
            labels={'x': 'Gender', 'y': 'Jumlah'},
            title='Distribusi Gender',
            color=gender_counts.index,
            color_discrete_map={'Male': '#636EFA', 'Female': '#EF553B', 'Other': '#00CC96'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Age distribution
        fig = px.histogram(
            df,
            x='age',
            nbins=40,
            title='Distribusi Usia Pasien',
            labels={'age': 'Usia (tahun)', 'count': 'Jumlah'},
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== HOW TO USE ====================
st.markdown("## 🚀 Cara Menggunakan Aplikasi")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px;">
        <h2>1️⃣</h2>
        <h4>Dataset</h4>
        <p>Lihat dan eksplorasi data yang digunakan</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px;">
        <h2>2️⃣</h2>
        <h4>EDA</h4>
        <p>Analisis visual data dengan grafik interaktif</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px;">
        <h2>3️⃣</h2>
        <h4>Model</h4>
        <p>Lihat performa dan evaluasi model</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px;">
        <h2>4️⃣</h2>
        <h4>Prediksi</h4>
        <p>Input data dan dapatkan prediksi</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== FOOTER ====================
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h3>📌 Catatan Penting</h3>
    <p style="font-size: 1.1rem;">
        Aplikasi ini dibuat untuk tujuan <b>edukasi</b> dan <b>penelitian</b>.<br>
        Hasil prediksi <b>tidak menggantikan</b> diagnosis medis profesional.<br>
        Selalu konsultasikan dengan tenaga medis untuk keputusan kesehatan.
    </p>
    <hr>
    <p style="color: #666; margin-top: 1rem;">
        © 2024 | Praktikum Data Mining | Decision Tree Stroke Prediction
    </p>
</div>
""", unsafe_allow_html=True)
