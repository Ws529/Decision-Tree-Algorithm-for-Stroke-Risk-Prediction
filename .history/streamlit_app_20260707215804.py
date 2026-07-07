import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Aplikasi Prediksi Stroke",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc96;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=100)
    st.title("Prediksi Stroke")
    st.markdown("---")
    
    st.markdown("""
    ### Navigasi
    Pilih halaman di atas untuk:
    - **Dataset**: Lihat data
    - **EDA**: Eksplorasi visual
    - **Model**: Performa model
    - **Prediksi**: Prediksi stroke
    - **Tentang**: Info proyek
    """)
    
    st.markdown("---")
    st.markdown("### Tim Pengembang")
    st.markdown("""
    - Ferly Ardiansyah
    - Bayu Aji Yuwono
    - Wawan suwandi
    """)
    
    st.markdown("---")
    st.markdown("### Info")
    st.info("**Mata Kuliah:** Data Mining\n\n**Algoritma:** Decision Tree")

# =====================================================================
# MAIN CONTENT
# =====================================================================

# Header
st.markdown('<p class="main-header">🏥 Sistem Prediksi Stroke</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prediksi Risiko Stroke Menggunakan Decision Tree</p>', unsafe_allow_html=True)
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        return df
    except:
        return None

df = load_data()

# Load Model Info
@st.cache_resource
def load_model_info():
    try:
        model = joblib.load('decision_tree.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, scaler, encoder
    except:
        return None, None, None

model, scaler, encoder = load_model_info()

# =====================================================================
# METRICS OVERVIEW
# =====================================================================
st.markdown("## 📊 Ringkasan Dataset")

if df is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Total Data",
            value=f"{len(df):,}",
            delta="5,110 pasien"
        )
    
    with col2:
        st.metric(
            label="🔢 Jumlah Fitur",
            value="11",
            delta="fitur"
        )
    
    with col3:
        stroke_pct = (df['stroke'].sum() / len(df)) * 100
        st.metric(
            label="⚠️ Kasus Stroke",
            value=f"{df['stroke'].sum()}",
            delta=f"{stroke_pct:.1f}%"
        )
    
    with col4:
        if model is not None:
            st.metric(
                label="🤖 Akurasi Model",
                value="~95%",
                delta="Decision Tree"
            )
        else:
            st.metric(
                label="🤖 Model",
                value="Belum Dimuat",
                delta="Latih model dulu"
            )

st.markdown("---")

# =====================================================================
# QUICK STATS
# =====================================================================
st.markdown("## 📈 Statistik Cepat")

if df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Distribusi Gender")
        gender_counts = df['gender'].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Distribusi Gender",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Distribusi Stroke")
        stroke_counts = df['stroke'].value_counts()
        fig = px.pie(
            values=stroke_counts.values,
            names=['Tidak Stroke', 'Stroke'],
            title="Distribusi Stroke",
            color_discrete_map={0: '#00CC96', 1: '#EF553B'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================================================
# KEY FEATURES
# =====================================================================
st.markdown("## 🎯 Fitur Utama Aplikasi")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>📊 Eksplorasi Dataset</h3>
        <p>Jelajahi dataset lengkap dengan berbagai filter dan statistik deskriptif.</p>
        <ul>
            <li>Preview data</li>
            <li>Statistik lengkap</li>
            <li>Download dataset</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <h3>📈 EDA Interaktif</h3>
        <p>Visualisasi data interaktif dengan Plotly untuk analisis mendalam.</p>
        <ul>
            <li>15+ visualisasi</li>
            <li>Korelasi fitur</li>
            <li>Distribusi data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
        <h3>🤖 Performa Model</h3>
        <p>Evaluasi model Decision Tree dengan berbagai metrik.</p>
        <ul>
            <li>Accuracy, Precision, Recall</li>
            <li>Confusion Matrix</li>
            <li>ROC Curve & AUC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# ABOUT STROKE
# =====================================================================
st.markdown("## 🏥 Tentang Stroke")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Apa itu Stroke?
    
    **Stroke** adalah kondisi medis serius yang terjadi ketika aliran darah ke bagian otak terganggu atau berkurang, 
    sehingga mencegah jaringan otak mendapatkan oksigen dan nutrisi. Sel-sel otak mulai mati dalam hitungan menit.
    
    ### 🔴 Faktor Risiko Stroke:
    
    1. **Usia**: Risiko meningkat seiring bertambahnya usia
    2. **Hipertensi**: Tekanan darah tinggi
    3. **Penyakit Jantung**: Riwayat penyakit jantung
    4. **Diabetes**: Kadar gula darah tinggi
    5. **Obesitas**: BMI tinggi
    6. **Merokok**: Kebiasaan merokok
    7. **Kolesterol Tinggi**: Kadar kolesterol tidak normal
    
    ### 💡 Pencegahan:
    
    - Jaga pola makan sehat
    - Olahraga teratur
    - Kontrol tekanan darah
    - Hindari merokok
    - Kelola stres
    - Cek kesehatan rutin
    """)

with col2:
    st.markdown("### 📊 Statistik Penting")
    
    if df is not None:
        st.info(f"""
        **Total Pasien**: {len(df):,}
        
        **Kasus Stroke**: {df['stroke'].sum()}
        
        **Persentase**: {(df['stroke'].sum()/len(df)*100):.2f}%
        
        **Usia Rata-rata**: {df['age'].mean():.1f} tahun
        
        **BMI Rata-rata**: {df['bmi'].mean():.1f}
        """)
    
    st.success("""
    **🎯 Deteksi Dini**
    
    Sistem ini membantu mendeteksi risiko stroke lebih awal menggunakan machine learning.
    """)

st.markdown("---")

# =====================================================================
# CALL TO ACTION
# =====================================================================
st.markdown("## Mulai Sekarang")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**1. Eksplorasi Data**\n\nLihat dataset dan statistik di halaman Dataset")

with col2:
    st.success("**2. Analisis Visual**\n\nJelajahi visualisasi di halaman EDA")

with col3:
    st.warning("**3. Prediksi Stroke**\n\nCoba prediksi di halaman Prediksi")

# =====================================================================
# KESIMPULAN SISTEM
# =====================================================================
st.markdown("---")
st.subheader("Kesimpulan Sistem Prediksi")

st.markdown("""
**Apa yang Dihasilkan oleh Sistem Prediksi Stroke?**

Sistem prediksi stroke menggunakan Decision Tree ini menghasilkan beberapa output penting:

**1. Prediksi Risiko Stroke**

Sistem akan mengklasifikasikan pasien ke dalam dua kategori:
- **Risiko Rendah (Tidak Stroke)**: Pasien diprediksi tidak memiliki risiko stroke
- **Risiko Tinggi (Stroke)**: Pasien diprediksi memiliki risiko stroke

**2. Probabilitas Prediksi**

Sistem memberikan nilai probabilitas untuk setiap kelas:
- **Probabilitas Tidak Stroke**: Persentase keyakinan model bahwa pasien tidak berisiko (0-100%)
- **Probabilitas Stroke**: Persentase keyakinan model bahwa pasien berisiko (0-100%)

**3. Analisis Faktor Risiko**

Sistem mengidentifikasi faktor-faktor risiko yang terdeteksi pada pasien:
- Usia di atas 45/60 tahun
- Riwayat hipertensi
- Riwayat penyakit jantung
- Kadar glukosa tinggi
- BMI tinggi (overweight/obesitas)
- Status merokok

**4. Rekomendasi Kesehatan**

Berdasarkan hasil prediksi, sistem memberikan rekomendasi:
- **Risiko Rendah**: Saran untuk mempertahankan gaya hidup sehat
- **Risiko Tinggi**: Anjuran untuk segera berkonsultasi dengan dokter

**5. Visualisasi Hasil**

Sistem menyajikan hasil dalam bentuk:
- Grafik batang probabilitas
- Kotak hasil prediksi dengan kode warna (hijau/merah)
- Daftar faktor risiko yang teridentifikasi
- File CSV yang dapat diunduh untuk dokumentasi

**Akurasi dan Keandalan**

Model Decision Tree yang digunakan memiliki:
- **Akurasi**: sekitar 95%
- **Skor AUC**: 0.942

Namun perlu diingat bahwa hasil prediksi ini hanya merupakan **alat bantu skrining awal** dan 
**bukan pengganti diagnosis medis profesional**. Selalu konsultasikan dengan dokter untuk 
pemeriksaan dan diagnosis yang akurat.
""")

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Sistem Prediksi Stroke</strong></p>
    <p>Proyek Data Mining | 2024</p>
    <p>Dikembangkan dengan menggunakan Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)
