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
st.markdown('<p class="main-header">Sistem Prediksi Stroke</p>', unsafe_allow_html=True)
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
st.markdown("## Ringkasan Dataset")

if df is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Data",
            value=f"{len(df):,}",
            delta="5,110 pasien"
        )
    
    with col2:
        st.metric(
            label="Jumlah Fitur",
            value="11",
            delta="fitur"
        )
    
    with col3:
        stroke_pct = (df['stroke'].sum() / len(df)) * 100
        st.metric(
            label="Kasus Stroke",
            value=f"{df['stroke'].sum()}",
            delta=f"{stroke_pct:.1f}%"
        )
    
    with col4:
        if model is not None:
            st.metric(
                label="Akurasi Model",
                value="~95%",
                delta="Decision Tree"
            )
        else:
            st.metric(
                label="Model",
                value="Belum Dimuat",
                delta="Latih model dulu"
            )

st.markdown("---")

# =====================================================================
# QUICK STATS
# =====================================================================
st.markdown("## Statistik Cepat")

if df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribusi Gender")
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
        st.markdown("### Distribusi Stroke")
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
st.markdown("## Fitur Utama Aplikasi")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>Eksplorasi Dataset</h3>
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
        <h3>EDA Interaktif</h3>
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
        <h3>Performa Model</h3>
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
st.markdown("## Tentang Stroke")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Apa itu Stroke?
    
    **Stroke** adalah kondisi medis serius yang terjadi ketika aliran darah ke bagian otak terganggu atau berkurang, 
    sehingga mencegah jaringan otak mendapatkan oksigen dan nutrisi. Sel-sel otak mulai mati dalam hitungan menit.
    
    ### Faktor Risiko Stroke:
    
    1. **Usia**: Risiko meningkat seiring bertambahnya usia
    2. **Hipertensi**: Tekanan darah tinggi
    3. **Penyakit Jantung**: Riwayat penyakit jantung
    4. **Diabetes**: Kadar gula darah tinggi
    5. **Obesitas**: BMI tinggi
    6. **Merokok**: Kebiasaan merokok
    7. **Kolesterol Tinggi**: Kadar kolesterol tidak normal
    
    ### Pencegahan:
    
    - Jaga pola makan sehat
    - Olahraga teratur
    - Kontrol tekanan darah
    - Hindari merokok
    - Kelola stres
    - Cek kesehatan rutin
    """)

with col2:
    st.markdown("### Statistik Penting")
    
    if df is not None:
        st.info(f"""
        **Total Pasien**: {len(df):,}
        
        **Kasus Stroke**: {df['stroke'].sum()}
        
        **Persentase**: {(df['stroke'].sum()/len(df)*100):.2f}%
        
        **Usia Rata-rata**: {df['age'].mean():.1f} tahun
        
        **BMI Rata-rata**: {df['bmi'].mean():.1f}
        """)
    
    st.success("""
    **Deteksi Dini**
    
    Sistem ini membantu mendeteksi risiko stroke lebih awal menggunakan machine learning.
    """)

st.markdown("---")

# =====================================================================
# PREDIKSI STROKE
# =====================================================================
st.markdown("---")
st.subheader("Prediksi Risiko Stroke")
st.write("Masukkan data pasien untuk memprediksi risiko stroke")

# Cek model
if model is not None and scaler is not None and encoder is not None:
    st.success("Model siap digunakan")
    
    with st.expander("Informasi Penting"):
        st.markdown("""
        **Cara Menggunakan:**
        1. Isi semua kolom di formulir dengan data pasien
        2. Klik tombol Prediksi Stroke
        3. Lihat hasil prediksi dan probabilitas
        
        **Disclaimer:**
        - Hasil prediksi hanya untuk skrining awal
        - BUKAN pengganti diagnosis medis profesional
        - Konsultasikan dengan dokter untuk pemeriksaan lebih lanjut
        - Akurasi model: sekitar 95%
        """)
    
    st.markdown("---")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Informasi Pribadi**")
            gender = st.selectbox("Jenis Kelamin", options=["Male", "Female", "Other"])
            age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=45, step=1)
            ever_married = st.selectbox("Status Pernikahan", options=["Yes", "No"])
            work_type = st.selectbox("Jenis Pekerjaan", 
                                     options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence_type = st.selectbox("Tipe Tempat Tinggal", options=["Urban", "Rural"])
        
        with col2:
            st.markdown("**Riwayat Kesehatan**")
            hypertension = st.selectbox("Hipertensi", options=[0, 1], 
                                        format_func=lambda x: "Ya" if x == 1 else "Tidak")
            heart_disease = st.selectbox("Penyakit Jantung", options=[0, 1], 
                                         format_func=lambda x: "Ya" if x == 1 else "Tidak")
            smoking_status = st.selectbox("Status Merokok", 
                                          options=["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        with col3:
            st.markdown("**Data Medis**")
            avg_glucose_level = st.number_input("Kadar Glukosa Rata-rata (mg/dL)", 
                                               min_value=50.0, max_value=300.0, value=106.0, step=0.1)
            bmi = st.number_input("BMI (Body Mass Index)", 
                                 min_value=10.0, max_value=100.0, value=28.0, step=0.1)
            st.caption("Referensi BMI: Kurus <18.5 | Normal 18.5-24.9 | Gemuk 25-29.9 | Obesitas ≥30")
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("Prediksi Stroke", use_container_width=True, type="primary")
    
    if submit_button:
        with st.spinner("Memproses prediksi..."):
            import time
            time.sleep(1)
            
            try:
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'age': [age],
                    'hypertension': [hypertension],
                    'heart_disease': [heart_disease],
                    'ever_married': [ever_married],
                    'work_type': [work_type],
                    'Residence_type': [residence_type],
                    'avg_glucose_level': [avg_glucose_level],
                    'bmi': [bmi],
                    'smoking_status': [smoking_status]
                })
                
                categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
                
                for col in categorical_cols:
                    if col in encoder:
                        le = encoder[col]
                        try:
                            input_data[col] = le.transform(input_data[col])
                        except:
                            input_data[col] = 0
                
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                st.markdown("---")
                st.subheader("Hasil Prediksi")
                
                prob_no_stroke = probability[0] * 100
                prob_stroke = probability[1] * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Probabilitas Tidak Stroke**")
                    st.progress(prob_no_stroke / 100)
                    st.markdown(f"<h2 style='text-align:center; color:#28a745;'>{prob_no_stroke:.2f}%</h2>", 
                               unsafe_allow_html=True)
                
                with col2:
                    st.write("**Probabilitas Stroke**")
                    st.progress(prob_stroke / 100)
                    st.markdown(f"<h2 style='text-align:center; color:#dc3545;'>{prob_stroke:.2f}%</h2>", 
                               unsafe_allow_html=True)
                
                st.markdown("---")
                
                if prediction == 0:
                    st.markdown("""
                    <div style='padding:20px; border-radius:10px; background-color:#d4edda; 
                    border:3px solid #28a745; text-align:center; color:#155724;'>
                        <h2>RISIKO RENDAH</h2>
                        <p>Pasien diprediksi TIDAK memiliki risiko stroke</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success(f"""
                    **Hasil: Risiko Rendah**
                    
                    Berdasarkan data yang diberikan, model memprediksi bahwa pasien memiliki risiko rendah 
                    terkena stroke dengan tingkat kepercayaan {prob_no_stroke:.2f}%.
                    
                    **Rekomendasi:**
                    - Pertahankan gaya hidup sehat
                    - Olahraga teratur
                    - Cek kesehatan rutin
                    - Pola makan seimbang
                    """)
                    
                else:
                    st.markdown("""
                    <div style='padding:20px; border-radius:10px; background-color:#f8d7da; 
                    border:3px solid #dc3545; text-align:center; color:#721c24;'>
                        <h2>RISIKO TINGGI</h2>
                        <p>Pasien diprediksi memiliki risiko stroke</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.error(f"""
                    **Hasil: Risiko Tinggi**
                    
                    Berdasarkan data yang diberikan, model memprediksi bahwa pasien memiliki risiko tinggi 
                    terkena stroke dengan tingkat kepercayaan {prob_stroke:.2f}%.
                    
                    **Rekomendasi Mendesak:**
                    - Segera konsultasi dengan dokter
                    - Pemeriksaan medis menyeluruh
                    - Kontrol tekanan darah dan gula darah
                    - Mulai program olahraga
                    - Diet sehat rendah garam dan lemak
                    - Hentikan kebiasaan merokok (jika merokok)
                    """)
                
                st.markdown("---")
                st.write("**Visualisasi Probabilitas**")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Tidak Stroke', 'Stroke'],
                        y=[prob_no_stroke, prob_stroke],
                        text=[f'{prob_no_stroke:.2f}%', f'{prob_stroke:.2f}%'],
                        textposition='auto',
                        marker=dict(color=['#28a745', '#dc3545'])
                    )
                ])
                
                fig.update_layout(
                    title='Distribusi Probabilitas Prediksi',
                    xaxis_title='Kategori',
                    yaxis_title='Probabilitas (%)',
                    height=400,
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.write("**Analisis Faktor Risiko**")
                
                risk_factors = []
                
                if age > 60:
                    risk_factors.append("Usia > 60 tahun (Risiko tinggi)")
                elif age > 45:
                    risk_factors.append("Usia > 45 tahun (Risiko sedang)")
                
                if hypertension == 1:
                    risk_factors.append("Memiliki hipertensi")
                
                if heart_disease == 1:
                    risk_factors.append("Memiliki penyakit jantung")
                
                if avg_glucose_level > 140:
                    risk_factors.append("Kadar glukosa tinggi (> 140 mg/dL)")
                elif avg_glucose_level > 110:
                    risk_factors.append("Kadar glukosa borderline (110-140 mg/dL)")
                
                if bmi > 30:
                    risk_factors.append("Obesitas (BMI > 30)")
                elif bmi > 25:
                    risk_factors.append("Kelebihan berat badan (BMI 25-30)")
                
                if smoking_status == "smokes":
                    risk_factors.append("Perokok aktif")
                elif smoking_status == "formerly smoked":
                    risk_factors.append("Mantan perokok")
                
                if risk_factors:
                    st.warning("**Faktor Risiko yang Teridentifikasi:**")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.success("Tidak ada faktor risiko utama yang teridentifikasi")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Pastikan model telah dilatih dengan benar")
else:
    st.warning("Model belum dimuat. Silakan latih model terlebih dahulu menggunakan Jupyter Notebook.")

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
