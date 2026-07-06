"""
HALAMAN DATASET
===============
Menampilkan dan mengeksplorasi dataset stroke prediction
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Dataset - Stroke Prediction",
    page_icon="📊",
    layout="wide"
)

# ==================== HEADER ====================
st.title("📊 Dataset Healthcare Stroke Prediction")
st.markdown("---")

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        return df
    except:
        st.error("❌ File dataset tidak ditemukan!")
        return None

df = load_data()

if df is not None:
    # ==================== UPLOAD OPTION ====================
    st.markdown("## 📁 Upload Dataset (Opsional)")
    uploaded_file = st.file_uploader("Upload file CSV Anda sendiri", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset berhasil di-upload!")
    
    st.markdown("---")
    
    # ==================== OVERVIEW ====================
    st.markdown("## 📋 Overview Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Baris", f"{len(df):,}")
    
    with col2:
        st.metric("Total Kolom", df.shape[1])
    
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    with col4:
        st.metric("Duplicate Data", df.duplicated().sum())
    
    st.markdown("---")
    
    # ==================== PREVIEW DATA ====================
    st.markdown("## 👀 Preview Data")
    
    # Options
    col1, col2 = st.columns([1, 3])
    with col1:
        n_rows = st.slider("Jumlah baris yang ditampilkan", 5, 100, 10)
    with col2:
        show_all = st.checkbox("Tampilkan semua kolom", value=True)
    
    if show_all:
        st.dataframe(df.head(n_rows), use_container_width=True, height=400)
    else:
        st.dataframe(df.head(n_rows), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # ==================== INFO DATASET ====================
    st.markdown("## ℹ️ Informasi Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Tipe Data")
        dtype_df = pd.DataFrame({
            'Kolom': df.dtypes.index,
            'Tipe Data': df.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 🔍 Deskripsi Kolom")
        descriptions = {
            'id': 'ID unik pasien',
            'gender': 'Jenis kelamin',
            'age': 'Usia (tahun)',
            'hypertension': 'Hipertensi (0=Tidak, 1=Ya)',
            'heart_disease': 'Penyakit jantung (0=Tidak, 1=Ya)',
            'ever_married': 'Status menikah',
            'work_type': 'Jenis pekerjaan',
            'Residence_type': 'Tipe tempat tinggal',
            'avg_glucose_level': 'Kadar glukosa rata-rata',
            'bmi': 'Body Mass Index',
            'smoking_status': 'Status merokok',
            'stroke': 'Target (0=Tidak, 1=Ya)'
        }
        desc_df = pd.DataFrame(list(descriptions.items()), columns=['Kolom', 'Deskripsi'])
        st.dataframe(desc_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # ==================== STATISTIK DESKRIPTIF ====================
    st.markdown("## 📈 Statistik Deskriptif")
    
    tab1, tab2 = st.tabs(["Kolom Numerikal", "Kolom Kategorikal"])
    
    with tab1:
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.dataframe(df.describe(include=['object']), use_container_width=True)
    
    st.markdown("---")
    
    # ==================== MISSING VALUES ====================
    st.markdown("## ❌ Missing Values Analysis")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Kolom': missing.index,
        'Missing Values': missing.values,
        'Persentase (%)': missing_pct.values
    }).sort_values(by='Missing Values', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(missing_df, use_container_width=True, height=400)
    
    with col2:
        # Visualisasi missing values
        fig = px.bar(
            missing_df[missing_df['Missing Values'] > 0],
            x='Kolom',
            y='Missing Values',
            title='Missing Values per Kolom',
            color='Missing Values',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    if missing.sum() == 0:
        st.success("✅ Tidak ada missing values dalam dataset!")
    else:
        st.warning(f"⚠️ Terdapat {missing.sum()} missing values yang perlu ditangani")
    
    st.markdown("---")
    
    # ==================== DUPLICATE DATA ====================
    st.markdown("## 🔁 Duplicate Data Analysis")
    
    duplicates = df.duplicated().sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Duplicates", duplicates)
    
    with col2:
        st.metric("Persentase", f"{(duplicates/len(df)*100):.2f}%")
    
    with col3:
        st.metric("Data Unik", len(df) - duplicates)
    
    if duplicates == 0:
        st.success("✅ Tidak ada data duplikat!")
    else:
        st.warning(f"⚠️ Terdapat {duplicates} data duplikat")
    
    st.markdown("---")
    
    # ==================== DOWNLOAD ====================
    st.markdown("## 💾 Download Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='stroke_dataset.csv',
            mime='text/csv'
        )
    
    with col2:
        # Download description
        st.download_button(
            label="📥 Download Deskripsi",
            data=df.describe().to_csv().encode('utf-8'),
            file_name='dataset_description.csv',
            mime='text/csv'
        )
    
    with col3:
        # Download info
        buffer = df.info(buf=None)
        st.info("ℹ️ Download info dataset dari menu File → Export")

else:
    st.error("❌ Dataset tidak dapat dimuat. Pastikan file 'healthcare-dataset-stroke-data.csv' ada di folder yang sama dengan app.py")
