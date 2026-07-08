import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Dataset - Stroke Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00cc96;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HEADER
# =====================================================================
st.markdown('<p class="main-header">📁 Dataset Explorer</p>', unsafe_allow_html=True)
st.markdown("Jelajahi dataset **Healthcare Stroke Prediction** secara detail")
st.markdown("---")

# =====================================================================
# LOAD DATA
# =====================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("❌ Dataset tidak ditemukan!")
    st.stop()

# =====================================================================
# DATASET INFO
# =====================================================================
st.markdown("## 📊 Informasi Dataset")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Baris", f"{len(df):,}")

with col2:
    st.metric("Total Kolom", f"{len(df.columns)}")

with col3:
    st.metric("Missing Values", f"{df.isnull().sum().sum()}")

with col4:
    st.metric("Duplicates", f"{df.duplicated().sum()}")

with col5:
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    st.metric("Memory Usage", f"{memory_mb:.2f} MB")

st.markdown("---")

# =====================================================================
# UPLOAD FEATURE
# =====================================================================
st.markdown("## 📤 Upload Dataset (Opsional)")

uploaded_file = st.file_uploader(
    "Upload file CSV untuk analisis",
    type=['csv'],
    help="Upload dataset CSV untuk dianalisis"
)

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset berhasil diupload! Shape: {df_uploaded.shape}")
        
        if st.checkbox("Gunakan dataset yang diupload"):
            df = df_uploaded
            st.info("Dataset yang digunakan: Dataset yang diupload")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# =====================================================================
# DATA PREVIEW
# =====================================================================
st.markdown("## 👀 Preview Data")

# Pilihan jumlah baris
n_rows = st.slider("Jumlah baris yang ditampilkan:", 5, 100, 10)

# Pilihan tampilan
view_option = st.radio(
    "Pilih tampilan:",
    ["Head (Awal)", "Tail (Akhir)", "Sample (Random)"],
    horizontal=True
)

if view_option == "Head (Awal)":
    st.dataframe(df.head(n_rows), use_container_width=True, height=400)
elif view_option == "Tail (Akhir)":
    st.dataframe(df.tail(n_rows), use_container_width=True, height=400)
else:
    st.dataframe(df.sample(n_rows), use_container_width=True, height=400)

st.markdown("---")

# =====================================================================
# COLUMN INFORMATION
# =====================================================================
st.markdown("## 📋 Informasi Kolom")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Daftar Kolom")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        st.write(f"{i}. **{col}** ({dtype})")

with col2:
    st.markdown("### Penjelasan Fitur")
    st.markdown("""
    - **id**: Identifikasi unik pasien
    - **gender**: Jenis kelamin (Male/Female/Other)
    - **age**: Usia pasien (tahun)
    - **hypertension**: Hipertensi (0=Tidak, 1=Ya)
    - **heart_disease**: Penyakit jantung (0=Tidak, 1=Ya)
    - **ever_married**: Status menikah (Yes/No)
    - **work_type**: Jenis pekerjaan
    - **Residence_type**: Tipe tempat tinggal (Urban/Rural)
    - **avg_glucose_level**: Rata-rata kadar glukosa (mg/dL)
    - **bmi**: Body Mass Index
    - **smoking_status**: Status merokok
    - **stroke**: Target (0=Tidak, 1=Ya)
    """)

st.markdown("---")

# =====================================================================
# DESCRIPTIVE STATISTICS
# =====================================================================
st.markdown("## 📊 Statistik Deskriptif")

tab1, tab2, tab3 = st.tabs(["Numerikal", "Kategorikal", "Missing Values"])

with tab1:
    st.markdown("### Statistik Fitur Numerikal")
    st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.markdown("### Statistik Fitur Kategorikal")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        st.markdown(f"#### {col}")
        value_counts = df[col].value_counts()
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.dataframe(value_counts, use_container_width=True)
        
        with col_b:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': col, 'y': 'Count'},
                title=f'Distribusi {col}',
                color=value_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Missing Values Analysis")
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    if len(missing_data) > 0:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(missing_data, use_container_width=True)
        
        with col2:
            fig = px.bar(
                missing_data,
                x='Column',
                y='Missing Percentage',
                title='Missing Values Percentage',
                color='Missing Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ Tidak ada missing values!")

st.markdown("---")

# =====================================================================
# DATA TYPES
# =====================================================================
st.markdown("## 🔢 Tipe Data")

dtype_df = pd.DataFrame({
    'Column': df.dtypes.index,
    'Data Type': df.dtypes.values.astype(str),
    'Non-Null Count': df.count().values,
    'Null Count': df.isnull().sum().values
})

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(dtype_df, use_container_width=True, height=400)

with col2:
    # Pie chart tipe data
    dtype_counts = df.dtypes.value_counts()
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title='Distribusi Tipe Data',
        hole=0.4
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================================================
# CORRELATION PREVIEW
# =====================================================================
st.markdown("## 🔗 Preview Korelasi")

numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Tidak cukup kolom numerikal untuk korelasi")

st.markdown("---")

# =====================================================================
# DOWNLOAD DATA
# =====================================================================
st.markdown("## 💾 Download Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='stroke_dataset.csv',
        mime='text/csv',
        use_container_width=True
    )

with col2:
    # Download only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    csv_numeric = numeric_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Numerikal",
        data=csv_numeric,
        file_name='stroke_numeric.csv',
        mime='text/csv',
        use_container_width=True
    )

with col3:
    # Download statistics
    stats = df.describe().T
    csv_stats = stats.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Statistik",
        data=csv_stats,
        file_name='stroke_statistics.csv',
        mime='text/csv',
        use_container_width=True
    )

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Dataset Explorer | Stroke Prediction System</p>
</div>
""", unsafe_allow_html=True)
