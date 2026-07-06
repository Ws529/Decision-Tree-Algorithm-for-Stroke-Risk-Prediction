import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="EDA - Stroke Prediction",
    page_icon="📈",
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
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD DATA
# =====================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        return df
    except:
        return None

df = load_data()

if df is None:
    st.error("❌ Dataset tidak ditemukan!")
    st.stop()

# =====================================================================
# HEADER
# =====================================================================
st.markdown('<p class="main-header">📈 Exploratory Data Analysis</p>', unsafe_allow_html=True)
st.markdown("Visualisasi interaktif untuk memahami pola dan hubungan dalam data")
st.markdown("---")

# =====================================================================
# TABS FOR VISUALIZATIONS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📉 Distribusi",
    "🔗 Korelasi",
    "📈 Advanced"
])

# =====================================================================
# TAB 1: OVERVIEW
# =====================================================================
with tab1:
    st.markdown("## 🎯 Overview Visualisasi")
    
    # 1. Distribusi Stroke
    st.markdown("### 1. Distribusi Target (Stroke)")
    stroke_counts = df['stroke'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=['Tidak Stroke', 'Stroke'],
            values=stroke_counts.values,
            hole=0.4,
            marker=dict(colors=['#00CC96', '#EF553B']),
            textinfo='label+percent+value',
            textfont=dict(size=14)
        )])
        fig.update_layout(
            title='Distribusi Stroke (Pie Chart)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Statistik:")
        st.metric("Total Pasien", f"{len(df):,}")
        st.metric("Tidak Stroke", f"{stroke_counts[0]:,} ({stroke_counts[0]/len(df)*100:.1f}%)")
        st.metric("Stroke", f"{stroke_counts[1]:,} ({stroke_counts[1]/len(df)*100:.1f}%)")
        st.warning(f"⚠️ Dataset Imbalanced\n\nRatio: {stroke_counts[0]/stroke_counts[1]:.1f}:1")
    
    st.markdown("---")
    
    # 2. Distribusi Gender
    st.markdown("### 2. Distribusi Gender vs Stroke")
    gender_stroke = pd.crosstab(df['gender'], df['stroke'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Tidak Stroke',
        x=gender_stroke.index,
        y=gender_stroke[0],
        marker_color='#00CC96',
        text=gender_stroke[0],
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name='Stroke',
        x=gender_stroke.index,
        y=gender_stroke[1],
        marker_color='#EF553B',
        text=gender_stroke[1],
        textposition='auto'
    ))
    fig.update_layout(
        title='👥 Gender vs Stroke',
        xaxis_title='Gender',
        yaxis_title='Jumlah',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 3 & 4. Hypertension dan Heart Disease
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 3. Hypertension vs Stroke")
        hyper_stroke = pd.crosstab(df['hypertension'], df['stroke'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Tidak Stroke',
            x=['Tidak', 'Ya'],
            y=hyper_stroke[0].values,
            marker_color='#00CC96'
        ))
        fig.add_trace(go.Bar(
            name='Stroke',
            x=['Tidak', 'Ya'],
            y=hyper_stroke[1].values,
            marker_color='#EF553B'
        ))
        fig.update_layout(
            title='💊 Hypertension vs Stroke',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 4. Heart Disease vs Stroke")
        heart_stroke = pd.crosstab(df['heart_disease'], df['stroke'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Tidak Stroke',
            x=['Tidak', 'Ya'],
            y=heart_stroke[0].values,
            marker_color='#00CC96'
        ))
        fig.add_trace(go.Bar(
            name='Stroke',
            x=['Tidak', 'Ya'],
            y=heart_stroke[1].values,
            marker_color='#EF553B'
        ))
        fig.update_layout(
            title='❤️ Heart Disease vs Stroke',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 5. Smoking Status
    st.markdown("### 5. Smoking Status vs Stroke")
    smoking_stroke = pd.crosstab(df['smoking_status'], df['stroke'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Tidak Stroke',
        x=smoking_stroke.index,
        y=smoking_stroke[0],
        marker_color='#00CC96'
    ))
    fig.add_trace(go.Bar(
        name='Stroke',
        x=smoking_stroke.index,
        y=smoking_stroke[1],
        marker_color='#EF553B'
    ))
    fig.update_layout(
        title='🚬 Smoking Status vs Stroke',
        xaxis_title='Status',
        yaxis_title='Jumlah',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 2: DISTRIBUSI
# =====================================================================
with tab2:
    st.markdown("## 📉 Distribusi Fitur Numerikal")
    
    # 6. Age Distribution
    st.markdown("### 6. Distribusi Usia")
    fig = px.histogram(
        df,
        x='age',
        color='stroke',
        title='📊 Distribusi Usia Pasien',
        labels={'age': 'Usia (tahun)', 'stroke': 'Stroke'},
        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
        barmode='overlay',
        nbins=40,
        opacity=0.7
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Usia Rata-rata", f"{df['age'].mean():.1f} tahun")
    with col2:
        st.metric("Usia Median", f"{df['age'].median():.1f} tahun")
    with col3:
        st.metric("Usia Range", f"{df['age'].min():.1f} - {df['age'].max():.1f}")
    
    st.markdown("---")
    
    # 7. BMI Distribution
    st.markdown("### 7. Distribusi BMI")
    fig = px.histogram(
        df.dropna(subset=['bmi']),
        x='bmi',
        color='stroke',
        title='⚖️ Distribusi BMI (Body Mass Index)',
        labels={'bmi': 'BMI', 'stroke': 'Stroke'},
        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
        barmode='overlay',
        nbins=40,
        opacity=0.7
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BMI Rata-rata", f"{df['bmi'].mean():.2f}")
    with col2:
        st.metric("BMI Median", f"{df['bmi'].median():.2f}")
    with col3:
        st.metric("Missing Values", f"{df['bmi'].isnull().sum()}")
    
    st.markdown("---")
    
    # 8. Glucose Distribution
    st.markdown("### 8. Distribusi Glucose Level")
    fig = px.histogram(
        df,
        x='avg_glucose_level',
        color='stroke',
        title='🩸 Distribusi Average Glucose Level',
        labels={'avg_glucose_level': 'Glucose Level (mg/dL)', 'stroke': 'Stroke'},
        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
        barmode='overlay',
        nbins=40,
        opacity=0.7
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Glucose Rata-rata", f"{df['avg_glucose_level'].mean():.2f} mg/dL")
    with col2:
        st.metric("Glucose Median", f"{df['avg_glucose_level'].median():.2f} mg/dL")
    with col3:
        st.metric("Glucose Max", f"{df['avg_glucose_level'].max():.2f} mg/dL")
    
    st.markdown("---")
    
    # 9 & 10. Boxplots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 9. Boxplot BMI")
        fig = px.box(
            df,
            x='stroke',
            y='bmi',
            color='stroke',
            title='📦 Boxplot BMI vs Stroke',
            color_discrete_map={0: '#00CC96', 1: '#EF553B'},
            labels={'stroke': 'Stroke (0=Tidak, 1=Ya)'}
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 10. Boxplot Age")
        fig = px.box(
            df,
            x='stroke',
            y='age',
            color='stroke',
            title='📦 Boxplot Age vs Stroke',
            color_discrete_map={0: '#00CC96', 1: '#EF553B'},
            labels={'stroke': 'Stroke (0=Tidak, 1=Ya)'}
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 3: KORELASI
# =====================================================================
with tab3:
    st.markdown("## 🔗 Analisis Korelasi")
    
    # 11. Heatmap Korelasi
    st.markdown("### 11. Heatmap Korelasi")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'id']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        title='🔥 Heatmap Korelasi Antar Fitur',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Korelasi dengan Stroke:")
    stroke_corr = corr_matrix['stroke'].sort_values(ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(stroke_corr, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=stroke_corr.values[1:],  # Exclude stroke vs stroke
            y=stroke_corr.index[1:],
            orientation='h',
            title='Korelasi Fitur dengan Stroke',
            labels={'x': 'Correlation', 'y': 'Feature'},
            color=stroke_corr.values[1:],
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 4: ADVANCED
# =====================================================================
with tab4:
    st.markdown("## 📈 Visualisasi Advanced")
    
    # 12. Scatter Age vs Glucose
    st.markdown("### 12. Scatter Plot: Age vs Glucose Level")
    fig = px.scatter(
        df,
        x='age',
        y='avg_glucose_level',
        color='stroke',
        title='📈 Relationship: Age vs Glucose Level',
        labels={'age': 'Usia (tahun)', 'avg_glucose_level': 'Glucose (mg/dL)'},
        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
        opacity=0.6,
        hover_data=['gender', 'bmi']
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 13. Scatter BMI vs Glucose
    st.markdown("### 13. Scatter Plot: BMI vs Glucose Level")
    fig = px.scatter(
        df.dropna(subset=['bmi']),
        x='bmi',
        y='avg_glucose_level',
        color='stroke',
        title='📈 Relationship: BMI vs Glucose Level',
        labels={'bmi': 'BMI', 'avg_glucose_level': 'Glucose (mg/dL)'},
        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
        opacity=0.6,
        hover_data=['age', 'gender']
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 14. Feature Distribution Overview
    st.markdown("### 14. Feature Distribution Overview")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Age Distribution', 'BMI Distribution',
                       'Glucose Distribution', 'Stroke Distribution')
    )
    
    fig.add_trace(go.Histogram(x=df['age'], name='Age', marker_color='#636EFA'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['bmi'].dropna(), name='BMI', marker_color='#EF553B'), row=1, col=2)
    fig.add_trace(go.Histogram(x=df['avg_glucose_level'], name='Glucose', marker_color='#00CC96'), row=2, col=1)
    fig.add_trace(go.Bar(x=['No', 'Yes'], y=df['stroke'].value_counts().values, marker_color=['#00CC96', '#EF553B']), row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False, title_text='📊 Overview Distribusi Fitur Utama')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 15. Scatter Matrix
    st.markdown("### 15. Scatter Matrix (Pairplot)")
    
    features_to_plot = st.multiselect(
        "Pilih fitur untuk scatter matrix:",
        ['age', 'bmi', 'avg_glucose_level', 'hypertension', 'heart_disease'],
        default=['age', 'bmi', 'avg_glucose_level']
    )
    
    if len(features_to_plot) >= 2:
        fig = px.scatter_matrix(
            df[features_to_plot + ['stroke']].dropna(),
            dimensions=features_to_plot,
            color='stroke',
            title='🔍 Scatter Matrix: Relationship Antar Fitur',
            color_discrete_map={0: '#00CC96', 1: '#EF553B'},
            opacity=0.6
        )
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pilih minimal 2 fitur untuk scatter matrix")

# =====================================================================
# INSIGHTS SUMMARY
# =====================================================================
st.markdown("---")
st.markdown("## 💡 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **🎯 Dataset Imbalanced**
    
    95% pasien tidak stroke, 5% stroke.
    Perlu handling khusus saat modeling.
    """)

with col2:
    st.success("""
    **📈 Age is Key**
    
    Usia memiliki korelasi tertinggi dengan stroke.
    Pasien lebih tua = risiko lebih tinggi.
    """)

with col3:
    st.warning("""
    **💊 Health Conditions**
    
    Hipertensi & penyakit jantung meningkatkan risiko stroke significantly.
    """)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>EDA Dashboard | Stroke Prediction System</p>
</div>
""", unsafe_allow_html=True)
