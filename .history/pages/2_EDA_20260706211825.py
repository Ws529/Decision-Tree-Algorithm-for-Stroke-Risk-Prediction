"""
HALAMAN EDA (Exploratory Data Analysis)
========================================
Visualisasi data dengan Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== CONFIG ====================
st.set_page_config(
    page_title="EDA - Stroke Prediction",
    page_icon="📈",
    layout="wide"
)

# ==================== HEADER ====================
st.title("📈 Exploratory Data Analysis (EDA)")
st.markdown("Analisis visual data untuk memahami pola dan hubungan antar variabel")
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
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📉 Distribution", "🔥 Correlation", "📊 Statistics"])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.markdown("## 📊 Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart distribusi stroke
            stroke_counts = df['stroke'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Tidak Stroke', 'Stroke'],
                values=stroke_counts.values,
                hole=0.4,
                marker=dict(colors=['#00CC96', '#EF553B']),
                textinfo='label+percent+value',
                textfont=dict(size=14)
            )])
            fig.update_layout(
                title='🎯 Distribusi Target Variable - Stroke',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            📌 **Insight:**
            - Tidak Stroke: {stroke_counts[0]:,} ({stroke_counts[0]/len(df)*100:.1f}%)
            - Stroke: {stroke_counts[1]:,} ({stroke_counts[1]/len(df)*100:.1f}%)
            - Dataset **sangat imbalanced** (rasio {stroke_counts[0]/stroke_counts[1]:.1f}:1)
            """)
        
        with col2:
            # Gender distribution
            gender_stroke = pd.crosstab(df['gender'], df['stroke'])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Tidak Stroke',
                x=gender_stroke.index,
                y=gender_stroke[0],
                marker_color='#00CC96'
            ))
            fig.add_trace(go.Bar(
                name='Stroke',
                x=gender_stroke.index,
                y=gender_stroke[1],
                marker_color='#EF553B'
            ))
            fig.update_layout(
                title='👥 Distribusi Gender vs Stroke',
                barmode='group',
                height=500,
                xaxis_title='Gender',
                yaxis_title='Jumlah'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            📌 **Insight:**
            - Distribusi gender relatif seimbang
            - Perlu analisis lebih lanjut untuk korelasi gender dengan stroke
            """)
        
        st.markdown("---")
        
        # Feature Distribution Overview
        st.markdown("### 📊 Feature Distribution Overview")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age Distribution', 'BMI Distribution', 
                          'Glucose Level Distribution', 'Stroke Distribution')
        )
        
        fig.add_trace(go.Histogram(x=df['age'], name='Age', marker_color='#636EFA'), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['bmi'].dropna(), name='BMI', marker_color='#EF553B'), row=1, col=2)
        fig.add_trace(go.Histogram(x=df['avg_glucose_level'], name='Glucose', marker_color='#00CC96'), row=2, col=1)
        fig.add_trace(go.Bar(x=['No Stroke', 'Stroke'], y=df['stroke'].value_counts().values,
                            marker_color=['#00CC96', '#EF553B']), row=2, col=2)
        
        fig.update_layout(height=700, showlegend=False, title_text='Feature Distribution Overview')
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: DISTRIBUTION ====================
    with tab2:
        st.markdown("## 📉 Distribusi Fitur")
        
        # Categorical features
        st.markdown("### 📊 Fitur Kategorikal")
        
        cat_col1, cat_col2 = st.columns(2)
        
        with cat_col1:
            # Hypertension
            hyper_stroke = pd.crosstab(df['hypertension'], df['stroke'])
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Tidak Stroke', x=['Tidak', 'Ya'], y=hyper_stroke[0].values, marker_color='#00CC96'))
            fig.add_trace(go.Bar(name='Stroke', x=['Tidak', 'Ya'], y=hyper_stroke[1].values, marker_color='#EF553B'))
            fig.update_layout(title='💊 Hypertension vs Stroke', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with cat_col2:
            # Heart Disease
            heart_stroke = pd.crosstab(df['heart_disease'], df['stroke'])
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Tidak Stroke', x=['Tidak', 'Ya'], y=heart_stroke[0].values, marker_color='#00CC96'))
            fig.add_trace(go.Bar(name='Stroke', x=['Tidak', 'Ya'], y=heart_stroke[1].values, marker_color='#EF553B'))
            fig.update_layout(title='❤️ Heart Disease vs Stroke', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Smoking Status
        smoking_stroke = pd.crosstab(df['smoking_status'], df['stroke'])
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Tidak Stroke', x=smoking_stroke.index, y=smoking_stroke[0], marker_color='#00CC96'))
        fig.add_trace(go.Bar(name='Stroke', x=smoking_stroke.index, y=smoking_stroke[1], marker_color='#EF553B'))
        fig.update_layout(title='🚬 Smoking Status vs Stroke', barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Numerical features
        st.markdown("### 📈 Fitur Numerikal")
        
        num_col1, num_col2 = st.columns(2)
        
        with num_col1:
            # Age histogram
            fig = px.histogram(df, x='age', color='stroke',
                             title='📊 Distribusi Usia',
                             color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                             barmode='overlay', nbins=40, opacity=0.7, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Usia Rata-rata", f"{df['age'].mean():.1f} tahun")
        
        with num_col2:
            # BMI histogram
            fig = px.histogram(df.dropna(subset=['bmi']), x='bmi', color='stroke',
                             title='⚖️ Distribusi BMI',
                             color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                             barmode='overlay', nbins=40, opacity=0.7, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("BMI Rata-rata", f"{df['bmi'].mean():.1f}")
        
        # Glucose histogram
        fig = px.histogram(df, x='avg_glucose_level', color='stroke',
                         title='🩸 Distribusi Glucose Level',
                         color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                         barmode='overlay', nbins=40, opacity=0.7, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Boxplots
        st.markdown("### 📦 Boxplot Analysis")
        
        box_col1, box_col2 = st.columns(2)
        
        with box_col1:
            fig = px.box(df, x='stroke', y='age',
                        title='📦 Boxplot Age vs Stroke',
                        color='stroke', color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                        height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with box_col2:
            fig = px.box(df, x='stroke', y='bmi',
                        title='📦 Boxplot BMI vs Stroke',
                        color='stroke', color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                        height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: CORRELATION ====================
    with tab3:
        st.markdown("## 🔥 Correlation Analysis")
        
        # Heatmap korelasi
        numeric_cols = df.select_dtypes(include=[np.number]).drop('id', axis=1, errors='ignore')
        correlation = numeric_cols.corr()
        
        fig = px.imshow(correlation, text_auto='.2f',
                       title='🔥 Heatmap Korelasi Antar Fitur',
                       color_continuous_scale='RdBu_r',
                       aspect='auto', height=600)
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Korelasi dengan target
        st.markdown("### 🎯 Korelasi dengan Target (Stroke)")
        
        corr_with_target = correlation['stroke'].sort_values(ascending=False).drop('stroke')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            corr_df = pd.DataFrame({
                'Feature': corr_with_target.index,
                'Correlation': corr_with_target.values
            })
            st.dataframe(corr_df, use_container_width=True, height=400)
        
        with col2:
            fig = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                        title='Korelasi Features dengan Stroke',
                        color='Correlation',
                        color_continuous_scale='RdBu_r',
                        height=400)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Scatter plots
        st.markdown("### 📈 Scatter Plot Analysis")
        
        scatter_col1, scatter_col2 = st.columns(2)
        
        with scatter_col1:
            fig = px.scatter(df, x='age', y='avg_glucose_level', color='stroke',
                           title='📈 Age vs Glucose Level',
                           color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                           opacity=0.6, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with scatter_col2:
            fig = px.scatter(df.dropna(subset=['bmi']), x='bmi', y='avg_glucose_level', color='stroke',
                           title='📈 BMI vs Glucose Level',
                           color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                           opacity=0.6, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 4: STATISTICS ====================
    with tab4:
        st.markdown("## 📊 Statistical Summary")
        
        # Summary statistics
        st.markdown("### 📈 Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Group by stroke
        st.markdown("### 🎯 Statistics by Stroke Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Tidak Stroke (0)")
            st.dataframe(df[df['stroke']==0].describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### Stroke (1)")
            st.dataframe(df[df['stroke']==1].describe(), use_container_width=True)

else:
    st.error("❌ Dataset tidak dapat dimuat!")
