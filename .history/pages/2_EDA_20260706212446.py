"""
Page 2: Exploratory Data Analysis (EDA)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="EDA - Stroke Prediction",
    page_icon="📊",
    layout="wide"
)

# ========================================
# LOAD DATA
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return df

# ========================================
# MAIN
# ========================================
def main():
    st.markdown('<h1 style="text-align: center; color: #1f77b4;">📊 EXPLORATORY DATA ANALYSIS</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    df = load_data()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overview", "📈 Distribution", "🔗 Correlation", "📉 Statistics"])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.markdown("### 🎯 Dataset Overview")
        
        # Stroke distribution - Pie Chart
        col1, col2 = st.columns([1, 1])
        
        with col1:
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
                title='🎯 Distribusi Target - Stroke',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Stroke Statistics")
            st.metric("Total Data", f"{len(df):,}")
            st.metric("Tidak Stroke", f"{stroke_counts[0]:,} ({stroke_counts[0]/len(df)*100:.2f}%)", delta=None)
            st.metric("Stroke", f"{stroke_counts[1]:,} ({stroke_counts[1]/len(df)*100:.2f}%)", delta=None)
            
            ratio = stroke_counts[0] / stroke_counts[1]
            st.warning(f"⚠️ **Dataset Imbalanced**  \nRatio: {ratio:.1f}:1")
        
        st.markdown("---")
        
        # Feature Distribution Overview
        st.markdown("### 📊 Feature Distribution Overview")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age Distribution', 'BMI Distribution', 
                          'Glucose Level Distribution', 'Stroke Count')
        )
        
        fig.add_trace(go.Histogram(x=df['age'], marker_color='#636EFA', name='Age'), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['bmi'].dropna(), marker_color='#EF553B', name='BMI'), row=1, col=2)
        fig.add_trace(go.Histogram(x=df['avg_glucose_level'], marker_color='#00CC96', name='Glucose'), row=2, col=1)
        fig.add_trace(go.Bar(x=['No Stroke', 'Stroke'], y=stroke_counts.values, 
                            marker_color=['#00CC96', '#EF553B'], name='Stroke'), row=2, col=2)
        
        fig.update_layout(height=700, showlegend=False, title_text="Overview Distribusi Fitur")
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: DISTRIBUTION ====================
    with tab2:
        st.markdown("### 📈 Feature Distribution Analysis")
        
        # Gender vs Stroke
        st.markdown("#### 👥 Gender Distribution")
        gender_stroke = pd.crosstab(df['gender'], df['stroke'])
        gender_stroke.columns = ['Tidak Stroke', 'Stroke']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Tidak Stroke', x=gender_stroke.index, y=gender_stroke['Tidak Stroke'], marker_color='#00CC96'))
        fig.add_trace(go.Bar(name='Stroke', x=gender_stroke.index, y=gender_stroke['Stroke'], marker_color='#EF553B'))
        fig.update_layout(title='Gender vs Stroke', barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        # Hypertension
        with col1:
            st.markdown("#### 💊 Hypertension vs Stroke")
            hyper_stroke = pd.crosstab(df['hypertension'], df['stroke'])
            hyper_stroke.index = ['Tidak Hipertensi', 'Hipertensi']
            hyper_stroke.columns = ['Tidak Stroke', 'Stroke']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Tidak Stroke', x=hyper_stroke.index, y=hyper_stroke['Tidak Stroke'], marker_color='#00CC96'))
            fig.add_trace(go.Bar(name='Stroke', x=hyper_stroke.index, y=hyper_stroke['Stroke'], marker_color='#EF553B'))
            fig.update_layout(title='Hypertension vs Stroke', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Heart Disease
        with col2:
            st.markdown("#### ❤️ Heart Disease vs Stroke")
            heart_stroke = pd.crosstab(df['heart_disease'], df['stroke'])
            heart_stroke.index = ['Tidak Penyakit Jantung', 'Penyakit Jantung']
            heart_stroke.columns = ['Tidak Stroke', 'Stroke']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Tidak Stroke', x=heart_stroke.index, y=heart_stroke['Tidak Stroke'], marker_color='#00CC96'))
            fig.add_trace(go.Bar(name='Stroke', x=heart_stroke.index, y=heart_stroke['Stroke'], marker_color='#EF553B'))
            fig.update_layout(title='Heart Disease vs Stroke', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Smoking Status
        st.markdown("#### 🚬 Smoking Status vs Stroke")
        smoking_stroke = pd.crosstab(df['smoking_status'], df['stroke'])
        smoking_stroke.columns = ['Tidak Stroke', 'Stroke']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Tidak Stroke', x=smoking_stroke.index, y=smoking_stroke['Tidak Stroke'], marker_color='#00CC96'))
        fig.add_trace(go.Bar(name='Stroke', x=smoking_stroke.index, y=smoking_stroke['Stroke'], marker_color='#EF553B'))
        fig.update_layout(title='Smoking Status vs Stroke', barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Age, BMI, Glucose - Histograms
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Age Distribution")
            fig = px.histogram(df, x='age', color='stroke',
                             color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                             barmode='overlay', nbins=40, opacity=0.7, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚖️ BMI Distribution")
            fig = px.histogram(df.dropna(subset=['bmi']), x='bmi', color='stroke',
                             color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                             barmode='overlay', nbins=40, opacity=0.7, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("#### 🩸 Glucose Distribution")
            fig = px.histogram(df, x='avg_glucose_level', color='stroke',
                             color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                             barmode='overlay', nbins=40, opacity=0.7, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Boxplots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Age Boxplot")
            fig = px.box(df, x='stroke', y='age', color='stroke',
                        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                        height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📦 BMI Boxplot")
            fig = px.box(df, x='stroke', y='bmi', color='stroke',
                        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                        height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: CORRELATION ====================
    with tab3:
        st.markdown("### 🔗 Correlation Analysis")
        
        # Heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).drop('id', axis=1, errors='ignore')
        correlation = numeric_cols.corr()
        
        fig = px.imshow(correlation, text_auto='.2f',
                       title='🔥 Heatmap Korelasi Antar Fitur',
                       color_continuous_scale='RdBu_r',
                       aspect='auto', height=600)
        fig.update_layout(title_x=0.5, title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with target
        st.markdown("#### 🎯 Korelasi dengan Target (Stroke)")
        target_corr = correlation['stroke'].sort_values(ascending=False)
        target_corr_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation': target_corr.values
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(target_corr_df, x='Feature', y='Correlation',
                        title='Korelasi dengan Stroke',
                        color='Correlation',
                        color_continuous_scale='RdYlGn',
                        height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(target_corr_df, use_container_width=True, height=400)
        
        # Scatter Plots
        st.markdown("### 📈 Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Age vs Glucose Level")
            fig = px.scatter(df, x='age', y='avg_glucose_level', color='stroke',
                           color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                           opacity=0.6, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### BMI vs Glucose Level")
            fig = px.scatter(df.dropna(subset=['bmi']), x='bmi', y='avg_glucose_level', color='stroke',
                           color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                           opacity=0.6, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter Matrix
        st.markdown("#### 🔍 Scatter Matrix")
        fig = px.scatter_matrix(df[['age', 'bmi', 'avg_glucose_level', 'stroke']].dropna(),
                               dimensions=['age', 'bmi', 'avg_glucose_level'],
                               color='stroke',
                               color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                               height=700, opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 4: STATISTICS ====================
    with tab4:
        st.markdown("### 📉 Statistical Summary")
        
        # Descriptive statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔢 Numerical Features")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### 📝 Categorical Features")
            st.dataframe(df.describe(include=['object']), use_container_width=True)
        
        st.markdown("---")
        
        # Statistics by Stroke
        st.markdown("#### 🎯 Statistics by Stroke Status")
        
        stroke_stats = df.groupby('stroke')[['age', 'bmi', 'avg_glucose_level']].agg(['mean', 'median', 'std'])
        st.dataframe(stroke_stats, use_container_width=True)
        
        # Visualize comparison
        metrics = ['age', 'bmi', 'avg_glucose_level']
        selected_metric = st.selectbox("Pilih Metrik:", metrics)
        
        fig = go.Figure()
        for stroke_val in [0, 1]:
            data = df[df['stroke'] == stroke_val][selected_metric].dropna()
            label = 'Tidak Stroke' if stroke_val == 0 else 'Stroke'
            color = '#00CC96' if stroke_val == 0 else '#EF553B'
            
            fig.add_trace(go.Box(y=data, name=label, marker_color=color))
        
        fig.update_layout(title=f'Comparison of {selected_metric} by Stroke Status',
                         yaxis_title=selected_metric, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    main()
