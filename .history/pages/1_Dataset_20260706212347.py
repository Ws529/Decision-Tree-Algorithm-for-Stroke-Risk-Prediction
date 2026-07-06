"""
Page 1: Dataset Explorer
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Dataset - Stroke Prediction",
    page_icon="📁",
    layout="wide"
)

# ========================================
# CUSTOM CSS
# ========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

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
    st.markdown('<h1 class="main-header">📁 DATASET EXPLORER</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Preview Data", "ℹ️ Info Dataset", "📈 Statistik", "❌ Missing Values", "💾 Download"])
    
    # ==================== TAB 1: PREVIEW DATA ====================
    with tab1:
        st.markdown("### 📊 Preview Dataset")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Menampilkan data dari total **{len(df):,}** baris")
        with col2:
            show_rows = st.selectbox("Tampilkan baris:", [10, 25, 50, 100, "All"])
        
        if show_rows == "All":
            st.dataframe(df, use_container_width=True, height=600)
        else:
            st.dataframe(df.head(show_rows), use_container_width=True)
        
        # Quick stats
        st.markdown("#### 📌 Quick Info")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
    
    # ==================== TAB 2: INFO DATASET ====================
    with tab2:
        st.markdown("### ℹ️ Informasi Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Deskripsi Kolom")
            
            info_data = {
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': [df[col].count() for col in df.columns],
                'Null Count': [df[col].isnull().sum() for col in df.columns],
                'Unique Values': [df[col].nunique() for col in df.columns]
            }
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True, height=450)
        
        with col2:
            st.markdown("#### 📖 Penjelasan Fitur")
            
            feature_desc = {
                'id': 'Identifikasi unik pasien',
                'gender': 'Jenis kelamin (Male/Female/Other)',
                'age': 'Usia pasien (tahun)',
                'hypertension': 'Riwayat hipertensi (0=Tidak, 1=Ya)',
                'heart_disease': 'Riwayat penyakit jantung (0=Tidak, 1=Ya)',
                'ever_married': 'Status pernikahan (Yes/No)',
                'work_type': 'Jenis pekerjaan',
                'Residence_type': 'Tipe tempat tinggal (Urban/Rural)',
                'avg_glucose_level': 'Rata-rata kadar glukosa (mg/dL)',
                'bmi': 'Body Mass Index',
                'smoking_status': 'Status merokok',
                'stroke': '🎯 Target: Stroke (0=Tidak, 1=Ya)'
            }
            
            for col, desc in feature_desc.items():
                with st.expander(f"**{col}**"):
                    st.write(desc)
                    st.write(f"Type: `{df[col].dtype}`")
                    st.write(f"Unique: {df[col].nunique()}")
                    if df[col].dtype == 'object':
                        st.write(f"Values: {df[col].unique()[:5]}")
    
    # ==================== TAB 3: STATISTIK ====================
    with tab3:
        st.markdown("### 📈 Statistik Deskriptif")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔢 Statistik Numerikal")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### 📝 Statistik Kategorikal")
            st.dataframe(df.describe(include=['object']), use_container_width=True)
        
        st.markdown("---")
        
        # Distribution by column
        st.markdown("#### 📊 Distribusi Nilai per Kolom")
        
        selected_col = st.selectbox("Pilih Kolom:", df.columns)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if df[selected_col].dtype == 'object':
                # Bar chart untuk kategorikal
                value_counts = df[selected_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                            labels={'x': selected_col, 'y': 'Count'},
                            title=f'Distribusi {selected_col}',
                            color=value_counts.values,
                            color_continuous_scale='Viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Histogram untuk numerikal
                fig = px.histogram(df, x=selected_col,
                                  title=f'Distribusi {selected_col}',
                                  nbins=30,
                                  color_discrete_sequence=['#636EFA'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 Value Counts")
            if df[selected_col].dtype == 'object':
                vc = df[selected_col].value_counts()
                vc_df = pd.DataFrame({
                    'Value': vc.index,
                    'Count': vc.values,
                    'Percentage': (vc.values / len(df) * 100).round(2)
                })
                st.dataframe(vc_df, use_container_width=True)
            else:
                st.write(f"**Mean:** {df[selected_col].mean():.2f}")
                st.write(f"**Median:** {df[selected_col].median():.2f}")
                st.write(f"**Std:** {df[selected_col].std():.2f}")
                st.write(f"**Min:** {df[selected_col].min():.2f}")
                st.write(f"**Max:** {df[selected_col].max():.2f}")
    
    # ==================== TAB 4: MISSING VALUES ====================
    with tab4:
        st.markdown("### ❌ Missing Values Analysis")
        
        # Calculate missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.warning(f"⚠️ Ditemukan {len(missing_df)} kolom dengan missing values")
                
                # Bar chart
                fig = px.bar(missing_df, x='Column', y='Missing Count',
                            title='Missing Values per Column',
                            color='Percentage',
                            color_continuous_scale='Reds',
                            text='Missing Count')
                fig.update_traces(textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Detail")
                st.dataframe(missing_df, use_container_width=True)
                
                st.info(f"""
                **Total Missing Values:** {missing_df['Missing Count'].sum()}  
                **Percentage:** {(missing_df['Missing Count'].sum() / (len(df) * df.shape[1]) * 100):.2f}%
                """)
        else:
            st.success("✅ Tidak ada missing values dalam dataset!")
        
        # Heatmap
        st.markdown("#### 🗺️ Missing Values Heatmap")
        missing_matrix = df.isnull().astype(int)
        fig = px.imshow(missing_matrix.T, 
                       labels=dict(x="Row Index", y="Column", color="Missing"),
                       aspect="auto",
                       color_continuous_scale=['#00CC96', '#EF553B'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 5: DOWNLOAD ====================
    with tab5:
        st.markdown("### 💾 Download Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📄 CSV Format")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name='stroke_dataset.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 📊 Excel Format")
            # Note: requires openpyxl
            try:
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Stroke Data')
                excel_data = buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name='stroke_dataset.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
            except:
                st.info("Install openpyxl untuk download Excel")
        
        with col3:
            st.markdown("#### 📋 JSON Format")
            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name='stroke_dataset.json',
                mime='application/json',
                use_container_width=True
            )
        
        st.markdown("---")
        st.info(f"**Dataset Size:** {len(df):,} rows × {df.shape[1]} columns")

# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    main()
