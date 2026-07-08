import streamlit as st

st.set_page_config(
    page_title="Tentang - Prediksi Stroke",
    page_icon="",
    layout="wide"
)

st.title("Tentang Proyek")
st.write("Informasi lengkap tentang sistem prediksi stroke")
st.markdown("---")

st.subheader("Deskripsi Proyek")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Sistem Prediksi Stroke Menggunakan Decision Tree**
    
    Sistem prediksi risiko stroke berbasis Machine Learning menggunakan algoritma Decision Tree. 
    Proyek ini bertujuan membantu deteksi dini risiko stroke pada pasien berdasarkan data kesehatan 
    dan karakteristik pribadi mereka.
    
    **Latar Belakang:**
    
    Stroke adalah penyebab utama kematian dan disabilitas di seluruh dunia. Deteksi dini faktor risiko stroke 
    sangat penting untuk pencegahan dan intervensi medis yang tepat waktu. Dengan menggunakan machine learning, 
    kita dapat mengidentifikasi pola dan faktor risiko yang mungkin tidak terlihat secara manual.
    
    **Tujuan Proyek:**
    
    1. Membangun model prediksi stroke yang akurat menggunakan Decision Tree
    2. Menganalisis faktor-faktor yang mempengaruhi risiko stroke
    3. Menyediakan alat interaktif untuk skrining risiko stroke
    4. Memberikan visualisasi data yang informatif dan mudah dipahami
    """)

with col2:
    st.info("""
    **Statistik Utama**
    
    **Dataset:**
    - 5,110 rekaman pasien
    - 11 fitur prediksi
    - 1 variabel target
    
    **Model:**
    - Algoritma: Decision Tree
    - Akurasi: sekitar 95%
    - Skor AUC: 0.942
    
    **Teknologi:**
    - Python 3.x
    - Scikit-learn
    - Streamlit
    - Plotly
    - Pandas & NumPy
    """)

st.markdown("---")

st.subheader("Algoritma: Decision Tree")

tab1, tab2 = st.tabs(["Penjelasan", "Kelebihan & Kekurangan"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Apa itu Decision Tree?**
        
        Decision Tree adalah algoritma supervised learning yang digunakan untuk klasifikasi dan regresi.
        Algoritma ini membuat struktur keputusan seperti pohon, dengan:
        
        - **Root Node**: Node paling atas (akar)
        - **Internal Nodes**: Node keputusan
        - **Branches**: Hasil dari keputusan
        - **Leaf Nodes**: Hasil akhir (prediksi)
        
        **Konsep Dasar:**
        
        Decision Tree bekerja dengan memisahkan dataset menjadi subset yang lebih kecil berdasarkan 
        fitur yang paling informatif. Proses ini diulang secara rekursif hingga kondisi berhenti 
        tercapai (misalnya kedalaman maksimum atau sampel minimum).
        """)
    
    with col2:
        st.markdown("""
        **Kriteria Split:**
        
        **1. Gini Impurity**
        ```
        Gini = 1 - Sum(pi^2)
        ```
        - Mengukur probabilitas kesalahan klasifikasi
        - Semakin rendah semakin baik
        
        **2. Entropy (Information Gain)**
        ```
        Entropy = -Sum(pi * log2(pi))
        ```
        - Mengukur ketidakpastian/disorder
        - Information Gain = pengurangan entropy
        
        **Hyperparameter:**
        
        - max_depth: Kedalaman maksimum pohon
        - min_samples_split: Sampel minimum untuk split
        - min_samples_leaf: Sampel minimum di leaf
        - criterion: Fungsi split (gini/entropy)
        """)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Kelebihan Decision Tree**
        
        1. **Mudah Dipahami**
           - Visualisasi intuitif
           - Dapat diinterpretasi oleh non-teknis
        
        2. **Tidak Perlu Scaling**
           - Tidak terpengaruh oleh scaling fitur
           - Bekerja dengan data kategorikal
        
        3. **Menangani Non-linear**
           - Dapat menangkap pola non-linear
           - Tidak ada asumsi distribusi
        
        4. **Feature Importance**
           - Menunjukkan fitur mana yang penting
           - Membantu seleksi fitur
        
        5. **Cepat**
           - Training dan prediksi cepat
           - Cocok untuk prediksi real-time
        """)
    
    with col2:
        st.warning("""
        **Kekurangan Decision Tree**
        
        1. **Overfitting**
           - Mudah overfit jika terlalu dalam
           - Perlu pruning/regularisasi
        
        2. **Instabilitas**
           - Sensitif terhadap perubahan data
           - Pohon bisa berbeda dengan data yang sedikit berbeda
        
        3. **Bias pada Imbalanced**
           - Cenderung bias ke kelas mayoritas
           - Perlu penanganan khusus (class_weight)
        
        4. **Optimum Global**
           - Algoritma greedy (optimum lokal)
           - Tidak menjamin optimum global
        
        **Solusi:**
        - Random Forest (ensemble)
        - Gradient Boosting
        - Cross-validation
        - Tuning hyperparameter
        """)

st.markdown("---")

st.subheader("Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Healthcare Stroke Prediction Dataset**
    
    Dataset yang digunakan adalah Healthcare Stroke Prediction Dataset yang berisi informasi kesehatan 
    dari 5,110 pasien dengan berbagai karakteristik dan riwayat medis.
    
    **Sumber Data:**
    - Dataset publik untuk penelitian
    - Data telah dianonimkan
    - Tidak mengandung informasi pribadi sensitif
    
    **Karakteristik Dataset:**
    - **Total Rekaman**: 5,110 pasien
    - **Total Fitur**: 11 fitur + 1 target
    - **Missing Values**: BMI (201 missing)
    - **Imbalanced**: 95% tidak stroke, 5% stroke
    - **Tipe Data**: Numerikal dan kategorikal
    """)
    
    st.markdown("""
    **Fitur Dataset:**
    
    | No | Fitur | Tipe | Deskripsi |
    |---|---------|------|-------------|
    | 1 | gender | Kategorikal | Jenis kelamin (Male/Female/Other) |
    | 2 | age | Numerikal | Usia pasien (0-82 tahun) |
    | 3 | hypertension | Binary | Hipertensi (0=Tidak, 1=Ya) |
    | 4 | heart_disease | Binary | Penyakit jantung (0=Tidak, 1=Ya) |
    | 5 | ever_married | Binary | Status pernikahan (Yes/No) |
    | 6 | work_type | Kategorikal | Jenis pekerjaan |
    | 7 | Residence_type | Binary | Tipe tempat tinggal (Urban/Rural) |
    | 8 | avg_glucose_level | Numerikal | Kadar glukosa (mg/dL) |
    | 9 | bmi | Numerikal | Body Mass Index |
    | 10 | smoking_status | Kategorikal | Status merokok |
    | 11 | **stroke** | **Binary** | **Target: Stroke (0/1)** |
    """)

with col2:
    st.info("""
    **Statistik Dataset**
    
    **Distribusi Target:**
    - Tidak Stroke: 4,861 (95.1%)
    - Stroke: 249 (4.9%)
    
    **Usia:**
    - Rata-rata: 43.2 tahun
    - Min: 0.08 tahun
    - Max: 82 tahun
    
    **BMI:**
    - Rata-rata: 28.9
    - Missing: 201 (3.9%)
    
    **Glukosa:**
    - Rata-rata: 106.1 mg/dL
    - Range: 55-272 mg/dL
    
    **Kondisi Kesehatan:**
    - Hipertensi: 9.7%
    - Penyakit Jantung: 5.4%
    """)

st.markdown("---")

st.subheader("Tools & Teknologi")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Library Python**
    
    **Data Processing:**
    - Pandas
    - NumPy
    
    **Machine Learning:**
    - Scikit-learn
    - Joblib
    
    **Visualisasi:**
    - Plotly
    - Matplotlib
    - Seaborn
    """)

with col2:
    st.markdown("""
    **Development Tools**
    
    **IDE & Notebook:**
    - Jupyter Notebook
    - VS Code
    
    **Web Framework:**
    - Streamlit
    
    **Version Control:**
    - Git
    - GitHub
    """)

with col3:
    st.markdown("""
    **Deployment**
    
    **Platform:**
    - Streamlit Cloud
    - Heroku (opsional)
    
    **Requirements:**
    - Python 3.8+
    - pip/conda
    
    **File Model:**
    - Format .pkl
    - Serialisasi Joblib
    """)

st.markdown("---")

st.subheader("Tim Pengembang")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>Ferly Ardiansyah</h3>
        <p><strong>ID:</strong> 312310448</p>
        <p>Pengembangan & Evaluasi Model</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>Bayu Aji Yuwono</h3>
        <p><strong>ID:</strong> 312310492</p>
        <p>Implementasi Algoritma</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>Wawan suwandi</h3>
        <p><strong>ID:</strong> 312310457</p>
        <p>UI/UX & Deployment</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("Lisensi & Disclaimer")

col1, col2 = st.columns(2)

with col1:
    st.warning("""
    **Disclaimer Medis**
    
    **PENTING:**
    
    Sistem prediksi ini dibuat untuk tujuan edukasi dan penelitian saja.
    
    - BUKAN alat diagnostik medis profesional
    - TIDAK MENGGANTIKAN konsultasi dengan dokter
    - Hasil prediksi untuk skrining awal saja
    - Akurasi tidak dijamin 100%
    
    Selalu konsultasikan dengan tenaga kesehatan profesional untuk diagnosis dan pengobatan.
    """)

with col2:
    st.info("""
    **Lisensi**
    
    **MIT License**
    
    Copyright (c) 2026
    
    Izin diberikan untuk menggunakan, menyalin, memodifikasi, dan mendistribusikan software ini untuk tujuan edukasi.
    
    **Atribusi Diperlukan:**
    - Kutip proyek ini jika digunakan dalam penelitian
    - Berikan kredit kepada tim
    - Link ke repository asli
    
    **Tanpa Garansi:**
    - Software disediakan "sebagaimana adanya"
    - Tidak ada tanggung jawab atas kerusakan
    - Gunakan dengan risiko sendiri
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Sistem Prediksi Stroke</strong></p>
    <p>Proyek Data Mining | 2026</p>
    <p style='margin-top: 10px;'>
        Ferly Ardiansyah, Bayu Aji Yuwono, Wawan suwandi
    </p>
</div>
""", unsafe_allow_html=True)
