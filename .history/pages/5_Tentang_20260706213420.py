import streamlit as st
import pandas as pd

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Tentang - Stroke Prediction",
    page_icon="ℹ️",
    layout="wide"
)

# =====================================================================
# HEADER
# =====================================================================
st.markdown('<p style="font-size:2.5rem; font-weight:bold; color:#1f77b4; text-align:center;">ℹ️ Tentang Proyek</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Informasi lengkap tentang sistem prediksi stroke</p>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# PROJECT OVERVIEW
# =====================================================================
st.markdown("## 📋 Deskripsi Proyek")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Stroke Prediction System using Decision Tree
    
    Sistem prediksi risiko stroke berbasis **Machine Learning** menggunakan algoritma **Decision Tree**. 
    Proyek ini bertujuan untuk membantu deteksi dini risiko stroke pada pasien berdasarkan data kesehatan 
    dan karakteristik pribadi mereka.
    
    **Latar Belakang:**
    
    Stroke adalah penyebab utama kematian dan kecacatan di seluruh dunia. Deteksi dini faktor risiko stroke 
    sangat penting untuk pencegahan dan intervensi medis yang tepat waktu. Dengan menggunakan machine learning,
    kita dapat mengidentifikasi pola dan faktor risiko yang mungkin tidak terlihat secara manual.
    
    **Tujuan Proyek:**
    
    1. Membangun model prediksi stroke yang akurat menggunakan Decision Tree
    2. Menganalisis faktor-faktor yang mempengaruhi risiko stroke
    3. Menyediakan tool interaktif untuk screening risiko stroke
    4. Memberikan visualisasi data yang informatif dan mudah dipahami
    """)

with col2:
    st.info("""
    ### 📊 Key Stats
    
    **Dataset:**
    - 5,110 data pasien
    - 11 fitur prediksi
    - 1 target variable
    
    **Model:**
    - Algoritma: Decision Tree
    - Accuracy: ~95%
    - AUC Score: 0.942
    
    **Tech Stack:**
    - Python 3.x
    - Scikit-learn
    - Streamlit
    - Plotly
    - Pandas & NumPy
    """)

st.markdown("---")

# =====================================================================
# ALGORITHM
# =====================================================================
st.markdown("## 🤖 Algoritma: Decision Tree")

tab1, tab2, tab3 = st.tabs(["Penjelasan", "Kelebihan & Kekurangan", "Cara Kerja"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Apa itu Decision Tree?
        
        **Decision Tree** adalah algoritma supervised learning yang digunakan untuk klasifikasi dan regresi.
        Algoritma ini membentuk struktur pohon keputusan, di mana:
        
        - **Root Node**: Node paling atas (akar)
        - **Internal Nodes**: Node keputusan
        - **Branches**: Hasil dari keputusan
        - **Leaf Nodes**: Hasil akhir (prediksi)
        
        ### Konsep Dasar:
        
        Decision Tree bekerja dengan cara **membagi dataset** menjadi subset yang lebih kecil berdasarkan 
        **fitur yang paling informatif**. Proses ini diulangi secara rekursif hingga mencapai kondisi berhenti 
        (misalnya: kedalaman maksimal atau jumlah sampel minimum).
        """)
    
    with col2:
        st.markdown("""
        ### Kriteria Split:
        
        **1. Gini Impurity**
        ```
        Gini = 1 - Σ(pi²)
        ```
        - Mengukur probabilitas salah klasifikasi
        - Semakin rendah = semakin baik
        
        **2. Entropy (Information Gain)**
        ```
        Entropy = -Σ(pi * log2(pi))
        ```
        - Mengukur ketidakpastian/disorder
        - Information Gain = penurunan entropy
        
        ### Hyperparameters:
        
        - `max_depth`: Kedalaman maksimal pohon
        - `min_samples_split`: Minimum sampel untuk split
        - `min_samples_leaf`: Minimum sampel di leaf
        - `criterion`: Fungsi split (gini/entropy)
        """)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        ### ✅ Kelebihan Decision Tree
        
        1. **Mudah Dipahami**
           - Visualisasi intuitif
           - Interpretable untuk non-technical user
        
        2. **Tidak Perlu Scaling**
           - Tidak terpengaruh skala fitur
           - Bekerja dengan data kategorikal
        
        3. **Handle Non-linear**
           - Dapat menangkap pola non-linear
           - Tidak ada asumsi distribusi data
        
        4. **Feature Importance**
           - Menunjukkan fitur mana yang penting
           - Membantu feature selection
        
        5. **Cepat**
           - Training dan prediksi cepat
           - Cocok untuk real-time prediction
        
        6. **Multi-output**
           - Bisa multi-class classification
           - Bisa untuk regression
        """)
    
    with col2:
        st.warning("""
        ### ⚠️ Kekurangan Decision Tree
        
        1. **Overfitting**
           - Mudah overfit jika terlalu dalam
           - Perlu pruning/regularization
        
        2. **Instability**
           - Sensitif terhadap perubahan data
           - Pohon bisa berbeda dengan data sedikit berbeda
        
        3. **Biased pada Imbalanced**
           - Cenderung bias ke kelas mayoritas
           - Perlu handling khusus (class_weight)
        
        4. **Global Optimum**
           - Greedy algorithm (lokal optimum)
           - Tidak garantee global optimum
        
        5. **Linear Relationships**
           - Kurang efektif untuk linear relationships
           - Butuh banyak split untuk linear pattern
        
        **Solusi:**
        - Random Forest (ensemble)
        - Gradient Boosting
        - Cross-validation
        - Hyperparameter tuning
        """)

with tab3:
    st.markdown("""
    ### 🔄 Cara Kerja Decision Tree
    
    #### Langkah-langkah Training:
    
    1. **Start dengan Root Node**
       - Semua data ada di root
       - Pilih fitur terbaik untuk split
    
    2. **Split Data**
       - Hitung Gini/Entropy untuk setiap fitur
       - Pilih fitur dengan Information Gain tertinggi
       - Bagi data menjadi subset
    
    3. **Recursive Splitting**
       - Ulangi proses untuk setiap subset
       - Buat internal nodes dan branches
    
    4. **Stopping Criteria**
       - Max depth tercapai
       - Min samples tercapai
       - Tidak ada Information Gain
       - Semua data di node sama class
    
    5. **Create Leaf Nodes**
       - Assign class label
       - Class mayoritas di node
    
    #### Langkah Prediksi:
    
    1. Start dari root
    2. Evaluasi kondisi di setiap node
    3. Ikuti branch sesuai kondisi
    4. Teruskan hingga leaf node
    5. Return class di leaf node
    
    #### Contoh Flow:
    
    ```
    Root: Age > 60?
    ├── No → BMI > 30?
    │   ├── No → No Stroke (Leaf)
    │   └── Yes → Hypertension?
    │       ├── No → No Stroke (Leaf)
    │       └── Yes → Stroke (Leaf)
    └── Yes → Heart Disease?
        ├── No → Glucose > 140?
        │   ├── No → No Stroke (Leaf)
        │   └── Yes → Stroke (Leaf)
        └── Yes → Stroke (Leaf)
    ```
    """)

st.markdown("---")

# =====================================================================
# DATASET
# =====================================================================
st.markdown("## 📊 Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Healthcare Stroke Prediction Dataset
    
    Dataset yang digunakan adalah **Healthcare Stroke Prediction Dataset** yang berisi informasi kesehatan 
    dari 5,110 pasien dengan berbagai karakteristik dan riwayat medis.
    
    **Sumber Data:**
    - Dataset publik untuk penelitian
    - Data telah di-anonimkan
    - Tidak mengandung informasi pribadi yang sensitif
    
    **Karakteristik Dataset:**
    - **Total Records**: 5,110 pasien
    - **Total Features**: 11 fitur + 1 target
    - **Missing Values**: BMI (201 missing)
    - **Imbalanced**: 95% tidak stroke, 5% stroke
    - **Data Types**: Numerikal dan kategorikal
    """)
    
    st.markdown("""
    ### Fitur-fitur Dataset:
    
    | No | Fitur | Tipe | Deskripsi |
    |----|-------|------|-----------|
    | 1 | gender | Kategorikal | Jenis kelamin (Male/Female/Other) |
    | 2 | age | Numerikal | Usia pasien (0-82 tahun) |
    | 3 | hypertension | Binary | Hipertensi (0=Tidak, 1=Ya) |
    | 4 | heart_disease | Binary | Penyakit jantung (0=Tidak, 1=Ya) |
    | 5 | ever_married | Binary | Status pernikahan (Yes/No) |
    | 6 | work_type | Kategorikal | Jenis pekerjaan |
    | 7 | Residence_type | Binary | Tempat tinggal (Urban/Rural) |
    | 8 | avg_glucose_level | Numerikal | Kadar glukosa (mg/dL) |
    | 9 | bmi | Numerikal | Body Mass Index |
    | 10 | smoking_status | Kategorikal | Status merokok |
    | 11 | **stroke** | **Binary** | **Target: Stroke (0/1)** |
    """)

with col2:
    st.info("""
    ### 📈 Statistik Dataset
    
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
    
    **Glucose:**
    - Rata-rata: 106.1 mg/dL
    - Range: 55-272 mg/dL
    
    **Health Conditions:**
    - Hipertensi: 9.7%
    - Penyakit Jantung: 5.4%
    """)

st.markdown("---")

# =====================================================================
# TOOLS & TECH
# =====================================================================
st.markdown("## 🛠️ Tools & Technologies")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🐍 Python Libraries
    
    **Data Processing:**
    - Pandas
    - NumPy
    
    **Machine Learning:**
    - Scikit-learn
    - Joblib
    
    **Visualization:**
    - Plotly
    - Matplotlib
    - Seaborn
    """)

with col2:
    st.markdown("""
    ### 💻 Development Tools
    
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
    ### 📦 Deployment
    
    **Platform:**
    - Streamlit Cloud
    - Heroku (optional)
    
    **Requirements:**
    - Python 3.8+
    - pip/conda
    
    **Model Files:**
    - .pkl format
    - Joblib serialization
    """)

st.markdown("---")

# =====================================================================
# TEAM
# =====================================================================
st.markdown("## 👨‍💻 Tim Pengembang")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>👤 Ferly Ardiansyah</h3>
        <p><strong>NIM:</strong> 312310448</p>
        <p><strong>Role:</strong> Data Scientist</p>
        <p>Fokus: Model Development & Evaluation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>👤 Bayu Aji Yuwono</h3>
        <p><strong>NIM:</strong> 312310492</p>
        <p><strong>Role:</strong> ML Engineer</p>
        <p>Fokus: Algorithm Implementation</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>👤 Wawan suwandi</h3>
        <p><strong>NIM:</strong> 312310457</p>
        <p><strong>Role:</strong> Full Stack Dev</p>
        <p>Fokus: UI/UX & Deployment</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# ACADEMIC INFO
# =====================================================================
st.markdown("## 🎓 Informasi Akademik")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 📚 Mata Kuliah
    
    **Nama:** Data Mining
    
    **Semester:** Genap 2023/2024
    
    **SKS:** 3 SKS
    
    **Dosen Pengampu:** [Nama Dosen]
    """)

with col2:
    st.success("""
    ### 🎯 Capaian Pembelajaran
    
    1. Memahami konsep data mining
    2. Mengimplementasikan algoritma ML
    3. Melakukan preprocessing data
    4. Evaluasi model dengan benar
    5. Deploy aplikasi ML
    6. Dokumentasi yang baik
    """)

st.markdown("---")

# =====================================================================
# REFERENCES
# =====================================================================
st.markdown("## 📖 Referensi")

st.markdown("""
### Literatur & Sumber:

1. **Scikit-learn Documentation**
   - https://scikit-learn.org/stable/modules/tree.html
   - Decision Tree Classifier official docs

2. **Streamlit Documentation**
   - https://docs.streamlit.io/
   - Web app framework for ML

3. **Plotly Documentation**
   - https://plotly.com/python/
   - Interactive visualization library

4. **Research Papers:**
   - "Stroke Prediction using Machine Learning Algorithms"
   - "Decision Trees in Healthcare Analytics"
   - "Handling Imbalanced Datasets in Medical Diagnosis"

5. **Books:**
   - "Hands-On Machine Learning" by Aurélien Géron
   - "Introduction to Data Mining" by Pang-Ning Tan
   - "Pattern Recognition and Machine Learning" by Christopher Bishop

6. **Online Courses:**
   - Coursera: Machine Learning by Andrew Ng
   - DataCamp: Decision Trees in Python
   - Kaggle: Machine Learning courses
""")

st.markdown("---")

# =====================================================================
# LICENSE & DISCLAIMER
# =====================================================================
st.markdown("## ⚖️ License & Disclaimer")

col1, col2 = st.columns(2)

with col1:
    st.warning("""
    ### ⚠️ Medical Disclaimer
    
    **PENTING:**
    
    Sistem prediksi ini dibuat untuk **tujuan edukasi dan penelitian**. 
    
    - **BUKAN** alat diagnosis medis profesional
    - **TIDAK MENGGANTIKAN** konsultasi dengan dokter
    - Hasil prediksi bersifat **screening awal**
    - Akurasi tidak 100% garantee
    
    **Selalu konsultasikan dengan tenaga medis profesional untuk diagnosis dan treatment.**
    """)

with col2:
    st.info("""
    ### 📜 License
    
    **MIT License**
    
    Copyright (c) 2024
    
    Permission is granted to use, copy, modify, and distribute this software for educational purposes.
    
    **Attribution Required:**
    - Cite this project if used in research
    - Give credit to the team
    - Link to original repository
    
    **No Warranty:**
    - Software provided "as is"
    - No liability for damages
    - Use at your own risk
    """)

st.markdown("---")

# =====================================================================
# CONTACT
# =====================================================================
st.markdown("## 📧 Kontak & Feedback")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 💬 Feedback
    
    Kirim feedback atau saran ke:
    - Email: [email@example.com]
    - Form: [Google Form Link]
    """)

with col2:
    st.markdown("""
    ### 🐛 Report Bug
    
    Laporkan bug atau issue:
    - GitHub Issues
    - Email ke tim
    """)

with col3:
    st.markdown("""
    ### 🤝 Contribute
    
    Ingin berkontribusi?
    - Fork repository
    - Submit pull request
    - Join discussion
    """)

st.markdown("---")

# =====================================================================
# ACKNOWLEDGMENTS
# =====================================================================
st.markdown("## 🙏 Acknowledgments")

st.success("""
Terima kasih kepada:

- **Dosen Pembimbing** atas guidance dan feedback
- **Scikit-learn Team** untuk library yang powerful
- **Streamlit Team** untuk framework yang amazing
- **Plotly Team** untuk visualization tools
- **Kaggle & UCI ML Repository** untuk dataset
- **Open Source Community** untuk resources dan tutorials
- **Keluarga & Teman** untuk dukungan moral

Special thanks to semua yang telah membantu dalam penyelesaian proyek ini! 🎉
""")

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 30px;'>
    <h3>🏥 Stroke Prediction System</h3>
    <p><strong>Powered by Decision Tree Algorithm</strong></p>
    <p>📚 Data Mining Project | 🎓 2024</p>
    <p>Developed with ❤️ using Python, Streamlit & Machine Learning</p>
    <br>
    <p style='font-size: 0.9rem;'>
        © 2024 Ferly Ardiansyah, Bayu Aji Yuwono, Wawan suwandi
    </p>
    <p style='font-size: 0.8rem; color: #999;'>
        For educational purposes only | Not for medical diagnosis
    </p>
</div>
""", unsafe_allow_html=True)
