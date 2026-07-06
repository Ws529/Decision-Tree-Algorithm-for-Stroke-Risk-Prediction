#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk menambahkan TAHAP 3 - Data Preprocessing
"""

import json

# Load notebook yang sudah ada
with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Function helper
def add_markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    }

def add_code(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

# TAHAP 3 CELLS
tahap3_cells = []

# HEADER TAHAP 3
tahap3_cells.append(add_markdown("""---
# TAHAP 3 — DATA PREPROCESSING
---

Tahap preprocessing adalah tahap penting untuk mempersiapkan data sebelum digunakan untuk training model.

Langkah-langkah yang akan dilakukan:
1. Menghapus kolom ID
2. Menangani Missing Value (BMI)
3. Label Encoding untuk fitur kategorikal
4. Feature Scaling (Standardization)
5. Memisahkan Feature dan Target
6. Train-Test Split (80:20)"""))

# 1. COPY DATAFRAME
tahap3_cells.append(add_markdown("""## 1. Copy DataFrame

Membuat salinan DataFrame untuk menjaga data original tetap utuh."""))

tahap3_cells.append(add_code("""# Membuat copy dataframe
df_clean = df.copy()

print("=" * 70)
print("✓ DataFrame berhasil di-copy")
print(f"Shape: {df_clean.shape}")
print("=" * 70)"""))

# 2. MENGHAPUS KOLOM ID
tahap3_cells.append(add_markdown("""## 2. Menghapus Kolom ID

Kolom ID tidak memberikan informasi prediktif dan perlu dihapus."""))

tahap3_cells.append(add_code("""# Menghapus kolom id
df_clean = df_clean.drop('id', axis=1)

print("=" * 70)
print("✓ Kolom 'id' berhasil dihapus")
print(f"Shape setelah drop: {df_clean.shape}")
print(f"Kolom tersisa: {list(df_clean.columns)}")
print("=" * 70)"""))

# 3. MENANGANI MISSING VALUE BMI
tahap3_cells.append(add_markdown("""## 3. Menangani Missing Value (BMI)

BMI memiliki 201 missing values yang akan diisi menggunakan **median** karena lebih robust terhadap outlier."""))

tahap3_cells.append(add_code("""# Cek missing value sebelum
print("Missing Value SEBELUM Imputasi:")
print("=" * 70)
print(df_clean.isnull().sum())
print("=" * 70)

# Imputasi BMI dengan median
bmi_median = df_clean['bmi'].median()
df_clean['bmi'] = df_clean['bmi'].fillna(bmi_median)

# Cek missing value setelah
print("\\nMissing Value SETELAH Imputasi:")
print("=" * 70)
print(df_clean.isnull().sum())
print("=" * 70)
print(f"\\n✓ BMI missing values diisi dengan median: {bmi_median:.2f}")"""))

# 4. IDENTIFIKASI FITUR KATEGORIKAL
tahap3_cells.append(add_markdown("""## 4. Identifikasi Fitur Kategorikal

Mengidentifikasi kolom kategorikal yang perlu di-encode."""))

tahap3_cells.append(add_code("""# Identifikasi kolom kategorikal
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

print("=" * 70)
print("FITUR KATEGORIKAL YANG PERLU DI-ENCODE:")
print("=" * 70)
for i, col in enumerate(categorical_cols, 1):
    unique_values = df_clean[col].nunique()
    print(f"{i}. {col:20s} → {unique_values} unique values")
    print(f"   Values: {df_clean[col].unique()[:5]}")
print("=" * 70)"""))

# 5. LABEL ENCODING
tahap3_cells.append(add_markdown("""## 5. Label Encoding

Mengkonversi fitur kategorikal menjadi numerik menggunakan Label Encoder.

**Strategi Encoding:**
- Gender: Male=1, Female=0, Other=2
- ever_married: Yes=1, No=0
- work_type: Encoding numerik
- Residence_type: Urban=1, Rural=0
- smoking_status: Encoding numerik"""))

tahap3_cells.append(add_code("""from sklearn.preprocessing import LabelEncoder

# Dictionary untuk menyimpan encoder
encoders = {}

# Encoding setiap kolom kategorikal
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    encoders[col] = le
    
    print(f"✓ {col:20s} → Encoded")
    print(f"  Classes: {le.classes_}")
    print(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print()

print("=" * 70)
print("✓ SEMUA FITUR KATEGORIKAL BERHASIL DI-ENCODE")
print("=" * 70)"""))

# 6. VERIFIKASI SETELAH ENCODING
tahap3_cells.append(add_markdown("""## 6. Verifikasi Data Setelah Encoding

Memastikan semua data sudah dalam format numerik."""))

tahap3_cells.append(add_code("""# Cek tipe data setelah encoding
print("TIPE DATA SETELAH ENCODING:")
print("=" * 70)
print(df_clean.dtypes)
print("=" * 70)

# Preview data
print("\\nPREVIEW DATA SETELAH PREPROCESSING:")
print("=" * 70)
display(df_clean.head(10))
print("=" * 70)

# Statistik deskriptif
print("\\nSTATISTIK DESKRIPTIF SETELAH PREPROCESSING:")
print("=" * 70)
display(df_clean.describe())
print("=" * 70)"""))

# 7. MEMISAHKAN FEATURE DAN TARGET
tahap3_cells.append(add_markdown("""## 7. Memisahkan Feature dan Target

Memisahkan dataset menjadi:
- **X**: Feature variables (input)
- **y**: Target variable (output/stroke)"""))

tahap3_cells.append(add_code("""# Memisahkan features dan target
X = df_clean.drop('stroke', axis=1)
y = df_clean['stroke']

print("=" * 70)
print("✓ FEATURES DAN TARGET BERHASIL DIPISAHKAN")
print("=" * 70)
print(f"X (Features) shape : {X.shape}")
print(f"y (Target) shape   : {y.shape}")
print(f"\\nJumlah Features    : {X.shape[1]}")
print(f"Features           : {list(X.columns)}")
print(f"\\nTarget distribution:")
print(y.value_counts())
print("=" * 70)"""))

# 8. TRAIN TEST SPLIT
tahap3_cells.append(add_markdown("""## 8. Train-Test Split (80:20)

Membagi data menjadi:
- **80% Training set** - Untuk melatih model
- **20% Testing set** - Untuk evaluasi model

Menggunakan **stratify** untuk menjaga proporsi kelas target tetap sama di train dan test."""))

tahap3_cells.append(add_code("""from sklearn.model_selection import train_test_split

# Split data dengan stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("=" * 70)
print("✓ DATA BERHASIL DIBAGI MENJADI TRAIN DAN TEST SET")
print("=" * 70)
print(f"Total data        : {len(X):,}")
print(f"Training set      : {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set       : {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
print("\\nShape:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")

# Cek distribusi target di train dan test
print("\\n" + "=" * 70)
print("DISTRIBUSI TARGET:")
print("=" * 70)
print("\\nTraining Set:")
print(y_train.value_counts())
print(f"Proportion: {y_train.value_counts(normalize=True).values}")

print("\\nTesting Set:")
print(y_test.value_counts())
print(f"Proportion: {y_test.value_counts(normalize=True).values}")
print("=" * 70)"""))

# 9. FEATURE SCALING
tahap3_cells.append(add_markdown("""## 9. Feature Scaling (Standardization)

Melakukan standardisasi fitur menggunakan **StandardScaler** agar semua fitur memiliki:
- Mean (rata-rata) = 0
- Standard Deviation = 1

**Penting:** Scaler di-fit hanya pada training set untuk menghindari data leakage."""))

tahap3_cells.append(add_code("""from sklearn.preprocessing import StandardScaler

# Inisialisasi scaler
scaler = StandardScaler()

# Fit dan transform pada training set
X_train_scaled = scaler.fit_transform(X_train)

# Transform pada testing set (gunakan scaler yang sudah di-fit)
X_test_scaled = scaler.transform(X_test)

# Convert kembali ke DataFrame untuk kemudahan interpretasi
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("=" * 70)
print("✓ FEATURE SCALING BERHASIL")
print("=" * 70)
print("\\nStatistik SEBELUM Scaling (X_train):")
display(X_train.describe())

print("\\nStatistik SETELAH Scaling (X_train_scaled):")
display(X_train_scaled.describe())
print("=" * 70)"""))

# 10. PERBANDINGAN SEBELUM DAN SESUDAH SCALING
tahap3_cells.append(add_markdown("""## 10. Visualisasi: Perbandingan Sebelum dan Sesudah Scaling

Membandingkan distribusi fitur sebelum dan sesudah scaling."""))

tahap3_cells.append(add_code("""# Visualisasi perbandingan
features_to_plot = ['age', 'avg_glucose_level', 'bmi']

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Sebelum Scaling', 'Setelah Scaling')
)

# Sebelum scaling
for feature in features_to_plot:
    fig.add_trace(
        go.Box(y=X_train[feature], name=feature, showlegend=True),
        row=1, col=1
    )

# Setelah scaling
for feature in features_to_plot:
    fig.add_trace(
        go.Box(y=X_train_scaled[feature], name=feature, showlegend=False),
        row=1, col=2
    )

fig.update_layout(
    height=500,
    title_text='📊 Perbandingan Distribusi Fitur: Sebelum vs Sesudah Scaling',
    title_x=0.5,
    title_font_size=18
)

fig.show()

print("=" * 70)
print("✓ Scaling membuat semua fitur berada di skala yang sama")
print("  Ini membantu algoritma Decision Tree bekerja lebih optimal")
print("=" * 70)"""))

# 11. RINGKASAN PREPROCESSING
tahap3_cells.append(add_markdown("""---
## 📌 RINGKASAN PREPROCESSING (TAHAP 3)
---

### ✅ Langkah yang Telah Dilakukan:

1. **Copy DataFrame** ✓
   - Data original tetap terjaga

2. **Menghapus Kolom ID** ✓
   - Kolom tidak informatif dihapus
   - Dari 12 kolom → 11 kolom

3. **Handle Missing Value** ✓
   - BMI: 201 missing values diisi dengan median (28.1)
   - Total missing values: 0

4. **Label Encoding** ✓
   - gender: Male/Female/Other → 0/1/2
   - ever_married: Yes/No → 1/0
   - work_type: 5 kategori → 0-4
   - Residence_type: Urban/Rural → 1/0
   - smoking_status: 4 kategori → 0-3

5. **Pemisahan Feature & Target** ✓
   - X: 10 features
   - y: 1 target (stroke)

6. **Train-Test Split** ✓
   - Training: 4,088 samples (80%)
   - Testing: 1,022 samples (20%)
   - Stratified split untuk menjaga proporsi kelas

7. **Feature Scaling** ✓
   - StandardScaler untuk normalisasi
   - Mean = 0, Std = 1

### 📊 Dataset Final:

**Training Set:**
- X_train_scaled: (4088, 10)
- y_train: (4088,)

**Testing Set:**
- X_test_scaled: (1022, 10)
- y_test: (1022,)

### 🎯 Langkah Selanjutnya:

**TAHAP 4**: Training Model Decision Tree
- Inisialisasi model
- Hyperparameter tuning
- Training dengan berbagai konfigurasi
- Memilih model terbaik

---"""))

# Tambahkan cells ke notebook
notebook['cells'].extend(tahap3_cells)

# Save notebook
with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("✓ TAHAP 3 BERHASIL DITAMBAHKAN!")
print("=" * 70)
print(f"Total cells baru: {len(tahap3_cells)}")
print(f"Total cells notebook: {len(notebook['cells'])}")
print("=" * 70)
print("\n📋 Konten TAHAP 3:")
print("  1. Copy DataFrame")
print("  2. Menghapus Kolom ID")
print("  3. Handle Missing Value (BMI)")
print("  4. Identifikasi Fitur Kategorikal")
print("  5. Label Encoding")
print("  6. Verifikasi Data")
print("  7. Pemisahan Feature & Target")
print("  8. Train-Test Split (80:20)")
print("  9. Feature Scaling (StandardScaler)")
print("  10. Visualisasi Scaling")
print("  11. Ringkasan Preprocessing")
print("=" * 70)
print("\n🎯 Siap lanjut ke TAHAP 4 - Training Model!")
print("=" * 70)
