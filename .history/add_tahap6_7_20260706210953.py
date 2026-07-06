#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

def add_markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split('\n')}

def add_code(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split('\n')}

cells = []

# ========== TAHAP 6 ==========
cells.append(add_markdown("""---
# TAHAP 6 — VISUALISASI MODEL
---

Memvisualisasikan Decision Tree dan menganalisis feature importance."""))

# 1. DECISION TREE VIZ
cells.append(add_markdown("""## 1. Decision Tree Visualization"""))

cells.append(add_code("""from sklearn.tree import plot_tree

# Visualisasi Decision Tree
plt.figure(figsize=(25, 15))
plot_tree(final_model, 
          feature_names=X.columns,
          class_names=['No Stroke', 'Stroke'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)

plt.title('Decision Tree Visualization (Max Depth 3 untuk Clarity)', fontsize=20, pad=20)
plt.tight_layout()
plt.show()

print("=" * 70)
print("✓ Decision Tree berhasil divisualisasikan")
print(f"  Total nodes: {final_model.tree_.node_count}")
print(f"  Max depth: {final_model.get_depth()}")
print("=" * 70)"""))

# 2. FEATURE IMPORTANCE
cells.append(add_markdown("""## 2. Feature Importance

Menganalisis fitur mana yang paling berpengaruh dalam prediksi."""))

cells.append(add_code("""# Get feature importance
feature_importance = final_model.feature_importances_

# Buat DataFrame
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
display(importance_df)
print("=" * 70)"""))

# 3. FEATURE IMPORTANCE BAR
cells.append(add_markdown("""## 3. Feature Importance Visualization"""))

cells.append(add_code("""# Visualisasi Feature Importance
fig = px.bar(importance_df, 
             x='Importance', 
             y='Feature',
             orientation='h',
             title='📊 Feature Importance - Decision Tree',
             labels={'Importance': 'Importance Score', 'Feature': 'Features'},
             color='Importance',
             color_continuous_scale='Viridis',
             height=600)

fig.update_layout(
    title_x=0.5,
    title_font_size=20,
    yaxis={'categoryorder':'total ascending'}
)

fig.show()"""))

# 4. TOP FEATURES
cells.append(add_markdown("""## 4. Top 5 Most Important Features"""))

cells.append(add_code("""# Top 5 features
top5 = importance_df.head(5)

print("=" * 70)
print("TOP 5 MOST IMPORTANT FEATURES")
print("=" * 70)
for idx, row in top5.iterrows():
    print(f"{idx+1}. {row['Feature']:25s}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
print("=" * 70)

# Visualisasi pie chart
fig = px.pie(top5, 
             values='Importance', 
             names='Feature',
             title='🥧 Top 5 Features Contribution',
             hole=0.4,
             height=500)

fig.update_layout(title_x=0.5, title_font_size=20)
fig.show()"""))

# 5. RINGKASAN TAHAP 6
cells.append(add_markdown("""---
## 📌 RINGKASAN VISUALISASI MODEL (TAHAP 6)
---

### ✅ Visualisasi yang Dibuat:

1. **Decision Tree Diagram** ✓
   - Visualisasi struktur pohon keputusan
   - Menampilkan split criteria dan class distribution

2. **Feature Importance Table** ✓
   - Ranking semua fitur berdasarkan importance
   - Score importance untuk setiap fitur

3. **Feature Importance Bar Chart** ✓
   - Visualisasi horizontal bar chart
   - Mudah membandingkan importance antar fitur

4. **Top 5 Features** ✓
   - Focus pada 5 fitur paling penting
   - Pie chart untuk melihat kontribusi relatif

### 🎯 Insights:

Fitur yang paling berpengaruh dalam prediksi stroke dapat membantu:
- Prioritas pemeriksaan kesehatan
- Focus area untuk prevention
- Understanding risk factors

### 📊 Next Steps:

**TAHAP 7**: Menyimpan Model
---"""))

# ========== TAHAP 7 ==========
cells.append(add_markdown("""---
# TAHAP 7 — MENYIMPAN MODEL
---

Menyimpan model, scaler, dan encoder untuk deployment."""))

# 1. SAVE MODEL
cells.append(add_markdown("""## 1. Simpan Model Decision Tree"""))

cells.append(add_code("""import joblib

# Simpan model
joblib.dump(final_model, 'decision_tree.pkl')

print("=" * 70)
print("✓ Model Decision Tree berhasil disimpan")
print("  File: decision_tree.pkl")
print("=" * 70)"""))

# 2. SAVE SCALER
cells.append(add_markdown("""## 2. Simpan Scaler"""))

cells.append(add_code("""# Simpan scaler
joblib.dump(scaler, 'scaler.pkl')

print("=" * 70)
print("✓ Scaler berhasil disimpan")
print("  File: scaler.pkl")
print("=" * 70)"""))

# 3. SAVE ENCODERS
cells.append(add_markdown("""## 3. Simpan Encoders"""))

cells.append(add_code("""# Simpan encoder dictionary
joblib.dump(encoders, 'encoder.pkl')

print("=" * 70)
print("✓ Encoders berhasil disimpan")
print("  File: encoder.pkl")
print("=" * 70)"""))

# 4. VERIFY SAVED FILES
cells.append(add_markdown("""## 4. Verifikasi File yang Tersimpan"""))

cells.append(add_code("""import os

print("=" * 70)
print("VERIFIKASI FILE")
print("=" * 70)

files = ['decision_tree.pkl', 'scaler.pkl', 'encoder.pkl']

for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024  # Convert to KB
        print(f"✓ {file:25s} | Size: {size:.2f} KB")
    else:
        print(f"✗ {file:25s} | NOT FOUND")

print("=" * 70)"""))

# 5. TEST LOAD MODEL
cells.append(add_markdown("""## 5. Test Load Model"""))

cells.append(add_code("""# Load model kembali untuk test
loaded_model = joblib.load('decision_tree.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_encoders = joblib.load('encoder.pkl')

# Test prediction
sample_prediction = loaded_model.predict(X_test_scaled[:5])

print("=" * 70)
print("TEST LOAD MODEL")
print("=" * 70)
print("✓ Model berhasil di-load")
print("✓ Scaler berhasil di-load")
print("✓ Encoders berhasil di-load")
print("\\nTest Prediction (5 samples):")
print(f"Actual : {y_test.iloc[:5].values}")
print(f"Predict: {sample_prediction}")
print("=" * 70)"""))

# RINGKASAN TAHAP 7
cells.append(add_markdown("""---
## 📌 RINGKASAN MENYIMPAN MODEL (TAHAP 7)
---

### ✅ File yang Tersimpan:

1. **decision_tree.pkl** ✓
   - Model Decision Tree yang telah dilatih
   - Siap untuk deployment

2. **scaler.pkl** ✓
   - StandardScaler untuk preprocessing
   - Penting untuk transform data baru

3. **encoder.pkl** ✓
   - Dictionary of LabelEncoders
   - Untuk encode fitur kategorikal

### 📁 Struktur File:

```
Stroke_prediction/
├── healthcare-dataset-stroke-data.csv
├── Implementasi_Decision_Tree_Stroke_Prediction.ipynb
├── decision_tree.pkl
├── scaler.pkl
└── encoder.pkl
```

### 🚀 Cara Menggunakan Model:

```python
# Load model
model = joblib.load('decision_tree.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoder.pkl')

# Prepare new data
# 1. Encode categorical features
# 2. Scale numerical features
# 3. Predict
prediction = model.predict(new_data_scaled)
```

### 🎯 Next Steps:

**TAHAP 8**: Membuat Aplikasi Streamlit
- User Interface untuk prediksi
- Dashboard analytics
- Model deployment

---"""))

# ========== KESIMPULAN AKHIR ==========
cells.append(add_markdown("""---
# 🎉 KESIMPULAN PROYEK
---

## ✅ Tahapan yang Telah Diselesaikan:

### **TAHAP 1: Analisis Dataset** ✓
- Load dan eksplorasi data
- Statistik deskriptif
- Missing value analysis
- Target distribution analysis

### **TAHAP 2: EDA (Exploratory Data Analysis)** ✓
- 15 visualisasi interaktif dengan Plotly
- Distribusi setiap fitur
- Korelasi antar variabel
- Pattern recognition

### **TAHAP 3: Data Preprocessing** ✓
- Handle missing values
- Label encoding
- Feature scaling
- Train-test split

### **TAHAP 4: Training Model** ✓
- Baseline model
- Hyperparameter tuning
- Grid Search CV
- Model selection

### **TAHAP 5: Evaluasi Model** ✓
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- ROC Curve & AUC Score

### **TAHAP 6: Visualisasi Model** ✓
- Decision Tree visualization
- Feature importance analysis
- Top features ranking

### **TAHAP 7: Menyimpan Model** ✓
- Save model, scaler, encoder
- Model siap deployment

---

## 🎯 Hasil Akhir:

✅ Model Decision Tree berhasil dibangun dan dievaluasi
✅ Performa model memenuhi/mendekati target praktikum
✅ Model telah disimpan dan siap untuk deployment
✅ Dokumentasi lengkap dan terstruktur

---

## 📊 Performance Summary:

**Model terpilih telah dievaluasi dengan:**
- Multiple evaluation metrics
- Visualisasi komprehensif
- Feature importance analysis

**Model siap untuk:**
- Deployment ke production
- Integrasi dengan aplikasi Streamlit
- Prediksi risiko stroke pada data baru

---

## 🚀 TAHAP 8 - Streamlit Application

**Siap untuk membuat aplikasi Streamlit dengan fitur:**
- Homepage dengan metrics
- Dataset viewer
- EDA interactive
- Model performance
- Stroke prediction
- About page

---

### 👨‍💻 Developed by:
- Ferly Ardiansyah (312310448)
- Bayu Aji Yuwono (312310492)
- Wawan suwandi (312310457)

### 📅 Date: 2024
### 🎓 Course: Data Mining

---"""))

# Tambahkan ke notebook
notebook['cells'].extend(cells)

with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("✓ TAHAP 6 & 7 BERHASIL DITAMBAHKAN!")
print("=" * 70)
print(f"Total cells baru: {len(cells)}")
print(f"Total cells notebook: {len(notebook['cells'])}")
print("=" * 70)
print("\n📋 Notebook LENGKAP!")
print("  TAHAP 1: Analisis Dataset ✓")
print("  TAHAP 2: EDA ✓")
print("  TAHAP 3: Preprocessing ✓")
print("  TAHAP 4: Training Model ✓")
print("  TAHAP 5: Evaluasi Model ✓")
print("  TAHAP 6: Visualisasi Model ✓")
print("  TAHAP 7: Menyimpan Model ✓")
print("=" * 70)
print("\n🎯 Siap untuk TAHAP 8 - Membuat Streamlit!")
print("=" * 70)
