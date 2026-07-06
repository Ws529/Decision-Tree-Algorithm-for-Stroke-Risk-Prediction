# 🚀 Cara Menjalankan Aplikasi Stroke Prediction

## 📋 Prerequisites

Pastikan Anda sudah menginstall:
- Python 3.8 atau lebih baru
- pip (Python package manager)
- Git (opsional, untuk clone repository)

## 🔧 Langkah-langkah Instalasi

### 1️⃣ Pastikan Berada di Folder yang Benar

```bash
cd Stroke_prediction
```

### 2️⃣ Install Dependencies

Jalankan command berikut untuk menginstall semua library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Atau jika menggunakan Python 3:

```bash
pip3 install -r requirements.txt
```

**Dependencies yang akan diinstall:**
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- joblib
- matplotlib
- seaborn

### 3️⃣ Training Model (WAJIB Dilakukan Pertama Kali)

Sebelum menjalankan aplikasi Streamlit, Anda **HARUS** melakukan training model terlebih dahulu.

#### Opsi A: Menggunakan Jupyter Notebook (Recommended)

```bash
jupyter notebook
```

Kemudian:
1. Buka file `Implementasi_Decision_Tree_Stroke_Prediction.ipynb`
2. Klik **Run All** atau jalankan cell satu per satu
3. Pastikan semua tahapan berjalan tanpa error
4. Model akan tersimpan otomatis sebagai:
   - `decision_tree.pkl`
   - `scaler.pkl`
   - `encoder.pkl`

#### Opsi B: Menggunakan VS Code

1. Install extension **Jupyter** di VS Code
2. Buka file `.ipynb`
3. Jalankan semua cell

#### Opsi C: Google Colab

1. Upload file `.ipynb` ke Google Colab
2. Upload dataset `healthcare-dataset-stroke-data.csv`
3. Jalankan semua cell
4. Download file model (.pkl) yang dihasilkan
5. Letakkan di folder `Stroke_prediction/`

### 4️⃣ Verifikasi File Model

Pastikan file-file berikut sudah ada di folder `Stroke_prediction/`:

```
✅ decision_tree.pkl
✅ scaler.pkl
✅ encoder.pkl
✅ healthcare-dataset-stroke-data.csv
```

Jika belum ada, ulangi Langkah 3.

### 5️⃣ Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Atau:

```bash
python -m streamlit run app.py
```

### 6️⃣ Akses Aplikasi

Setelah command dijalankan, aplikasi akan terbuka di browser:

```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Jika browser tidak terbuka otomatis, copy URL tersebut dan paste di browser.

## 🎯 Navigasi Aplikasi

Aplikasi memiliki 6 halaman:

### 🏠 **Home**
- Overview sistem
- Quick statistics
- Informasi stroke
- Fitur utama

### 📁 **Dataset**
- Preview data
- Upload dataset
- Statistik deskriptif
- Download data

### 📈 **EDA** (Exploratory Data Analysis)
- 15+ visualisasi interaktif
- Distribusi fitur
- Korelasi
- Scatter plots

### 🤖 **Model**
- Performance metrics
- Confusion matrix
- ROC curve
- Feature importance
- Decision tree visualization

### 🔮 **Prediksi**
- Form input data pasien
- Prediksi risiko stroke
- Probabilitas
- Risk factor analysis
- Download hasil

### ℹ️ **Tentang**
- Informasi proyek
- Algoritma Decision Tree
- Dataset info
- Tim pengembang

## 🛑 Troubleshooting

### Problem 1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solusi:**
```bash
pip install -r requirements.txt
```

### Problem 2: Model tidak ditemukan

**Error:**
```
❌ Model belum tersedia!
```

**Solusi:**
- Jalankan notebook Jupyter terlebih dahulu (Langkah 3)
- Pastikan file .pkl tersimpan di folder yang sama dengan app.py

### Problem 3: Port sudah digunakan

**Error:**
```
Port 8501 is already in use
```

**Solusi:**
```bash
streamlit run app.py --server.port 8502
```

### Problem 4: Dataset tidak ditemukan

**Error:**
```
FileNotFoundError: healthcare-dataset-stroke-data.csv
```

**Solusi:**
- Pastikan file CSV ada di folder yang sama dengan app.py
- Check nama file (case-sensitive)

### Problem 5: Python version

**Error:**
```
Python 3.8+ required
```

**Solusi:**
```bash
python --version  # Check versi Python
```

Upgrade Python jika versi < 3.8

## 📦 Deploy ke Streamlit Cloud

### 1. Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/stroke-prediction.git
git push -u origin main
```

### 2. Deploy di Streamlit Cloud

1. Buka https://streamlit.io/cloud
2. Sign in dengan GitHub
3. Klik **New app**
4. Pilih repository: `stroke-prediction`
5. Main file: `app.py`
6. Klik **Deploy**

### 3. Konfigurasi

Pastikan file `requirements.txt` ada dan lengkap.

## 🔧 Konfigurasi Lanjutan

### Mengubah Port

Edit file `.streamlit/config.toml`:

```toml
[server]
port = 8080  # Ganti dengan port yang diinginkan
```

### Mengubah Theme

Edit file `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"  # Merah
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

## 📞 Bantuan

Jika mengalami kesulitan:

1. Check dokumentasi: `README.md`
2. Check error message di terminal
3. Pastikan semua dependencies terinstall
4. Pastikan Python version >= 3.8
5. Restart terminal/IDE

## ✅ Checklist Sebelum Menjalankan

- [ ] Python 3.8+ terinstall
- [ ] Dependencies terinstall (`pip install -r requirements.txt`)
- [ ] Notebook Jupyter sudah dijalankan (TAHAP 1-7)
- [ ] File model (.pkl) sudah tersimpan
- [ ] Dataset CSV tersedia
- [ ] Terminal/command prompt terbuka di folder yang benar

## 🎉 Selamat!

Jika semua langkah diikuti dengan benar, aplikasi Stroke Prediction seharusnya sudah berjalan dengan baik!

Selamat menggunakan! 🏥
