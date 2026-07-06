# Cara Upload Proyek ke Streamlit Cloud

Berikut langkah paling sederhana agar proyek ini bisa dipublikasikan ke Streamlit Cloud.

## 1. Siapkan repository di GitHub

Jika proyek belum di GitHub, lakukan langkah berikut:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Ws529/Decision-Tree-Algorithm-for-Stroke-Risk-Prediction.git
git push -u origin main
```

## 2. Pastikan file penting ada

Pastikan file berikut ada di root repository:

- app.py
- requirements.txt
- healthcare-dataset-stroke-data.csv
- decision_tree.pkl
- scaler.pkl
- encoder.pkl

> Jika file model belum ada, jalankan notebook training terlebih dahulu untuk menghasilkan file model tersebut.

## 3. Deploy ke Streamlit Cloud

1. Buka https://streamlit.io/cloud
2. Login menggunakan akun GitHub
3. Klik New app
4. Pilih repository https://github.com/Ws529/Decision-Tree-Algorithm-for-Stroke-Risk-Prediction.git
5. Pilih branch utama (biasanya main)
6. Tentukan main file path: app.py
7. Klik Deploy

## 4. Tunggu proses deploy

Streamlit Cloud akan:
- menginstal dependency dari requirements.txt
- menjalankan aplikasi dari app.py
- menampilkan URL aplikasi Anda

## 5. Jika ada masalah

### Masalah: dependency error
Periksa isi requirements.txt dan pastikan semua paket ada.

### Masalah: model tidak ditemukan
Pastikan file berikut ada di repository root:
- decision_tree.pkl
- scaler.pkl
- encoder.pkl

### Masalah: aplikasi tidak berjalan
Cek log error di Streamlit Cloud dan pastikan:
- file app.py benar
- repository terhubung dengan benar
- branch yang dipilih benar

## 6. Hasil akhir

Setelah berhasil, aplikasi Anda akan tersedia melalui URL seperti:

```text
https://your-app-name.streamlit.app
```
