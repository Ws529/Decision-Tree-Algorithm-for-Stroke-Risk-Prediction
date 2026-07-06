#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk membuat Jupyter Notebook lengkap TAHAP 1 dan TAHAP 2
"""

import json

# Template notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

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

# Mulai tambahkan cells
cells = []

# HEADER
cells.append(add_markdown("""# IMPLEMENTASI ALGORITMA DECISION TREE UNTUK PREDIKSI RISIKO STROKE

### Mata Kuliah
**Data Mining**

### Disusun Oleh
* Ferly Ardiansyah (312310448)
* Bayu Aji Yuwono (312310492)
* Wawan suwandi (312310457)

### Algoritma
Decision Tree

### Dataset
Healthcare Stroke Prediction Dataset

---"""))

# TAHAP 1 HEADER
cells.append(add_markdown("""---
# TAHAP 1 — ANALISIS DATASET
---"""))

# A. IMPORT LIBRARY
cells.append(add_markdown("""## A. Import Library"""))
cells.append(add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")
print("✓ Library berhasil diimport")"""))

# B. LOAD DATASET
cells.append(add_markdown("""## B. Load Dataset"""))
cells.append(add_code("""df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(f"Dataset shape: {df.shape}")
print(f"Total data: {df.shape[0]:,} baris, {df.shape[1]} kolom")"""))

# C. PREVIEW DATA
cells.append(add_markdown("""## C. Preview 10 Data Pertama"""))
cells.append(add_code("""display(df.head(10))"""))

# D. INFO DATASET
cells.append(add_markdown("""## D. Informasi Dataset"""))
cells.append(add_code("""print("=" * 70)
df.info()
print("=" * 70)"""))

# E. STATISTIK DESKRIPTIF
cells.append(add_markdown("""## E. Statistik Deskriptif"""))
cells.append(add_code("""print("Statistik Numerikal:")
display(df.describe())
print("\\nStatistik Kategorikal:")
display(df.describe(include=['object']))"""))

# F. MISSING VALUES
cells.append(add_markdown("""## F. Missing Values"""))
cells.append(add_code("""missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0]
if len(missing_df) > 0:
    print("Missing Values:")
    display(missing_df)
else:
    print("✓ Tidak ada missing values")"""))

# G. DISTRIBUSI TARGET
cells.append(add_markdown("""## G. Distribusi Target (Stroke)"""))
cells.append(add_code("""stroke_dist = df['stroke'].value_counts()
print(f"Tidak Stroke: {stroke_dist[0]:,} ({stroke_dist[0]/len(df)*100:.2f}%)")
print(f"Stroke      : {stroke_dist[1]:,} ({stroke_dist[1]/len(df)*100:.2f}%)")
print(f"Ratio       : {stroke_dist[0]/stroke_dist[1]:.1f}:1")"""))

# TAHAP 2 HEADER
cells.append(add_markdown("""---
# TAHAP 2 — EXPLORATORY DATA ANALYSIS (EDA)
---"""))

# 1. PIE CHART STROKE
cells.append(add_markdown("""## 1. Distribusi Stroke (Pie Chart)"""))
cells.append(add_code("""stroke_counts = df['stroke'].value_counts()
fig = go.Figure(data=[go.Pie(
    labels=['Tidak Stroke', 'Stroke'],
    values=stroke_counts.values,
    hole=0.4,
    marker=dict(colors=['#00CC96', '#EF553B'])
)])
fig.update_layout(title='🎯 Distribusi Stroke', height=500)
fig.show()"""))

# 2. GENDER DISTRIBUTION
cells.append(add_markdown("""## 2. Distribusi Gender"""))
cells.append(add_code("""gender_stroke = pd.crosstab(df['gender'], df['stroke'])
fig = go.Figure()
fig.add_trace(go.Bar(name='Tidak Stroke', x=gender_stroke.index, y=gender_stroke[0], marker_color='#00CC96'))
fig.add_trace(go.Bar(name='Stroke', x=gender_stroke.index, y=gender_stroke[1], marker_color='#EF553B'))
fig.update_layout(title='👥 Gender vs Stroke', barmode='group', height=500)
fig.show()"""))

# 3. HYPERTENSION
cells.append(add_markdown("""## 3. Distribusi Hypertension"""))
cells.append(add_code("""hyper_stroke = pd.crosstab(df['hypertension'], df['stroke'])
fig = go.Figure()
fig.add_trace(go.Bar(name='Tidak Stroke', x=['Tidak', 'Ya'], y=hyper_stroke[0].values, marker_color='#00CC96'))
fig.add_trace(go.Bar(name='Stroke', x=['Tidak', 'Ya'], y=hyper_stroke[1].values, marker_color='#EF553B'))
fig.update_layout(title='💊 Hypertension vs Stroke', barmode='group', height=500)
fig.show()"""))

# 4. HEART DISEASE
cells.append(add_markdown("""## 4. Distribusi Heart Disease"""))
cells.append(add_code("""heart_stroke = pd.crosstab(df['heart_disease'], df['stroke'])
fig = go.Figure()
fig.add_trace(go.Bar(name='Tidak Stroke', x=['Tidak', 'Ya'], y=heart_stroke[0].values, marker_color='#00CC96'))
fig.add_trace(go.Bar(name='Stroke', x=['Tidak', 'Ya'], y=heart_stroke[1].values, marker_color='#EF553B'))
fig.update_layout(title='❤️ Heart Disease vs Stroke', barmode='group', height=500)
fig.show()"""))

# 5. SMOKING STATUS
cells.append(add_markdown("""## 5. Distribusi Smoking Status"""))
cells.append(add_code("""smoking_stroke = pd.crosstab(df['smoking_status'], df['stroke'])
fig = go.Figure()
fig.add_trace(go.Bar(name='Tidak Stroke', x=smoking_stroke.index, y=smoking_stroke[0], marker_color='#00CC96'))
fig.add_trace(go.Bar(name='Stroke', x=smoking_stroke.index, y=smoking_stroke[1], marker_color='#EF553B'))
fig.update_layout(title='🚬 Smoking Status vs Stroke', barmode='group', height=500)
fig.show()"""))

# 6. HISTOGRAM AGE
cells.append(add_markdown("""## 6. Histogram Age"""))
cells.append(add_code("""fig = px.histogram(df, x='age', color='stroke',
                   title='📊 Distribusi Usia',
                   color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                   barmode='overlay', nbins=40, height=500, opacity=0.7)
fig.show()
print(f"Usia rata-rata: {df['age'].mean():.2f} tahun")"""))

# 7. HISTOGRAM BMI
cells.append(add_markdown("""## 7. Histogram BMI"""))
cells.append(add_code("""fig = px.histogram(df.dropna(subset=['bmi']), x='bmi', color='stroke',
                   title='⚖️ Distribusi BMI',
                   color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                   barmode='overlay', nbins=40, height=500, opacity=0.7)
fig.show()
print(f"BMI rata-rata: {df['bmi'].mean():.2f}")"""))

# 8. HISTOGRAM GLUCOSE
cells.append(add_markdown("""## 8. Histogram Glucose Level"""))
cells.append(add_code("""fig = px.histogram(df, x='avg_glucose_level', color='stroke',
                   title='🩸 Distribusi Glucose Level',
                   color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                   barmode='overlay', nbins=40, height=500, opacity=0.7)
fig.show()
print(f"Glucose rata-rata: {df['avg_glucose_level'].mean():.2f}")"""))

# 9. BOXPLOT BMI
cells.append(add_markdown("""## 9. Boxplot BMI"""))
cells.append(add_code("""fig = px.box(df, x='stroke', y='bmi',
             title='📦 Boxplot BMI vs Stroke',
             color='stroke', color_discrete_map={0: '#00CC96', 1: '#EF553B'},
             height=500)
fig.show()"""))

# 10. BOXPLOT AGE
cells.append(add_markdown("""## 10. Boxplot Age"""))
cells.append(add_code("""fig = px.box(df, x='stroke', y='age',
             title='📦 Boxplot Age vs Stroke',
             color='stroke', color_discrete_map={0: '#00CC96', 1: '#EF553B'},
             height=500)
fig.show()"""))

# 11. HEATMAP KORELASI
cells.append(add_markdown("""## 11. Heatmap Korelasi"""))
cells.append(add_code("""numeric_cols = df.select_dtypes(include=[np.number]).drop('id', axis=1)
correlation = numeric_cols.corr()
fig = px.imshow(correlation, text_auto='.2f',
                title='🔥 Heatmap Korelasi',
                color_continuous_scale='RdBu_r',
                height=600)
fig.show()
print("\\nKorelasi dengan Stroke:")
print(correlation['stroke'].sort_values(ascending=False))"""))

# 12. SCATTER AGE VS GLUCOSE
cells.append(add_markdown("""## 12. Scatter Plot Age vs Glucose"""))
cells.append(add_code("""fig = px.scatter(df, x='age', y='avg_glucose_level', color='stroke',
                 title='📈 Age vs Glucose Level',
                 color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                 opacity=0.6, height=600)
fig.show()"""))

# 13. SCATTER BMI VS GLUCOSE
cells.append(add_markdown("""## 13. Scatter Plot BMI vs Glucose"""))
cells.append(add_code("""fig = px.scatter(df.dropna(subset=['bmi']), x='bmi', y='avg_glucose_level', color='stroke',
                 title='📈 BMI vs Glucose Level',
                 color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                 opacity=0.6, height=600)
fig.show()"""))

# 14. FEATURE DISTRIBUTION OVERVIEW
cells.append(add_markdown("""## 14. Feature Distribution Overview"""))
cells.append(add_code("""fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('Age', 'BMI', 'Glucose', 'Stroke'))
fig.add_trace(go.Histogram(x=df['age'], marker_color='#636EFA'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['bmi'].dropna(), marker_color='#EF553B'), row=1, col=2)
fig.add_trace(go.Histogram(x=df['avg_glucose_level'], marker_color='#00CC96'), row=2, col=1)
fig.add_trace(go.Bar(x=['No', 'Yes'], y=df['stroke'].value_counts().values, marker_color=['#00CC96', '#EF553B']), row=2, col=2)
fig.update_layout(height=700, title_text='📊 Overview Distribusi Fitur', showlegend=False)
fig.show()"""))

# 15. PAIRPLOT
cells.append(add_markdown("""## 15. Scatter Matrix"""))
cells.append(add_code("""fig = px.scatter_matrix(df[['age', 'bmi', 'avg_glucose_level', 'stroke']].dropna(),
                        dimensions=['age', 'bmi', 'avg_glucose_level'],
                        color='stroke',
                        title='🔍 Scatter Matrix',
                        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                        height=800, opacity=0.6)
fig.show()"""))

# RINGKASAN TAHAP 2
cells.append(add_markdown("""---
## 📌 RINGKASAN TAHAP 2

### ✅ Temuan Penting:
1. Dataset sangat imbalanced (95:5)
2. Age berkorelasi tinggi dengan stroke
3. Hypertension & heart disease meningkatkan risiko stroke
4. BMI memiliki 201 missing values
5. Glucose level pada pasien stroke cenderung lebih tinggi

### 🎯 Langkah Selanjutnya:
**TAHAP 3**: Data Preprocessing
---"""))

# Assign cells ke notebook
notebook['cells'] = cells

# Save notebook
with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("✓ NOTEBOOK BERHASIL DIBUAT!")
print("=" * 70)
print(f"Total cells: {len(cells)}")
print(f"File: Implementasi_Decision_Tree_Stroke_Prediction.ipynb")
print("=" * 70)
print("\n📋 Konten:")
print("  ✓ TAHAP 1: Analisis Dataset (lengkap)")
print("  ✓ TAHAP 2: EDA dengan Plotly (15 visualisasi)")
print("\n🎯 Siap dijalankan di Jupyter Notebook!")
print("=" * 70)
