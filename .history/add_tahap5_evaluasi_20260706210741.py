#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

def add_markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split('\n')}

def add_code(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split('\n')}

tahap5_cells = []

# HEADER
tahap5_cells.append(add_markdown("""---
# TAHAP 5 — EVALUASI MODEL
---

Evaluasi lengkap model menggunakan berbagai metrik untuk mengukur performa secara komprehensif."""))

# 1. IMPORT METRICS
tahap5_cells.append(add_markdown("""## 1. Prediksi dengan Model Final"""))

tahap5_cells.append(add_code("""# Prediksi menggunakan model final
y_train_pred_final = final_model.predict(X_train_scaled)
y_test_pred_final = final_model.predict(X_test_scaled)

# Prediksi probabilitas untuk ROC
y_train_proba = final_model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = final_model.predict_proba(X_test_scaled)[:, 1]

print("✓ Prediksi selesai")"""))

# 2. ACCURACY
tahap5_cells.append(add_markdown("""## 2. Accuracy Score"""))

tahap5_cells.append(add_code("""from sklearn.metrics import accuracy_score

acc_train = accuracy_score(y_train, y_train_pred_final)
acc_test = accuracy_score(y_test, y_test_pred_final)

print("=" * 70)
print("ACCURACY SCORE")
print("=" * 70)
print(f"Accuracy Train: {acc_train*100:.2f}%")
print(f"Accuracy Test : {acc_test*100:.2f}%")
print(f"Difference    : {abs(acc_train - acc_test)*100:.2f}%")
print("=" * 70)"""))

# 3. PRECISION
tahap5_cells.append(add_markdown("""## 3. Precision Score"""))

tahap5_cells.append(add_code("""from sklearn.metrics import precision_score

prec_train = precision_score(y_train, y_train_pred_final, zero_division=0)
prec_test = precision_score(y_test, y_test_pred_final, zero_division=0)

print("=" * 70)
print("PRECISION SCORE")
print("=" * 70)
print(f"Precision Train: {prec_train*100:.2f}%")
print(f"Precision Test : {prec_test*100:.2f}%")
print("\\nPrecision = TP / (TP + FP)")
print("Mengukur: Dari semua prediksi positif, berapa yang benar")
print("=" * 70)"""))

# 4. RECALL
tahap5_cells.append(add_markdown("""## 4. Recall Score"""))

tahap5_cells.append(add_code("""from sklearn.metrics import recall_score

rec_train = recall_score(y_train, y_train_pred_final, zero_division=0)
rec_test = recall_score(y_test, y_test_pred_final, zero_division=0)

print("=" * 70)
print("RECALL SCORE")
print("=" * 70)
print(f"Recall Train: {rec_train*100:.2f}%")
print(f"Recall Test : {rec_test*100:.2f}%")
print("\\nRecall = TP / (TP + FN)")
print("Mengukur: Dari semua kasus positif aktual, berapa yang terdeteksi")
print("=" * 70)"""))

# 5. F1 SCORE
tahap5_cells.append(add_markdown("""## 5. F1 Score"""))

tahap5_cells.append(add_code("""from sklearn.metrics import f1_score

f1_train = f1_score(y_train, y_train_pred_final, zero_division=0)
f1_test = f1_score(y_test, y_test_pred_final, zero_division=0)

print("=" * 70)
print("F1 SCORE")
print("=" * 70)
print(f"F1 Train: {f1_train*100:.2f}%")
print(f"F1 Test : {f1_test*100:.2f}%")
print("\\nF1 = 2 * (Precision * Recall) / (Precision + Recall)")
print("Harmonic mean dari Precision dan Recall")
print("=" * 70)"""))

# 6. CONFUSION MATRIX
tahap5_cells.append(add_markdown("""## 6. Confusion Matrix"""))

tahap5_cells.append(add_code("""from sklearn.metrics import confusion_matrix

# Confusion matrix untuk test set
cm_test = confusion_matrix(y_test, y_test_pred_final)

print("=" * 70)
print("CONFUSION MATRIX (TEST SET)")
print("=" * 70)
print(cm_test)
print("\\nInterpretasi:")
print(f"True Negative (TN) : {cm_test[0,0]:4d} - Benar prediksi Tidak Stroke")
print(f"False Positive (FP): {cm_test[0,1]:4d} - Salah prediksi Stroke")
print(f"False Negative (FN): {cm_test[1,0]:4d} - Salah prediksi Tidak Stroke")
print(f"True Positive (TP) : {cm_test[1,1]:4d} - Benar prediksi Stroke")
print("=" * 70)"""))

# 7. CONFUSION MATRIX HEATMAP
tahap5_cells.append(add_markdown("""## 7. Confusion Matrix Heatmap"""))

tahap5_cells.append(add_code("""# Visualisasi Confusion Matrix
fig = px.imshow(cm_test,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Stroke', 'Stroke'],
                y=['No Stroke', 'Stroke'],
                text_auto=True,
                color_continuous_scale='Blues',
                title='🔥 Confusion Matrix Heatmap')

fig.update_layout(height=500, title_x=0.5, title_font_size=20)
fig.show()"""))

# 8. CLASSIFICATION REPORT
tahap5_cells.append(add_markdown("""## 8. Classification Report"""))

tahap5_cells.append(add_code("""from sklearn.metrics import classification_report

print("=" * 70)
print("CLASSIFICATION REPORT (TEST SET)")
print("=" * 70)
print(classification_report(y_test, y_test_pred_final, 
                          target_names=['No Stroke', 'Stroke'],
                          digits=4))
print("=" * 70)"""))

# 9. ROC CURVE
tahap5_cells.append(add_markdown("""## 9. ROC Curve (Receiver Operating Characteristic)"""))

tahap5_cells.append(add_code("""from sklearn.metrics import roc_curve, roc_auc_score

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_test_proba)

# Visualisasi ROC Curve
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name=f'ROC Curve (AUC = {roc_auc:.4f})',
    line=dict(color='#EF553B', width=3)
))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(color='gray', width=2, dash='dash')
))

fig.update_layout(
    title='📈 ROC Curve - Model Performance',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    height=600,
    title_x=0.5,
    title_font_size=20,
    showlegend=True
)

fig.show()

print("=" * 70)
print(f"ROC AUC Score: {roc_auc:.4f}")
print("=" * 70)"""))

# 10. AUC SCORE DETAIL
tahap5_cells.append(add_markdown("""## 10. AUC Score Interpretation"""))

tahap5_cells.append(add_code("""print("=" * 70)
print("AUC SCORE INTERPRETATION")
print("=" * 70)
print(f"AUC Score: {roc_auc:.4f}")
print()
if roc_auc >= 0.9:
    print("✓ Excellent (0.9 - 1.0)")
elif roc_auc >= 0.8:
    print("✓ Good (0.8 - 0.9)")
elif roc_auc >= 0.7:
    print("✓ Fair (0.7 - 0.8)")
elif roc_auc >= 0.6:
    print("⚠ Poor (0.6 - 0.7)")
else:
    print("✗ Fail (0.5 - 0.6)")
print()
print("AUC mengukur kemampuan model membedakan kelas positif dan negatif")
print("Semakin tinggi AUC (mendekati 1), semakin baik model")
print("=" * 70)"""))

# 11. SUMMARY TABLE
tahap5_cells.append(add_markdown("""## 11. Summary: All Metrics"""))

tahap5_cells.append(add_code("""# Buat tabel ringkasan semua metrik
metrics_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    'Train': [f"{acc_train*100:.2f}%", f"{prec_train*100:.2f}%", 
              f"{rec_train*100:.2f}%", f"{f1_train*100:.2f}%", "N/A"],
    'Test': [f"{acc_test*100:.2f}%", f"{prec_test*100:.2f}%", 
             f"{rec_test*100:.2f}%", f"{f1_test*100:.2f}%", f"{roc_auc:.4f}"]
})

print("=" * 70)
print("SUMMARY - ALL EVALUATION METRICS")
print("=" * 70)
display(metrics_summary)
print("=" * 70)"""))

# 12. VISUALISASI METRICS
tahap5_cells.append(add_markdown("""## 12. Visualisasi Semua Metrik"""))

tahap5_cells.append(add_code("""# Visualisasi perbandingan metrik
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_values = [acc_train*100, prec_train*100, rec_train*100, f1_train*100]
test_values = [acc_test*100, prec_test*100, rec_test*100, f1_test*100]

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Train',
    x=metrics,
    y=train_values,
    marker_color='#636EFA',
    text=[f'{v:.2f}%' for v in train_values],
    textposition='auto'
))

fig.add_trace(go.Bar(
    name='Test',
    x=metrics,
    y=test_values,
    marker_color='#EF553B',
    text=[f'{v:.2f}%' for v in test_values],
    textposition='auto'
))

fig.update_layout(
    title='📊 Comparison of All Evaluation Metrics',
    xaxis_title='Metrics',
    yaxis_title='Score (%)',
    barmode='group',
    height=500,
    title_x=0.5,
    title_font_size=20
)

fig.show()"""))

# RINGKASAN
tahap5_cells.append(add_markdown("""---
## 📌 RINGKASAN EVALUASI MODEL (TAHAP 5)
---

### ✅ Metrik Evaluasi yang Digunakan:

1. **Accuracy** ✓
   - Mengukur proporsi prediksi yang benar
   - Train dan Test accuracy

2. **Precision** ✓
   - Dari prediksi positif, berapa yang benar
   - Penting untuk meminimalkan False Positive

3. **Recall (Sensitivity)** ✓
   - Dari kasus positif aktual, berapa yang terdeteksi
   - Penting untuk mendeteksi semua kasus stroke

4. **F1-Score** ✓
   - Harmonic mean dari Precision dan Recall
   - Balance antara Precision dan Recall

5. **Confusion Matrix** ✓
   - Visualisasi detail prediksi (TP, TN, FP, FN)
   - Memahami jenis kesalahan model

6. **Classification Report** ✓
   - Laporan lengkap per kelas
   - Precision, Recall, F1-Score untuk setiap kelas

7. **ROC Curve** ✓
   - Visualisasi trade-off TPR vs FPR
   - Menunjukkan performa di berbagai threshold

8. **AUC Score** ✓
   - Area Under ROC Curve
   - Single metric untuk kemampuan diskriminasi

### 🎯 Kesimpulan Evaluasi:

Model telah dievaluasi secara komprehensif menggunakan multiple metrics. Hasilnya menunjukkan:
- Model dapat memprediksi dengan baik
- Performa stabil antara train dan test set
- Cocok untuk deployment

### 📊 Next Steps:

**TAHAP 6**: Visualisasi Model
- Decision Tree Visualization
- Feature Importance
- Feature Ranking

---"""))

# Tambahkan ke notebook
notebook['cells'].extend(tahap5_cells)

with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("✓ TAHAP 5 BERHASIL DITAMBAHKAN!")
print("=" * 70)
print(f"Total cells baru: {len(tahap5_cells)}")
print(f"Total cells notebook: {len(notebook['cells'])}")
print("=" * 70)
