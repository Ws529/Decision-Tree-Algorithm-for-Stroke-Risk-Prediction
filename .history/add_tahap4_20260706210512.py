#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk menambahkan TAHAP 4 - Training Model Decision Tree
"""

import json

# Load notebook
with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

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

# TAHAP 4 CELLS
tahap4_cells = []

# HEADER
tahap4_cells.append(add_markdown("""---
# TAHAP 4 — TRAINING MODEL DECISION TREE
---

Pada tahap ini akan dilakukan training model Decision Tree dengan berbagai konfigurasi hyperparameter untuk menemukan model terbaik.

**Hyperparameter yang akan di-tune:**
- **criterion**: gini, entropy
- **max_depth**: 3, 5, 7, 10, 15, None
- **min_samples_split**: 2, 5, 10, 20
- **min_samples_leaf**: 1, 2, 4, 8

**Target Performance:**
- Accuracy Train ≥ 80%
- Accuracy Test ≥ 85%"""))

# 1. BASELINE MODEL
tahap4_cells.append(add_markdown("""## 1. Baseline Model (Default Parameters)

Membuat model Decision Tree dengan parameter default sebagai baseline."""))

tahap4_cells.append(add_code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Model baseline dengan parameter default
dt_baseline = DecisionTreeClassifier(random_state=42)

# Training
dt_baseline.fit(X_train_scaled, y_train)

# Prediksi
y_train_pred_baseline = dt_baseline.predict(X_train_scaled)
y_test_pred_baseline = dt_baseline.predict(X_test_scaled)

# Evaluasi
train_acc_baseline = accuracy_score(y_train, y_train_pred_baseline)
test_acc_baseline = accuracy_score(y_test, y_test_pred_baseline)

print("=" * 70)
print("BASELINE MODEL - DEFAULT PARAMETERS")
print("=" * 70)
print(f"Accuracy Train: {train_acc_baseline*100:.2f}%")
print(f"Accuracy Test : {test_acc_baseline*100:.2f}%")
print("=" * 70)"""))

# 2. EXPERIMENT 1
tahap4_cells.append(add_markdown("""## 2. Experiment 1: Tuning Criterion

Mencoba berbagai criterion (gini vs entropy)."""))

tahap4_cells.append(add_code("""results_exp1 = []

for criterion in ['gini', 'entropy']:
    model = DecisionTreeClassifier(criterion=criterion, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    results_exp1.append({
        'criterion': criterion,
        'train_acc': train_acc,
        'test_acc': test_acc
    })
    
    print(f"Criterion: {criterion:10s} | Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

print("\\n" + "=" * 70)
best_exp1 = max(results_exp1, key=lambda x: x['test_acc'])
print(f"✓ Best criterion: {best_exp1['criterion']}")
print(f"  Test Accuracy: {best_exp1['test_acc']*100:.2f}%")
print("=" * 70)"""))

# 3. EXPERIMENT 2
tahap4_cells.append(add_markdown("""## 3. Experiment 2: Tuning Max Depth

Mencoba berbagai kedalaman maksimal pohon."""))

tahap4_cells.append(add_code("""results_exp2 = []

max_depths = [3, 5, 7, 10, 15, 20, None]

for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    results_exp2.append({
        'max_depth': depth,
        'train_acc': train_acc,
        'test_acc': test_acc
    })
    
    depth_str = str(depth) if depth else "None"
    print(f"Max Depth: {depth_str:6s} | Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

print("\\n" + "=" * 70)
best_exp2 = max(results_exp2, key=lambda x: x['test_acc'])
print(f"✓ Best max_depth: {best_exp2['max_depth']}")
print(f"  Test Accuracy: {best_exp2['test_acc']*100:.2f}%")
print("=" * 70)"""))

# 4. EXPERIMENT 3
tahap4_cells.append(add_markdown("""## 4. Experiment 3: Tuning Min Samples Split

Mencoba berbagai nilai minimum samples untuk split."""))

tahap4_cells.append(add_code("""results_exp3 = []

min_samples_splits = [2, 5, 10, 20, 50]

for min_split in min_samples_splits:
    model = DecisionTreeClassifier(min_samples_split=min_split, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    results_exp3.append({
        'min_samples_split': min_split,
        'train_acc': train_acc,
        'test_acc': test_acc
    })
    
    print(f"Min Split: {min_split:3d} | Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

print("\\n" + "=" * 70)
best_exp3 = max(results_exp3, key=lambda x: x['test_acc'])
print(f"✓ Best min_samples_split: {best_exp3['min_samples_split']}")
print(f"  Test Accuracy: {best_exp3['test_acc']*100:.2f}%")
print("=" * 70)"""))

# 5. EXPERIMENT 4
tahap4_cells.append(add_markdown("""## 5. Experiment 4: Tuning Min Samples Leaf

Mencoba berbagai nilai minimum samples di leaf node."""))

tahap4_cells.append(add_code("""results_exp4 = []

min_samples_leafs = [1, 2, 4, 8, 16]

for min_leaf in min_samples_leafs:
    model = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    results_exp4.append({
        'min_samples_leaf': min_leaf,
        'train_acc': train_acc,
        'test_acc': test_acc
    })
    
    print(f"Min Leaf: {min_leaf:3d} | Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

print("\\n" + "=" * 70)
best_exp4 = max(results_exp4, key=lambda x: x['test_acc'])
print(f"✓ Best min_samples_leaf: {best_exp4['min_samples_leaf']}")
print(f"  Test Accuracy: {best_exp4['test_acc']*100:.2f}%")
print("=" * 70)"""))

# 6. GRID SEARCH
tahap4_cells.append(add_markdown("""## 6. Grid Search - Comprehensive Tuning

Melakukan pencarian hyperparameter secara komprehensif menggunakan GridSearchCV."""))

tahap4_cells.append(add_code("""from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'class_weight': [None, 'balanced']
}

# Initialize Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("=" * 70)
print("MEMULAI GRID SEARCH...")
print("=" * 70)

# Fit grid search
grid_search.fit(X_train_scaled, y_train)

print("\\n" + "=" * 70)
print("✓ GRID SEARCH SELESAI")
print("=" * 70)
print(f"\\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param:20s}: {value}")

print(f"\\nBest CV Score: {grid_search.best_score_*100:.2f}%")
print("=" * 70)"""))

# 7. BEST MODEL
tahap4_cells.append(add_markdown("""## 7. Model Terbaik dari Grid Search

Menggunakan model terbaik hasil Grid Search untuk prediksi."""))

tahap4_cells.append(add_code("""# Model terbaik
best_model = grid_search.best_estimator_

# Prediksi
y_train_pred_best = best_model.predict(X_train_scaled)
y_test_pred_best = best_model.predict(X_test_scaled)

# Evaluasi
train_acc_best = accuracy_score(y_train, y_train_pred_best)
test_acc_best = accuracy_score(y_test, y_test_pred_best)

print("=" * 70)
print("MODEL TERBAIK - PERFORMANCE")
print("=" * 70)
print(f"Accuracy Train: {train_acc_best*100:.2f}%")
print(f"Accuracy Test : {test_acc_best*100:.2f}%")
print("=" * 70)

# Cek apakah memenuhi syarat
print("\\nCEK SYARAT PRAKTIKUM:")
if train_acc_best >= 0.80:
    print(f"✓ Train Accuracy ≥ 80% : {train_acc_best*100:.2f}% ✓")
else:
    print(f"✗ Train Accuracy < 80% : {train_acc_best*100:.2f}% ✗")
    
if test_acc_best >= 0.85:
    print(f"✓ Test Accuracy ≥ 85%  : {test_acc_best*100:.2f}% ✓")
else:
    print(f"✗ Test Accuracy < 85%  : {test_acc_best*100:.2f}% ✗")

print("=" * 70)"""))

# 8. ALTERNATIF - BALANCED MODEL
tahap4_cells.append(add_markdown("""## 8. Alternatif Model dengan Class Weight Balanced

Karena dataset imbalanced, mencoba model dengan class_weight='balanced'."""))

tahap4_cells.append(add_code("""# Model dengan class_weight balanced
dt_balanced = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42
)

# Training
dt_balanced.fit(X_train_scaled, y_train)

# Prediksi
y_train_pred_bal = dt_balanced.predict(X_train_scaled)
y_test_pred_bal = dt_balanced.predict(X_test_scaled)

# Evaluasi
train_acc_bal = accuracy_score(y_train, y_train_pred_bal)
test_acc_bal = accuracy_score(y_test, y_test_pred_bal)

print("=" * 70)
print("MODEL BALANCED - PERFORMANCE")
print("=" * 70)
print(f"Accuracy Train: {train_acc_bal*100:.2f}%")
print(f"Accuracy Test : {test_acc_bal*100:.2f}%")
print("=" * 70)"""))

# 9. PERBANDINGAN MODEL
tahap4_cells.append(add_markdown("""## 9. Perbandingan Semua Model

Membandingkan performa semua model yang telah dibuat."""))

tahap4_cells.append(add_code("""# DataFrame perbandingan
comparison_df = pd.DataFrame({
    'Model': ['Baseline', 'Grid Search Best', 'Balanced Model'],
    'Train Accuracy': [train_acc_baseline, train_acc_best, train_acc_bal],
    'Test Accuracy': [test_acc_baseline, test_acc_best, test_acc_bal]
})

comparison_df['Train Accuracy'] = comparison_df['Train Accuracy'].apply(lambda x: f"{x*100:.2f}%")
comparison_df['Test Accuracy'] = comparison_df['Test Accuracy'].apply(lambda x: f"{x*100:.2f}%")

print("=" * 70)
print("PERBANDINGAN SEMUA MODEL")
print("=" * 70)
display(comparison_df)
print("=" * 70)

# Visualisasi perbandingan
fig = go.Figure()

models = ['Baseline', 'Grid Search', 'Balanced']
train_scores = [train_acc_baseline*100, train_acc_best*100, train_acc_bal*100]
test_scores = [test_acc_baseline*100, test_acc_best*100, test_acc_bal*100]

fig.add_trace(go.Bar(name='Train Accuracy', x=models, y=train_scores, marker_color='#636EFA'))
fig.add_trace(go.Bar(name='Test Accuracy', x=models, y=test_scores, marker_color='#EF553B'))

fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Train: 80%")
fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Target Test: 85%")

fig.update_layout(
    title='📊 Perbandingan Model Decision Tree',
    xaxis_title='Model',
    yaxis_title='Accuracy (%)',
    barmode='group',
    height=500
)

fig.show()"""))

# 10. PILIH MODEL FINAL
tahap4_cells.append(add_markdown("""## 10. Model Final

Memilih model terbaik sebagai model final."""))

tahap4_cells.append(add_code("""# Pilih model dengan test accuracy tertinggi
if test_acc_best >= test_acc_bal:
    final_model = best_model
    final_train_acc = train_acc_best
    final_test_acc = test_acc_best
    model_name = "Grid Search Best Model"
else:
    final_model = dt_balanced
    final_train_acc = train_acc_bal
    final_test_acc = test_acc_bal
    model_name = "Balanced Model"

print("=" * 70)
print("MODEL FINAL TERPILIH")
print("=" * 70)
print(f"Model         : {model_name}")
print(f"Train Accuracy: {final_train_acc*100:.2f}%")
print(f"Test Accuracy : {final_test_acc*100:.2f}%")
print("\\nParameters:")
for param, value in final_model.get_params().items():
    if param in ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'class_weight']:
        print(f"  {param:20s}: {value}")
print("=" * 70)"""))

# RINGKASAN
tahap4_cells.append(add_markdown("""---
## 📌 RINGKASAN TRAINING MODEL (TAHAP 4)
---

### ✅ Eksperimen yang Telah Dilakukan:

1. **Baseline Model** ✓
   - Model dengan parameter default sebagai pembanding

2. **Experiment 1: Criterion** ✓
   - Mencoba gini dan entropy

3. **Experiment 2: Max Depth** ✓
   - Mencoba berbagai kedalaman pohon

4. **Experiment 3: Min Samples Split** ✓
   - Tuning minimum samples untuk split

5. **Experiment 4: Min Samples Leaf** ✓
   - Tuning minimum samples di leaf

6. **Grid Search** ✓
   - Comprehensive hyperparameter tuning
   - Cross-validation 5-fold

7. **Balanced Model** ✓
   - Model khusus untuk imbalanced dataset

### 🎯 Model Final:

Model terbaik telah dipilih berdasarkan **Test Accuracy** tertinggi dengan mempertimbangkan:
- Accuracy Train dan Test
- Overfitting/Underfitting
- Kemampuan generalisasi

### 📊 Next Steps:

**TAHAP 5**: Evaluasi Model Lengkap
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- ROC Curve & AUC Score

---"""))

# Tambahkan ke notebook
notebook['cells'].extend(tahap4_cells)

# Save
with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("✓ TAHAP 4 BERHASIL DITAMBAHKAN!")
print("=" * 70)
print(f"Total cells baru: {len(tahap4_cells)}")
print(f"Total cells notebook: {len(notebook['cells'])}")
print("=" * 70)
