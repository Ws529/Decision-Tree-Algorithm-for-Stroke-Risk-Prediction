# 📊 Flowchart & Alur Program - Stroke Prediction System

## 🎯 Overview Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                  STROKE PREDICTION SYSTEM                        │
│                   (Decision Tree Algorithm)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2 KOMPONEN UTAMA                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Jupyter Notebook (Training & Development)                   │
│  2. Streamlit App (Production & User Interface)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 FLOWCHART 1: Training Model (Jupyter Notebook)

```
START
  │
  ▼
┌─────────────────────────┐
│ Load Dataset CSV        │
│ (5,110 rows × 12 cols)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TAHAP 1: Analisis Data  │
│ • Preview data          │
│ • Shape & info          │
│ • Missing values        │
│ • Statistik deskriptif  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TAHAP 2: EDA            │
│ • 15 visualisasi        │
│ • Distribusi fitur      │
│ • Korelasi              │
│ • Pattern analysis      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TAHAP 3: Preprocessing  │
│ • Drop ID column        │
│ • Handle missing (BMI)  │
│ • Label Encoding        │
│ • Feature Scaling       │
│ • Train-Test Split      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TAHAP 4: Training Model │
│ • Baseline model        │
│ • Hyperparameter tuning │
│ • Grid Search CV        │
│ • Select best model     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TAHAP 5: Evaluasi       │
│ • Accuracy, Precision   │
│ • Recall, F1-Score      │
│ • Confusion Matrix      │
│ • ROC Curve & AUC       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TAHAP 6: Visualisasi    │
│ • Decision Tree diagram │
│ • Feature Importance    │
│ • Top features ranking  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TAHAP 7: Save Model     │
│ • decision_tree.pkl     │
│ • scaler.pkl            │
│ • encoder.pkl           │
└────────────┬────────────┘
             │
             ▼
           [END]
   Model Siap Deploy!
```

---

## 🌐 FLOWCHART 2: Streamlit Application Flow

```
                           START
                             │
                             ▼
                    ┌────────────────┐
                    │  User Access   │
                    │  localhost:8501│
                    └───────┬────────┘
                            │
                            ▼
          ┌─────────────────────────────────┐
          │      STREAMLIT MULTIPAGE         │
          │         (6 Halaman)              │
          └─────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────────┐
          │                 │                     │
          ▼                 ▼                     ▼
    ┌─────────┐      ┌─────────┐          ┌─────────┐
    │  HOME   │      │ DATASET │          │   EDA   │
    └─────────┘      └─────────┘          └─────────┘
          │                 │                     │
          │                 │                     │
          ▼                 ▼                     ▼
    ┌─────────┐      ┌─────────┐          ┌─────────┐
    │  MODEL  │      │ PREDIKSI│          │ TENTANG │
    └─────────┘      └─────────┘          └─────────┘
```

### Detail Setiap Halaman:

#### 🏠 **HOME PAGE**
```
Start
  │
  ▼
┌──────────────────────┐
│ Load Model & Data    │
│ • decision_tree.pkl  │
│ • scaler.pkl         │
│ • encoder.pkl        │
│ • dataset.csv        │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Display Overview     │
│ • Total data metric  │
│ • Features metric    │
│ • Stroke cases       │
│ • Model accuracy     │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Quick Statistics     │
│ • Gender pie chart   │
│ • Stroke pie chart   │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Feature Cards        │
│ • Dataset Explorer   │
│ • EDA Interaktif     │
│ • Model Performance  │
└──────────────────────┘
```

#### 📁 **DATASET PAGE**
```
Start
  │
  ▼
┌──────────────────────┐
│ Load Dataset         │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Upload Feature       │ ◄──── USER UPLOAD CSV
│ • File uploader      │
│ • Validate CSV       │
│ • Replace/Use new    │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Preview Data         │
│ • Head/Tail/Sample   │
│ • Slider for rows    │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Show Information     │
│ • Column info        │
│ • Data types         │
│ • Missing values     │
│ • Duplicates         │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Statistics Tabs      │
│ • Numerikal stats    │
│ • Kategorikal stats  │
│ • Missing analysis   │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Download Options     │ ◄──── USER DOWNLOAD
│ • Full CSV           │
│ • Numeric only       │
│ • Statistics         │
└──────────────────────┘
```

#### 📈 **EDA PAGE (15 Visualisasi)**
```
Start
  │
  ▼
┌──────────────────────┐
│ Load Dataset         │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ TABS Navigation      │
│ • Overview           │
│ • Distribusi         │
│ • Korelasi           │
│ • Advanced           │
└─────────┬────────────┘
          │
          ├─────► TAB 1: OVERVIEW
          │       ├─ 1. Pie: Stroke Distribution
          │       ├─ 2. Bar: Gender vs Stroke
          │       ├─ 3. Bar: Hypertension vs Stroke
          │       ├─ 4. Bar: Heart Disease vs Stroke
          │       └─ 5. Bar: Smoking Status vs Stroke
          │
          ├─────► TAB 2: DISTRIBUSI
          │       ├─ 6. Histogram: Age
          │       ├─ 7. Histogram: BMI
          │       ├─ 8. Histogram: Glucose
          │       ├─ 9. Boxplot: BMI vs Stroke
          │       └─ 10. Boxplot: Age vs Stroke
          │
          ├─────► TAB 3: KORELASI
          │       ├─ 11. Heatmap: Correlation Matrix
          │       └─ Bar: Correlation with Stroke
          │
          └─────► TAB 4: ADVANCED
                  ├─ 12. Scatter: Age vs Glucose
                  ├─ 13. Scatter: BMI vs Glucose
                  ├─ 14. Subplot: 4 Distributions
                  └─ 15. Scatter Matrix: Pairplot
```

#### 🤖 **MODEL PAGE**
```
Start
  │
  ▼
┌──────────────────────┐
│ Load Model Files     │
│ • Check .pkl exists  │
└─────────┬────────────┘
          │
          ├─── YES ──┐
          │          ▼
          │    ┌──────────────────┐
          │    │ Display Info     │
          │    │ • Algorithm      │
          │    │ • Parameters     │
          │    │ • Tree depth     │
          │    │ • Total nodes    │
          │    └────────┬─────────┘
          │             │
          │             ▼
          │    ┌──────────────────┐
          │    │ Show Metrics     │
          │    │ • Accuracy       │
          │    │ • Precision      │
          │    │ • Recall         │
          │    │ • F1-Score       │
          │    │ • ROC AUC        │
          │    └────────┬─────────┘
          │             │
          │             ▼
          │    ┌──────────────────┐
          │    │ Visualizations   │
          │    │ • Confusion Mat. │
          │    │ • ROC Curve      │
          │    │ • Feature Imp.   │
          │    │ • Decision Tree  │
          │    └──────────────────┘
          │
          └─── NO ───┐
                     ▼
               ┌──────────────────┐
               │ Show Error       │
               │ "Model not found"│
               │ "Train first"    │
               └──────────────────┘
```

#### 🔮 **PREDIKSI PAGE**
```
Start
  │
  ▼
┌──────────────────────┐
│ Check Model Loaded   │
└─────────┬────────────┘
          │
          ├─── NO ──► ERROR: Model not found
          │
          └─── YES
                │
                ▼
         ┌──────────────────┐
         │ Display Form     │
         │ Input Fields:    │
         │ • Gender         │
         │ • Age            │
         │ • Hypertension   │
         │ • Heart Disease  │
         │ • Ever Married   │
         │ • Work Type      │
         │ • Residence      │
         │ • Glucose Level  │
         │ • BMI            │
         │ • Smoking Status │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ User Fill Form   │ ◄─── USER INPUT
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Click "Prediksi" │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Preprocessing    │
         │ • Encode categor.│
         │ • Scale features │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Model Predict    │
         │ • Get prediction │
         │ • Get probability│
         └────────┬─────────┘
                  │
          ┌───────┴───────┐
          │               │
      Stroke=0        Stroke=1
          │               │
          ▼               ▼
    ┌──────────┐    ┌──────────┐
    │ LOW RISK │    │HIGH RISK │
    │ Display: │    │ Display: │
    │ • Result │    │ • Result │
    │ • Prob % │    │ • Prob % │
    │ • Chart  │    │ • Chart  │
    │ • Advice │    │ • Warning│
    └──────────┘    └──────────┘
          │               │
          └───────┬───────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Show Analysis    │
         │ • Risk factors   │
         │ • Probability    │
         │ • Visualization  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Download Result  │ ◄─── USER DOWNLOAD
         │ • CSV format     │
         └──────────────────┘
```

#### ℹ️ **TENTANG PAGE**
```
Start
  │
  ▼
┌──────────────────────┐
│ Display Information  │
│ • Deskripsi proyek   │
│ • Algoritma detail   │
│ • Dataset info       │
│ • Tools & tech       │
│ • Tim pengembang     │
│ • Academic info      │
│ • References         │
│ • License            │
│ • Contact            │
└──────────────────────┘
```

---

## 🔄 FLOWCHART 3: Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                        │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Raw Dataset     │
│  (CSV File)      │
│  5,110 × 12      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  PREPROCESSING   │
│  • Drop ID       │
│  • Fill NA (BMI) │
│  • Encode        │
│  • Scale         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  SPLIT DATA      │
│  Train: 80%      │
│  Test:  20%      │
└────────┬─────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  Training Set   │   │   Testing Set   │
│  4,088 samples  │   │   1,022 samples │
└────────┬────────┘   └────────┬────────┘
         │                     │
         ▼                     │
┌─────────────────┐            │
│ TRAIN MODEL     │            │
│ Decision Tree   │            │
│ • Hyperparams   │            │
│ • Fit data      │            │
└────────┬────────┘            │
         │                     │
         ▼                     ▼
┌──────────────────────────────────┐
│        EVALUATE MODEL            │
│  • Predict on train & test       │
│  • Calculate metrics             │
│  • Generate visualizations       │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│      SAVE TRAINED MODEL          │
│  • decision_tree.pkl             │
│  • scaler.pkl                    │
│  • encoder.pkl                   │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│     STREAMLIT APPLICATION        │
│  • Load saved models             │
│  • User interface                │
│  • Real-time prediction          │
└──────────────────────────────────┘
             │
             ▼
      ┌──────────────┐
      │ NEW USER DATA│
      │ (Form Input) │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │ PREPROCESS   │
      │ (Same steps) │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │  PREDICT     │
      │ (Model .pkl) │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │ SHOW RESULT  │
      │ • Risk level │
      │ • Probability│
      │ • Advice     │
      └──────────────┘
```

---

## 🎯 FLOWCHART 4: Decision Making Process

```
┌─────────────────────────────────────────────────────────────┐
│           DECISION TREE PREDICTION LOGIC                    │
└─────────────────────────────────────────────────────────────┘

                    [START: New Patient Data]
                              │
                              ▼
                        ┌──────────┐
                        │ Age > 60?│
                        └─────┬────┘
                              │
                    ┌─────────┴─────────┐
                   NO                   YES
                    │                    │
                    ▼                    ▼
            ┌──────────────┐    ┌──────────────┐
            │ BMI > 30?    │    │Heart Disease?│
            └──────┬───────┘    └──────┬───────┘
                   │                    │
           ┌───────┴────────┐   ┌──────┴──────┐
          NO               YES  NO            YES
           │                │    │              │
           ▼                ▼    ▼              ▼
    ┌──────────┐    ┌──────────┐│         [STROKE]
    │NO STROKE │    │Hyperten? ││         HIGH RISK
    │LOW RISK  │    └────┬─────┘│
    └──────────┘         │      │
                   ┌─────┴──┐   │
                  NO       YES  │
                   │        │   │
                   ▼        ▼   ▼
            ┌──────────┐ [STROKE]
            │NO STROKE │ HIGH RISK
            │LOW RISK  │
            └──────────┘

[Output]
  │
  ├─► Prediction: 0 (No Stroke) or 1 (Stroke)
  ├─► Probability: [P(No Stroke), P(Stroke)]
  └─► Confidence: % value
```

---

## 📊 FLOWCHART 5: Feature Importance Flow

```
┌─────────────────────────────────────────────────────────────┐
│              FEATURE IMPORTANCE ANALYSIS                    │
└─────────────────────────────────────────────────────────────┘

         [Decision Tree Model Trained]
                     │
                     ▼
         ┌───────────────────────┐
         │ Extract Feature       │
         │ Importance Values     │
         │ from model.feature_   │
         │ importances_          │
         └──────────┬────────────┘
                    │
                    ▼
         ┌───────────────────────┐
         │ Rank Features         │
         │ by Importance Score   │
         │ (Highest to Lowest)   │
         └──────────┬────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│ TOP 5 Features:  │  │ Visualization:   │
│ 1. Age           │  │ • Bar Chart      │
│ 2. Glucose       │  │ • Pie Chart      │
│ 3. BMI           │  │ • Table          │
│ 4. Hypertension  │  │                  │
│ 5. Heart Disease │  │                  │
└──────────────────┘  └──────────────────┘
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
         ┌───────────────────────┐
         │ Use for:              │
         │ • Model explanation   │
         │ • Feature selection   │
         │ • Medical insights    │
         │ • Risk prioritization │
         └───────────────────────┘
```

---

## 🔐 FLOWCHART 6: Error Handling & Validation

```
┌─────────────────────────────────────────────────────────────┐
│                ERROR HANDLING FLOW                          │
└─────────────────────────────────────────────────────────────┘

    [User Action]
         │
         ▼
    ┌─────────────┐
    │ Validate    │
    │ Input       │
    └──────┬──────┘
           │
     ┌─────┴─────┐
    Valid?      Invalid
     │             │
     │             ▼
     │      ┌──────────────┐
     │      │ Show Error   │
     │      │ Message      │
     │      │ • Red box    │
     │      │ • Clear msg  │
     │      └──────────────┘
     │             │
     │             ▼
     │      [Return to Form]
     │
     ▼
┌──────────────┐
│ Try Process  │
└──────┬───────┘
       │
  ┌────┴────┐
Success  Exception
  │          │
  │          ▼
  │    ┌──────────────┐
  │    │ Catch Error  │
  │    │ • Log error  │
  │    │ • User msg   │
  │    └──────────────┘
  │          │
  │          ▼
  │    [Show Error Page]
  │
  ▼
[Continue]
```

---

## 📝 KETERANGAN SIMBOL

```
┌──────┐
│ Box  │  = Proses/Aksi
└──────┘

   │
   ▼      = Alur/Flow

  ┌─┐
  │?│     = Decision/Kondisi
  └─┘

[Text]    = Output/Result

◄────     = Input dari User

─────►    = Output ke User
```

---

## 🎯 KEY POINTS

### ✅ Upload Dataset Feature:
- **Lokasi**: pages/1_Dataset.py (line 72-84)
- **Format**: CSV file
- **Validasi**: Auto validate
- **Action**: Replace current dataset atau use alongside

### ✅ Visualisasi Hasil:
- **15+ Visualisasi** di pages/2_EDA.py
- **Interactive Plotly** charts
- **4 Tabs** untuk organized view:
  - Overview (5 viz)
  - Distribusi (5 viz)
  - Korelasi (2 viz)
  - Advanced (3 viz)

### ✅ Download Features:
- **Dataset CSV** - Full dataset
- **Numeric only** - Only numeric columns
- **Statistics** - Descriptive stats
- **Prediction Result** - After prediction

---

## 🚀 Execution Flow Summary

```
1. USER → Access Streamlit App
2. APP → Load models & data
3. USER → Navigate pages
4. USER → Upload CSV (optional)
5. USER → View visualizations
6. USER → Input data for prediction
7. APP → Process & predict
8. APP → Show result with visualization
9. USER → Download result (optional)
10. USER → Explore other features
```

---

**📌 Semua flowchart di atas menggambarkan alur lengkap dari training hingga deployment!**
