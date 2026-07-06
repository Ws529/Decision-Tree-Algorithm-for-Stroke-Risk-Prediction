import json

# Baca notebook yang sudah ada
with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Cell-cell TAHAP 2
new_cells = []

# ========== HEADER TAHAP 2 ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# TAHAP 2 — EXPLORATORY DATA ANALYSIS (EDA)\n",
        "---\n",
        "\n",
        "Pada tahap ini akan dilakukan visualisasi data menggunakan **Plotly** untuk memahami pola, distribusi, dan hubungan antar variabel dalam dataset."
    ]
})

# ========== 1. DISTRIBUSI STROKE ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Distribusi Stroke (Pie Chart)"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "stroke_counts = df['stroke'].value_counts()\n",
        "labels = ['Tidak Stroke', 'Stroke']\n",
        "values = stroke_counts.values\n",
        "colors = ['#00CC96', '#EF553B']\n",
        "\n",
        "fig = go.Figure(data=[go.Pie(\n",
        "    labels=labels, values=values, hole=0.4,\n",
        "    marker=dict(colors=colors, line=dict(color='white', width=2)),\n",
        "    textinfo='label+percent+value', textfont=dict(size=14)\n",
        ")])\n",
        "\n",
        "fig.update_layout(\n",
        "    title={'text': '🎯 Distribusi Target Variable - Stroke', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},\n",
        "    height=500, showlegend=True\n",
        ")\n",
        "fig.show()\n",
        "\n",
        "print(f\"Tidak Stroke: {stroke_counts[0]:,} ({stroke_counts[0]/len(df)*100:.2f}%)\")\n",
        "print(f\"Stroke: {stroke_counts[1]:,} ({stroke_counts[1]/len(df)*100:.2f}%)\")"
    ]
})

# ========== 2. DISTRIBUSI GENDER ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Distribusi Gender"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "gender_stroke = pd.crosstab(df['gender'], df['stroke'])\n",
        "gender_stroke.columns = ['Tidak Stroke', 'Stroke']\n",
        "\n",
        "fig = go.Figure()\n",
        "fig.add_trace(go.Bar(name='Tidak Stroke', x=gender_stroke.index, y=gender_stroke['Tidak Stroke'], marker_color='#00CC96'))\n",
        "fig.add_trace(go.Bar(name='Stroke', x=gender_stroke.index, y=gender_stroke['Stroke'], marker_color='#EF553B'))\n",
        "\n",
        "fig.update_layout(\n",
        "    title={'text': '👥 Distribusi Gender vs Stroke', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},\n",
        "    xaxis_title='Gender', yaxis_title='Jumlah', barmode='group', height=500\n",
        ")\n",
        "fig.show()"
    ]
})

# ========== 3. DISTRIBUSI HYPERTENSION ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. Distribusi Hypertension"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "hyper_stroke = pd.crosstab(df['hypertension'], df['stroke'])\n",
        "hyper_stroke.index = ['Tidak Hipertensi', 'Hipertensi']\n",
        "hyper_stroke.columns = ['Tidak Stroke', 'Stroke']\n",
        "\n",
        "fig = go.Figure()\n",
        "fig.add_trace(go.Bar(name='Tidak Stroke', x=hyper_stroke.index, y=hyper_stroke['Tidak Stroke'], marker_color='#00CC96'))\n",
        "fig.add_trace(go.Bar(name='Stroke', x=hyper_stroke.index, y=hyper_stroke['Stroke'], marker_color='#EF553B'))\n",
        "\n",
        "fig.update_layout(\n",
        "    title={'text': '💊 Distribusi Hypertension vs Stroke', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},\n",
        "    xaxis_title='Hypertension', yaxis_title='Jumlah', barmode='group', height=500\n",
        ")\n",
        "fig.show()"
    ]
})

# ========== 4. DISTRIBUSI HEART DISEASE ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. Distribusi Heart Disease"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "heart_stroke = pd.crosstab(df['heart_disease'], df['stroke'])\n",
        "heart_stroke.index = ['Tidak Penyakit Jantung', 'Penyakit Jantung']\n",
        "heart_stroke.columns = ['Tidak Stroke', 'Stroke']\n",
        "\n",
        "fig = go.Figure()\n",
        "fig.add_trace(go.Bar(name='Tidak Stroke', x=heart_stroke.index, y=heart_stroke['Tidak Stroke'], marker_color='#00CC96'))\n",
        "fig.add_trace(go.Bar(name='Stroke', x=heart_stroke.index, y=heart_stroke['Stroke'], marker_color='#EF553B'))\n",
        "\n",
        "fig.update_layout(\n",
        "    title={'text': '❤️ Distribusi Heart Disease vs Stroke', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},\n",
        "    xaxis_title='Heart Disease', yaxis_title='Jumlah', barmode='group', height=500\n",
        ")\n",
        "fig.show()"
    ]
})

# ========== 5. DISTRIBUSI SMOKING STATUS ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 5. Distribusi Smoking Status"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "smoking_stroke = pd.crosstab(df['smoking_status'], df['stroke'])\n",
        "smoking_stroke.columns = ['Tidak Stroke', 'Stroke']\n",
        "\n",
        "fig = go.Figure()\n",
        "fig.add_trace(go.Bar(name='Tidak Stroke', x=smoking_stroke.index, y=smoking_stroke['Tidak Stroke'], marker_color='#00CC96'))\n",
        "fig.add_trace(go.Bar(name='Stroke', x=smoking_stroke.index, y=smoking_stroke['Stroke'], marker_color='#EF553B'))\n",
        "\n",
        "fig.update_layout(\n",
        "    title={'text': '🚬 Distribusi Smoking Status vs Stroke', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},\n",
        "    xaxis_title='Smoking Status', yaxis_title='Jumlah', barmode='group', height=500\n",
        ")\n",
        "fig.show()"
    ]
})

# ========== 6. HISTOGRAM AGE ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. Histogram Age"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.histogram(df, x='age', color='stroke',\n",
        "                   title='📊 Distribusi Usia Pasien',\n",
        "                   labels={'age': 'Usia (tahun)', 'stroke': 'Stroke'},\n",
        "                   color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "                   barmode='overlay', nbins=40, height=500)\n",
        "\n",
        "fig.update_traces(opacity=0.7)\n",
        "fig.update_layout(title_x=0.5, title_font_size=20)\n",
        "fig.show()\n",
        "\n",
        "print(f\"Usia rata-rata: {df['age'].mean():.2f} tahun\")\n",
        "print(f\"Usia median: {df['age'].median():.2f} tahun\")"
    ]
})

# ========== 7. HISTOGRAM BMI ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 7. Histogram BMI"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.histogram(df.dropna(subset=['bmi']), x='bmi', color='stroke',\n",
        "                   title='⚖️ Distribusi BMI (Body Mass Index)',\n",
        "                   labels={'bmi': 'BMI', 'stroke': 'Stroke'},\n",
        "                   color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "                   barmode='overlay', nbins=40, height=500)\n",
        "\n",
        "fig.update_traces(opacity=0.7)\n",
        "fig.update_layout(title_x=0.5, title_font_size=20)\n",
        "fig.show()\n",
        "\n",
        "print(f\"BMI rata-rata: {df['bmi'].mean():.2f}\")\n",
        "print(f\"BMI median: {df['bmi'].median():.2f}\")"
    ]
})

# ========== 8. HISTOGRAM GLUCOSE ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 8. Histogram Average Glucose Level"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.histogram(df, x='avg_glucose_level', color='stroke',\n",
        "                   title='🩸 Distribusi Average Glucose Level',\n",
        "                   labels={'avg_glucose_level': 'Glucose Level (mg/dL)', 'stroke': 'Stroke'},\n",
        "                   color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "                   barmode='overlay', nbins=40, height=500)\n",
        "\n",
        "fig.update_traces(opacity=0.7)\n",
        "fig.update_layout(title_x=0.5, title_font_size=20)\n",
        "fig.show()\n",
        "\n",
        "print(f\"Glucose rata-rata: {df['avg_glucose_level'].mean():.2f} mg/dL\")\n",
        "print(f\"Glucose median: {df['avg_glucose_level'].median():.2f} mg/dL\")"
    ]
})

# ========== 9. BOXPLOT BMI ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 9. Boxplot BMI"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.box(df, x='stroke', y='bmi',\n",
        "             title='📦 Boxplot BMI berdasarkan Stroke',\n",
        "             labels={'bmi': 'BMI', 'stroke': 'Stroke (0=Tidak, 1=Ya)'},\n",
        "             color='stroke', color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "             height=500)\n",
        "\n",
        "fig.update_layout(title_x=0.5, title_font_size=20, showlegend=False)\n",
        "fig.show()"
    ]
})

# ========== 10. BOXPLOT AGE ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 10. Boxplot Age"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.box(df, x='stroke', y='age',\n",
        "             title='📦 Boxplot Age berdasarkan Stroke',\n",
        "             labels={'age': 'Usia (tahun)', 'stroke': 'Stroke (0=Tidak, 1=Ya)'},\n",
        "             color='stroke', color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "             height=500)\n",
        "\n",
        "fig.update_layout(title_x=0.5, title_font_size=20, showlegend=False)\n",
        "fig.show()"
    ]
})

# ========== 11. HEATMAP KORELASI ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 11. Heatmap Korelasi"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "numeric_cols = df.select_dtypes(include=[np.number]).drop('id', axis=1)\n",
        "correlation = numeric_cols.corr()\n",
        "\n",
        "fig = px.imshow(correlation, text_auto='.2f',\n",
        "                title='🔥 Heatmap Korelasi Antar Fitur',\n",
        "                color_continuous_scale='RdBu_r',\n",
        "                aspect='auto', height=600)\n",
        "\n",
        "fig.update_layout(title_x=0.5, title_font_size=20)\n",
        "fig.show()\n",
        "\n",
        "print(\"\\nKorelasi dengan Target (Stroke):\")\n",
        "print(correlation['stroke'].sort_values(ascending=False))"
    ]
})

# ========== 12. SCATTER AGE VS GLUCOSE ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 12. Scatter Plot Age vs Glucose"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.scatter(df, x='age', y='avg_glucose_level', color='stroke',\n",
        "                 title='📈 Scatter Plot: Age vs Average Glucose Level',\n",
        "                 labels={'age': 'Usia (tahun)', 'avg_glucose_level': 'Glucose Level (mg/dL)'},\n",
        "                 color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "                 opacity=0.6, height=600)\n",
        "\n",
        "fig.update_layout(title_x=0.5, title_font_size=20)\n",
        "fig.show()"
    ]
})

# ========== 13. SCATTER BMI VS GLUCOSE ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 13. Scatter Plot BMI vs Glucose"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.scatter(df.dropna(subset=['bmi']), x='bmi', y='avg_glucose_level', color='stroke',\n",
        "                 title='📈 Scatter Plot: BMI vs Average Glucose Level',\n",
        "                 labels={'bmi': 'BMI', 'avg_glucose_level': 'Glucose Level (mg/dL)'},\n",
        "                 color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "                 opacity=0.6, height=600)\n",
        "\n",
        "fig.update_layout(title_x=0.5, title_font_size=20)\n",
        "fig.show()"
    ]
})

# ========== 14. FEATURE DISTRIBUTION ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 14. Feature Distribution Overview"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = make_subplots(rows=2, cols=2,\n",
        "                    subplot_titles=('Age Distribution', 'BMI Distribution',\n",
        "                                    'Glucose Level Distribution', 'Stroke Distribution'))\n",
        "\n",
        "fig.add_trace(go.Histogram(x=df['age'], name='Age', marker_color='#636EFA'), row=1, col=1)\n",
        "fig.add_trace(go.Histogram(x=df['bmi'].dropna(), name='BMI', marker_color='#EF553B'), row=1, col=2)\n",
        "fig.add_trace(go.Histogram(x=df['avg_glucose_level'], name='Glucose', marker_color='#00CC96'), row=2, col=1)\n",
        "fig.add_trace(go.Bar(x=['No Stroke', 'Stroke'], y=df['stroke'].value_counts().values, \n",
        "                     marker_color=['#00CC96', '#EF553B']), row=2, col=2)\n",
        "\n",
        "fig.update_layout(height=700, title_text='📊 Overview Distribusi Fitur Utama', title_x=0.5, showlegend=False)\n",
        "fig.show()"
    ]
})

# ========== 15. PAIRPLOT ALTERNATIF ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 15. Relationship Analysis - Multiple Features"]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig = px.scatter_matrix(df[['age', 'bmi', 'avg_glucose_level', 'stroke']].dropna(),\n",
        "                        dimensions=['age', 'bmi', 'avg_glucose_level'],\n",
        "                        color='stroke',\n",
        "                        title='🔍 Scatter Matrix: Relationship Antar Fitur Numerikal',\n",
        "                        color_discrete_map={0: '#00CC96', 1: '#EF553B'},\n",
        "                        height=800, opacity=0.6)\n",
        "\n",
        "fig.update_layout(title_x=0.5, title_font_size=20)\n",
        "fig.show()"
    ]
})

# ========== RINGKASAN TAHAP 2 ==========
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 📌 RINGKASAN EDA (TAHAP 2)\n",
        "---\n",
        "\n",
        "### ✅ Temuan Penting dari Visualisasi:\n",
        "\n",
        "1. **Target Variable (Stroke)**:\n",
        "   - Sangat imbalanced: 95% tidak stroke vs 5% stroke\n",
        "   - Perlu strategi khusus saat training\n",
        "\n",
        "2. **Age (Usia)**:\n",
        "   - Usia lebih tua memiliki risiko stroke lebih tinggi\n",
        "   - Distribusi age pada kasus stroke cenderung lebih tinggi\n",
        "\n",
        "3. **Hypertension & Heart Disease**:\n",
        "   - Pasien dengan hipertensi/penyakit jantung lebih berisiko stroke\n",
        "   - Korelasi positif dengan stroke\n",
        "\n",
        "4. **BMI**:\n",
        "   - Rata-rata BMI sekitar 28.89 (kategori overweight)\n",
        "   - Memiliki 201 missing values yang perlu ditangani\n",
        "\n",
        "5. **Glucose Level**:\n",
        "   - Distribusi cukup bervariasi\n",
        "   - Pasien stroke cenderung memiliki glucose level lebih tinggi\n",
        "\n",
        "6. **Korelasi**:\n",
        "   - Age memiliki korelasi tertinggi dengan stroke\n",
        "   - Hypertension, heart_disease, dan avg_glucose_level juga berkorelasi positif\n",
        "\n",
        "### 🎯 Langkah Selanjutnya:\n",
        "\n",
        "**TAHAP 3**: Data Preprocessing\n",
        "- Menghapus kolom id\n",
        "- Handle missing values (BMI)\n",
        "- Encoding fitur kategorikal\n",
        "- Feature scaling\n",
        "- Train-test split\n",
        "\n",
        "---"
    ]
})

# Tambahkan cell baru ke notebook
notebook['cells'].extend(new_cells)

# Simpan notebook
with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ TAHAP 2 berhasil ditambahkan ke notebook!")
print(f"✓ Total {len(new_cells)} cells baru ditambahkan")
