"""
Script untuk menambahkan TAHAP 2 - EDA ke notebook
"""

import json

# Cell untuk TAHAP 2 header
tahap2_cells = []

# Header TAHAP 2
tahap2_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# TAHAP 2 — EXPLORATORY DATA ANALYSIS (EDA)\n",
        "---\n",
        "\n",
        "Pada tahap ini akan dilakukan visualisasi data menggunakan **Plotly** untuk memahami pola, distribusi, dan hubungan antar variabel dalam dataset.\n",
        "\n",
        "Visualisasi yang akan dibuat:\n",
        "1. Distribusi Stroke (Pie Chart)\n",
        "2. Distribusi Gender\n",
        "3. Distribusi Hypertension\n",
        "4. Distribusi Heart Disease\n",
        "5. Distribusi Smoking Status\n",
        "6. Histogram Age\n",
        "7. Histogram BMI\n",
        "8. Histogram Average Glucose Level\n",
        "9. Boxplot BMI\n",
        "10. Boxplot Age\n",
        "11. Heatmap Korelasi\n",
        "12. Scatter Plot Age vs Glucose\n",
        "13. Scatter Plot BMI vs Glucose\n",
        "14. Feature Distribution\n",
        "15. Analysis tambahan"
    ]
})

# 1. Distribusi Stroke - Pie Chart
tahap2_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Distribusi Stroke (Pie Chart)\n",
        "\n",
        "Visualisasi proporsi pasien yang mengalami stroke vs tidak mengalami stroke dalam bentuk pie chart interaktif."
    ]
})

tahap2_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==========================================================\n",
        "# 1. DISTRIBUSI STROKE (PIE CHART)\n",
        "# ==========================================================\n",
        "\n",
        "stroke_counts = df['stroke'].value_counts()\n",
        "labels = ['Tidak Stroke', 'Stroke']\n",
        "values = stroke_counts.values\n",
        "colors = ['#00CC96', '#EF553B']\n",
        "\n",
        "fig = go.Figure(data=[go.Pie(\n",
        "    labels=labels,\n",
        "    values=values,\n",
        "    hole=0.4,\n",
        "    marker=dict(colors=colors, line=dict(color='white', width=2)),\n",
        "    textinfo='label+percent+value',\n",
        "    textfont=dict(size=14),\n",
        "    hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>'\n",
        ")])\n",
        "\n",
        "fig.update_layout(\n",
        "    title={\n",
        "        'text': '🎯 Distribusi Target Variable - Stroke',\n",
        "        'x': 0.5,\n",
        "        'xanchor': 'center',\n",
        "        'font': {'size': 20, 'family': 'Arial Black'}\n",
        "    },\n",
        "    height=500,\n",
        "    showlegend=True,\n",
        "    legend=dict(\n",
        "        orientation='h',\n",
        "        yanchor='bottom',\n",
        "        y=-0.15,\n",
        "        xanchor='center',\n",
        "        x=0.5\n",
        "    )\n",
        ")\n",
        "\n",
        "fig.show()\n",
        "\n",
        "print(\"\\n📊 INTERPRETASI:\")\n",
        "print(\"=\" * 70)\n",
        "print(f\"• Tidak Stroke: {stroke_counts[0]:,} pasien ({stroke_counts[0]/len(df)*100:.2f}%)\")\n",
        "print(f\"• Stroke      : {stroke_counts[1]:,} pasien ({stroke_counts[1]/len(df)*100:.2f}%)\")\n",
        "print(f\"\\n⚠️  Dataset sangat imbalanced dengan rasio {stroke_counts[0]/stroke_counts[1]:.1f}:1\")\n",
        "print(\"   Model perlu strategi khusus untuk menangani ketidakseimbangan ini.\")\n",
        "print(\"=\" * 70)"
    ]
})

# 2. Distribusi Gender
tahap2_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Distribusi Gender\n",
        "\n",
        "Visualisasi distribusi jenis kelamin pasien dengan breakdown berdasarkan status stroke."
    ]
})

tahap2_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==========================================================\n",
        "# 2. DISTRIBUSI GENDER\n",
        "# ==========================================================\n",
        "\n",
        "# Data untuk gender\n",
        "gender_stroke = pd.crosstab(df['gender'], df['stroke'])\n",
        "gender_stroke.columns = ['Tidak Stroke', 'Stroke']\n",
        "\n",
        "fig = go.Figure()\n",
        "\n",
        "fig.add_trace(go.Bar(\n",
        "    name='Tidak Stroke',\n",
        "    x=gender_stroke.index,\n",
        "    y=gender_stroke['Tidak Stroke'],\n",
        "    marker_color='#00CC96',\n",
        "    text=gender_stroke['Tidak Stroke'],\n",
        "    textposition='auto',\n",
        "    hovertemplate='<b>%{x}</b><br>Tidak Stroke: %{y}<extra></extra>'\n",
        "))\n",
        "\n",
        "fig.add_trace(go.Bar(\n",
        "    name='Stroke',\n",
        "    x=gender_stroke.index,\n",
        "    y=gender_stroke['Stroke'],\n",
        "    marker_color='#EF553B',\n",
        "    text=gender_stroke['Stroke'],\n",
        "    textposition='auto',\n",
        "    hovertemplate='<b>%{x}</b><br>Stroke: %{y}<extra></extra>'\n",
        "))\n",
        "\n",
        "fig.update_layout(\n",
        "    title={\n",
        "        'text': '👥 Distribusi Gender vs Stroke',\n",
        "        'x': 0.5,\n",
        "        'xanchor': 'center',\n",
        "        'font': {'size': 20, 'family': 'Arial Black'}\n",
        "    },\n",
        "    xaxis_title='Gender',\n",
        "    yaxis_title='Jumlah Pasien',\n",
        "    barmode='group',\n",
        "    height=500,\n",
        "    legend=dict(\n",
        "        orientation='h',\n",
        "        yanchor='bottom',\n",
        "        y=1.02,\n",
        "        xanchor='center',\n",
        "        x=0.5\n",
        "    ),\n",
        "    hovermode='x unified'\n",
        ")\n",
        "\n",
        "fig.show()\n",
        "\n",
        "print(\"\\n📊 INTERPRETASI:\")\n",
        "print(\"=\" * 70)\n",
        "for gender in df['gender'].unique():\n",
        "    total = len(df[df['gender'] == gender])\n",
        "    stroke = len(df[(df['gender'] == gender) & (df['stroke'] == 1)])\n",
        "    print(f\"• {gender:10s}: {total:,} pasien, Stroke: {stroke:3d} ({stroke/total*100:.2f}%)\")\n",
        "print(\"=\" * 70)"
    ]
})

print(json.dumps(tahap2_cells, indent=2))
