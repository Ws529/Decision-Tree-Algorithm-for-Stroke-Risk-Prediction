import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load dataset
try:
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
except FileNotFoundError:
    raise SystemExit('Dataset healthcare-dataset-stroke-data.csv tidak ditemukan.')

# Preprocessing
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Fill missing bmi with median
if 'bmi' in df.columns:
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Label encode categorical features
label_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoders = {}
for col in label_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Split features and target
if 'stroke' not in df.columns:
    raise SystemExit('Kolom target stroke tidak ditemukan.')

X = df.drop(columns=['stroke'])
y = df['stroke']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_scaled, y)

# Simpan model dan preprocessing
joblib.dump(model, 'decision_tree.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoder.pkl')

print('Model, scaler, encoder berhasil disimpan: decision_tree.pkl, scaler.pkl, encoder.pkl')
