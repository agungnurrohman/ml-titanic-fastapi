import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== 1. DATA UNDERSTANDING ==========
# Memuat dataset dan menampilkan informasi dasar untuk memahami struktur data
df = pd.read_csv('train.csv')

print("Info Dataset:")
print(f"Jumlah baris: {len(df)}")
print(f"\nKolom: {df.columns.tolist()}")
print(f"\nBeberapa baris pertama:")
print(df.head())

# Target: Survived (0 = Not Survived, 1 = Survived)
print(f"\nDistribusi Target:")
print(df['Survived'].value_counts())

# ========== 2. FEATURE CLEANING ==========
# Mengatasi nilai kosong pada kolom untuk mencegah error
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ========== 3. FEATURE CREATION ==========
# Membuat fitur baru dari data mentah untuk meningkatkan informasi model
# Encode Sex
df['Sex_Encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# Family Size
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Is Alone
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)

# Fare per person
df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']

# Age Group
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
df['Age_Group'] = df['Age_Group'].astype(int)

# Title from Name
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
df['Title'] = df['Title'].fillna(0)

# ========== 4. FEATURE TRANSFORMATION ==========
# Mengubah fitur kategorikal menjadi numerik untuk input model
le_embarked = LabelEncoder()

df['Embarked_Encoded'] = le_embarked.fit_transform(df['Embarked'])

# ========== 5. FEATURE SELECTION & EVALUASI ==========
# Memilih fitur yang relevan dan mengevaluasi model
feature_columns = ['Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare',
                   'Embarked_Encoded', 'Family_Size', 'Is_Alone', 'Fare_Per_Person', 'Age_Group', 'Title']

X = df[feature_columns]
y = df['Survived']

print(f"\nFitur yang dipilih: {feature_columns}")
print(f"\nStatistik fitur:")
print(X.describe())

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nUkuran set pelatihan: {len(X_train)}")
print(f"Ukuran set pengujian: {len(X_test)}")

# Latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Buat prediksi pada data testing
y_pred = model.predict(X_test)

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy:.4f}")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

print("\nMatriks Kebingungan:")
print(confusion_matrix(y_test, y_pred))

# Analisis kepentingan fitur
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nKepentingan Fitur:")
print(feature_importance)

# ========== 6. REFLEKSI TEKNIS ==========
# Simpan model dan encoder untuk penggunaan di masa depan
joblib.dump(model, 'titanic_model.pkl')
joblib.dump(le_embarked, 'embarked_encoder.pkl')

print("\nModel dan encoder embarked telah disimpan!")

# Contoh prediksi untuk validasi model
sample_passenger = X_test.iloc[0:1]
prediction = model.predict(sample_passenger)
prediction_proba = model.predict_proba(sample_passenger)

predicted_survival = prediction[0]
print(f"\n--- Contoh Prediksi ---")
print(f"Prediksi Survival: {predicted_survival} (0 = Tidak Selamat, 1 = Selamat)")
print(f"Probabilitas Prediksi: {prediction_proba[0]}")
