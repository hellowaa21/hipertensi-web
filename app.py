import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Judul Halaman ===
st.title("🩺 Prediksi Risiko Hipertensi")
st.write("Masukkan data pasien untuk memprediksi risiko hipertensi menggunakan model Naive Bayes.")

# === Dataset Buatan Sederhana ===
data = {
    'age': [25, 40, 55, 60, 35, 50, 45, 30, 65, 70],
    'sex': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    'bmi': [22.5, 27.8, 29.5, 31.0, 26.2, 28.4, 30.1, 23.5, 32.0, 33.5],
    'ap_hi': [120, 135, 150, 160, 128, 140, 155, 118, 165, 170],
    'ap_lo': [80, 85, 95, 100, 82, 88, 92, 78, 100, 105],
    'cholesterol': [1, 2, 2, 3, 1, 2, 2, 1, 3, 3],
    'gluc': [1, 1, 2, 3, 1, 2, 2, 1, 3, 3],
    'smoke': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    'alco': [0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    'active': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
    'hypertension': [0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# === Split Data ===
X = df.drop('hypertension', axis=1)
y = df['hypertension']

# === Model Naive Bayes ===
model = GaussianNB()
model.fit(X, y)

# === Form Input Manual ===
st.subheader("Masukkan Data Pasien")
age = st.number_input("Usia (tahun):", 18, 100, 30)
sex = st.selectbox("Jenis Kelamin:", ["Perempuan", "Laki-laki"])
bmi = st.number_input("BMI (Body Mass Index):", 10.0, 50.0, 25.0)
ap_hi = st.number_input("Tekanan Darah Sistolik (mmHg):", 80, 250, 120)
ap_lo = st.number_input("Tekanan Darah Diastolik (mmHg):", 50, 150, 80)
cholesterol = st.selectbox("Tingkat Kolesterol:", [1, 2, 3])
gluc = st.selectbox("Tingkat Glukosa:", [1, 2, 3])
smoke = st.radio("Apakah pasien merokok?", ["Tidak", "Ya"])
alco = st.radio("Apakah pasien mengonsumsi alkohol?", ["Tidak", "Ya"])
active = st.radio("Apakah pasien aktif berolahraga?", ["Tidak", "Ya"])

# === Encode Input ===
sex_val = 1 if sex == "Laki-laki" else 0
smoke_val = 1 if smoke == "Ya" else 0
alco_val = 1 if alco == "Ya" else 0
active_val = 1 if active == "Ya" else 0

# === Prediksi ===
if st.button("Prediksi Risiko"):
    new_patient = np.array([[age, sex_val, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke_val, alco_val, active_val]])
    prob = model.predict_proba(new_patient)[0][1] * 100
    pred = model.predict(new_patient)[0]

    st.subheader("Hasil Prediksi:")
    if pred == 1:
        st.error(f"⚠️ Pasien berisiko tinggi mengalami hipertensi ({prob:.2f}%)")
    else:
        st.success(f"✅ Pasien berisiko rendah mengalami hipertensi ({prob:.2f}%)")