import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# CONFIGURASI HALAMAN WEB
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Heart Screening - SDGs 3",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------------------------------------
# FUNGSI LOAD MODEL & SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_ml_components():
    model_path = os.path.join('models', 'heart_disease_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    features_path = os.path.join('models', 'feature_columns.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    
    return model, scaler, feature_cols

try:
    model, scaler, feature_cols = load_ml_components()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file .pkl sudah tersimpan di folder 'models/'.")
    st.stop()

# ---------------------------------------------------------
# HEADER & HERO SECTION LANDING PAGE
# ---------------------------------------------------------
st.title("❤️ AI Heart Disease Risk Prediction System")
st.subheader("Sistem Deteksi Dini Risiko Penyakit Jantung Berbasis Machine Learning")
st.markdown("""
Aplikasi web ini memanfaatkan teknologi **Artificial Intelligence** untuk membantu mendeteksi potensi risiko penyakit jantung pada pasien secara presisi berdasarkan parameter klinis dasar.
""")

st.divider()

# ---------------------------------------------------------
# TAB NAVEGASI: PREDIKSI & INFORMASI PROJECT
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🩺 Skrining Prediksi Pasien", "ℹ️ Informasi Project & Edukasi SDGs 3"])

# =========================================================
# TAB 1: FORM INPUT & RESULT PREDIKSI
# =========================================================
with tab1:
    st.markdown("### Input Parameter Kesehatan Pasien")
    st.write("Isi data medis pasien di bawah ini untuk mendapatkan estimasi diagnosa AI secara *real-time*:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Umur Pasien (Tahun)", min_value=1, max_value=120, value=52)
        sex = st.selectbox("Jenis Kelamin", options=["Laki-laki (M)", "Perempuan (F)"])
        chest_pain = st.selectbox(
            "Tipe Nyeri Dada (Chest Pain)",
            options=[
                "ASY (Asymptomatic / Tanpa Gejala Khas)",
                "ATA (Atypical Angina)",
                "NAP (Non-Anginal Pain)",
                "TA (Typical Angina)"
            ]
        )
        resting_bp = st.number_input("Tekanan Darah Istirahat (Resting BP mm Hg)", min_value=50, max_value=250, value=125)

    with col2:
        cholesterol = st.number_input("Kadar Kolesterol Serum (mm/dl)", min_value=50, max_value=600, value=210)
        fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", options=["Tidak (0)", "Ya (1)"])
        resting_ecg = st.selectbox("Hasil ECG Istirahat (Resting ECG)", options=["Normal", "LVH", "ST"])
        max_hr = st.number_input("Detak Jantung Maksimum (Max HR)", min_value=60, max_value=220, value=150)

    with col3:
        exercise_angina = st.selectbox("Nyeri Dada Saat Olahraga? (Exercise Angina)", options=["Tidak (N)", "Ya (Y)"])
        oldpeak = st.number_input("Depresi ST (Oldpeak)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
        st_slope = st.selectbox("Kemiringan Segmen ST (ST Slope)", options=["Flat", "Up", "Down"])

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("🔍 Jalankan Analisis Prediksi AI", type="primary", use_container_width=True)

    if btn_predict:
        # 1. Konversi Teks Pilihan User ke Format Data Mentah
        sex_val = 'M' if 'Laki-laki' in sex else 'F'
        cp_val = chest_pain.split(" ")[0]
        fbs_val = 1 if 'Ya' in fasting_bs else 0
        ecg_val = resting_ecg
        angina_val = 'Y' if 'Ya' in exercise_angina else 'N'
        slope_val = st_slope

        # 2. Buat DataFrame Input Pasien
        raw_input = pd.DataFrame([{
            'Age': age,
            'Sex': sex_val,
            'ChestPainType': cp_val,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fbs_val,
            'RestingECG': ecg_val,
            'MaxHR': max_hr,
            'ExerciseAngina': angina_val,
            'Oldpeak': oldpeak,
            'ST_Slope': slope_val
        }])

        # 3. One-Hot Encoding Sesuai Format Training
        encoded_input = pd.get_dummies(raw_input, dtype=int)
        
        # Samakan kolom input dengan kolom saat training
        full_input = pd.DataFrame(0, index=[0], columns=feature_cols)
        for col in encoded_input.columns:
            if col in full_input.columns:
                full_input[col] = encoded_input[col]

        # 4. Feature Scaling & Prediksi
        input_scaled = scaler.transform(full_input)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        # 5. Tampilkan Hasil
        st.divider()
        st.markdown("### 📊 Hasil Prediksi Diagnosa AI")
        
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if prediction == 1:
                st.error("⚠️ **DIAGNOSA: PASIEN BERISIKO TERKENA PENYAKIT JANTUNG**")
                st.write(f"Tingkat Keyakinan Model: **{prob[1]*100:.2f}%**")
                st.warning("💡 **Saran Tindakan:** Pasien disarankan untuk segera melakukan pemeriksaan lanjutan dan konsultasi dengan dokter spesialis jantung.")
            else:
                st.success("✅ **DIAGNOSA: KONDISI PASIEN NORMAL / SEHAT**")
                st.write(f"Tingkat Keyakinan Model: **{prob[0]*100:.2f}%**")
                st.info("💡 **Saran Tindakan:** Pertahankan gaya hidup sehat, pola makan seimbang, serta olahraga teratur.")

        with res_col2:
            st.metric(label="Risiko Penyakit Jantung", value=f"{prob[1]*100:.1f}%")

# =========================================================
# TAB 2: INFORMASI PROJECT, SDGs 3, & ALGORITMA
# =========================================================
with tab2:
    st.markdown("### 🎯 Tentang Project AI Ini")
    
    st.markdown("""
    Project Machine Learning ini dikembangkan untuk mendukung pencapaian **SDGs 3: Good Health and Well-Being (Kehidupan Sehat dan Sejahtera)**, khususnya **Target 3.4**: *Mengurangi angka kematian akibat penyakit tidak menular melalui pencegahan dan penanganan*.
    """)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **🔍 Masalah Kesehatan yang Diselesaikan:**
        Penyakit jantung kardiovaskular merupakan penyebab kematian nomor 1 di dunia. Banyak penderita tidak menyadari gejalanya hingga kondisi memburuk. Model ini hadir sebagai instrumen *skrining awal* yang cepat, efisien, dan tanpa biaya tinggi.
        """)
        
        st.success("""
        **📂 Sumber Dataset:**
        * **Dataset:** Heart Failure Prediction Dataset (Kaggle)
        * **Jumlah Sampel Data:** 918 baris data klinis pasien
        * **Fitur Utama:** Umur, Jenis Kelamin, Kolesterol, Tekanan Darah, Detak Jantung Maksimum, Hasil ECG, dll.
        """)

    with col_info2:
        st.warning("""
        **🤖 Algoritma Machine Learning yang Dibandingkan:**
        Project ini melatih dan membandingkan **6 Algoritma Klasifikasi**:
        1. **Logistic Regression**
        2. **Decision Tree Classifier**
        3. **Random Forest Classifier**
        4. **K-Nearest Neighbors (KNN)**
        5. **Support Vector Machine (SVM)**
        6. **Gaussian Naive Bayes**
        
        *Model terbaik yang memiliki performa akurasi paling optimal dipilih dan digunakan pada aplikasi web ini.*
        """)

    st.divider()
    
    st.markdown("### 👤 Pengembang Project")
    st.markdown("""
    * **Nama Pengembang:** Muhammad Hidayat, Ni Made Santi & Israwati
    * **Peran Project:** AI & Machine Learning Researcher, Developer Web
    * **Lingkungan Pengembangan:** Antigravity IDE, Jupyter Notebook, Python, Streamlit
    """)

# Footer Halaman
st.divider()
st.caption("© 2026 AI for SDGs 3 Project | Dikembangkan untuk Keperluan Tugas Kuliah & Akademik Ilmu Komputer.")