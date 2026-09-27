# Project Machine Learning - SDGs 3: Good Health and Well-Being
## Prediksi Risiko Penyakit Jantung menggunakan Algoritma Klasifikasi

Proyek ini dikembangkan untuk memenuhi tugas kelompok Machine Learning dengan berfokus pada penerapan kecerdasan buatan untuk mendukung capaian **Sustainable Development Goals (SDGs) 3: Good Health and Well-Being**.

---

## 👥 Anggota Kelompok (Kelompok 3)

* **Muhammad Hidayat** (NIM: F1G125043)
* **Ni Made Santi Wardani** (NIM: F1G125014)
* **Israwati** (NIM: F1G125009)

**Program Studi:** Ilmu Komputer  
**Fakultas:** Matematika dan Ilmu Pengetahuan Alam  
**Instansi:** Universitas Halu Oleo  

---

## 📌 Latar Belakang & Keterkaitan SDGs

* **Topik SDGs:** Target 3.4 — Mengurangi sepertiga dari kematian dini akibat penyakit tidak menular melalui pencegahan dan pengobatan, serta meningkatkan kesehatan mental dan kesejahteraan.
* **Masalah:** Penyakit jantung merupakan salah satu penyebab utama kematian dini di seluruh dunia. Penanganan dini melalui deteksi risiko berbasis data klinis sangat krusial untuk mencegah komplikasi fatal.
* **Solusi:** Membangun model prediksi risiko penyakit jantung berbasis machine learning untuk membantu penapisan awal risiko kesehatan pasien.

---

## 🎯 Tujuan Proyek

1. Mengembangkan model klasifikasi machine learning untuk memprediksi risiko penyakit jantung pasien berdasarkan fitur-fitur klinis.
2. Membandingkan performa dari 6 algoritma machine learning:
   * **Logistic Regression**
   * **K-Nearest Neighbors (KNN)**
   * **Naive Bayes**
   * **Random Forest**
   * **Support Vector Machine (SVM)**
   * **Decision Tree**
3. Menentukan algoritma terbaik berdasarkan metrik evaluasi (Accuracy, Precision, Recall, dan F1-Score).

---

## 📊 Dataset

* **Nama Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle)
* **Deskripsi:** Dataset ini menggabungkan 5 himpunan data penyakit jantung yang berbeda dengan 11 fitur klinis yang digunakan untuk memprediksi kemungkinan pasien mengalami penyakit jantung.

---

## 📈 Hasil Evaluasi Model

Berikut adalah tabel hasil perbandingan performa 6 model Machine Learning yang dievaluasi:

| No | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 1 | **Logistic Regression** | **0.896739** | **0.895238** | **0.921569** | **0.908213** |
| 2 | **K-Nearest Neighbors** | 0.885870 | 0.893204 | 0.901961 | 0.897561 |
| 3 | **Naive Bayes** | 0.875000 | 0.907216 | 0.862745 | 0.884422 |
| 4 | **Random Forest** | 0.875000 | 0.891089 | 0.882353 | 0.886700 |
| 5 | **Support Vector Machine** | 0.853261 | 0.850467 | 0.892157 | 0.870813 |
| 6 | **Decision Tree** | 0.777174 | 0.790476 | 0.813725 | 0.801932 |

### 💡 Kesimpulan Performa:
* **Logistic Regression** memperoleh performa terbaik di antara seluruh model dengan **Akurasi 89.67%**, **Recall 92.16%**, dan **F1-Score 90.82%**.
* Model **K-Nearest Neighbors (KNN)** menempati posisi kedua terbaik dengan Akurasi 88.59% dan F1-Score 89.76%.
* High Recall pada Logistic Regression sangat menguntungkan dalam konteks medis (SDGs 3) karena dapat meminimalisir kesalahan deteksi pada pasien yang sebenarnya berisiko tinggi (*False Negative*).

---

## 🛠️ Teknologi & Library yang Digunakan

* **Bahasa Pemrograman:** Python
* **Library Utama:**
  * `Pandas` & `NumPy` — Data Manipulation & Processing
  * `Matplotlib` & `Seaborn` — Data Visualization
  * `Scikit-Learn` — Model Building, Evaluation, & Pipeline

---

## 📁 Struktur Repositori

```text
.
├── datasets/              # Dataset mentah (heart.csv)
├── notebooks/             # Jupyter Notebook (EDA & Pemodelan)
├── models/                # Models tersimpan
├── venv/                  # Virtual environtment
├── README.md              # Dokumentasi proyek
├── app.py                 # File menjalankan sistem
└── requirements.txt       # Daftar pustaka / dependensi Python
