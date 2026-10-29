# 🔬 Breast Cancer Wisconsin Classification & Clinical Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

Proyek *end-to-end* Machine Learning dan web dashboard interaktif untuk klasifikasi diagnosis kanker payudara (**Jinak / Benign** vs **Ganas / Malignant**) berdasarkan *Breast Cancer Wisconsin Dataset*. 

Proyek ini mencakup **audit mendalam 10 masalah kritis EDA**, pemodelan **Random Forest Classifier**, eksperimen reduksi dimensi menggunakan **Principal Component Analysis (PCA)**, serta pengerjaan **UI/UX Dashboard klinis modern**.

---

## 📌 Ringkasan Hasil & Metrik Utama

| Metrik Pemodelan | Baseline Model (30 Fitur Asli) | Optimized + PCA (10 Komponen Utama) |
| :--- | :---: | :---: |
| **Akurasi Tes** | **95.61%** | **92.11%** |
| **ROC-AUC Score** | **0.9850** | **0.9785** |
| **F1-Score (Jinak)** | 0.96 | 0.94 |
| **F1-Score (Ganas)** | 0.93 | 0.89 |
| **Variansi Data Dipertahankan** | 100% | **95.67%** (Distorsi < 4.34%) |
| **Jumlah Fitur Input** | 30 Fitur Geometris | **10 Komponen Terkompresi** |

> **Kesimpulan Klinis:** Pemangkasan fitur dari 30 menjadi 10 komponen via PCA menyebabkan *trade-off* penurunan akurasi yang relatif kecil (~3%), tetapi secara drastis menghilangkan multikolinearitas ekstrem ($VIF > 250$) dan menghasilkan model inferensi yang jauh lebih ringan, efisien, dan cepat untuk kebutuhan *web dashboard*.

---

## 📊 Bagian 1: Audit 10 Masalah Kritis EDA & Mitigasi Medis

Sebelum tahap pemodelan, data mentah diaudit secara ketat untuk mendeteksi distorsi atau kebocoran statistik:

1. **Data Leakage (Kebocoran Data):** 
   * *Masalah:* Kalkulasi statistik scaling pada seluruh dataset sebelum *data-split* merembeskan informasi data uji ke data latih.
   * *Mitigasi:* Mengisolasi proses rekayasa fitur di dalam `Scikit-Learn Pipeline`.
2. **Skewed Data Scaling:**
   * *Masalah:* Fitur *skewed* tinggi (seperti `area error`).
   * *Mitigasi:* Transformasi logaritmik (`np.log1p`) & `RobustScaler`.
3. **Missing Value Audit:**
   * *Hasil:* Total *missing value* 0.0% (dataset bersih tanpa perlu imputasi).
4. **Multikolinearitas Ekstrem ($VIF > 250$):**
   * *Masalah:* Fitur ukuran geometris (`mean radius`, `mean perimeter`, `mean area`) saling berkorelasi linear kuat.
   * *Mitigasi:* Algoritma berbasis Tree (*Random Forest*) yang robust terhadap redundansi fitur.
5. **Class Imbalance:**
   * *Distribusi:* 357 tumor Jinak vs 212 kanker Ganas.
   * *Mitigasi:* Evaluasi berbasis `StratifiedKFold` dan metrik **ROC-AUC**.
6. **Bias Linearitas (Pearson vs. Mutual Information):**
   * Membandingkan korelasi linear Pearson dengan *Mutual Information* untuk menangkap pola non-linear pada fitur kompleks (seperti `smoothness error`).
7. **Sanity Check kardinalitas ID:**
   * Verifikasi unik untuk mencegah *overfitting* akibat kolom indeks buatan.
8. **Masking Effect & True Positive Signal (Outlier Klinis):**
   * *Temuan Penting:* Deteksi outlier pada `worst area` menunjukkan bahwa 100% data outlier bernilai ekstrim adalah sel **Kanker Ganas (Malignant)**.
   * *Tindakan:* Outlier **dilarang dihapus** karena merupakan sinyal klinis vital (*True Positive Signal*).

---

## ⚙️ Bagian 2: Baseline Model & Hyperparameter Tuning

- **Algoritma:** Random Forest Classifier
- **Teknik Tuning:** `GridSearchCV` (5-Fold Cross Validation)
- **Parameter Diuji:** `n_estimators`, `max_depth`, `min_samples_split`.
- **Evaluasi Matrix (114 Data Uji):**
  - **Terprediksi Benar:** 108 Data
  - **Kesalahan Prediksi:** 6 Data (3 False Positive, 3 False Negative)

---

## 📐 Bagian 3: Reduksi Dimensi dengan PCA

- **Tujuan:** Meringkas 30 fitur menjadi **10 Komponen Utama (PC1–PC10)** untuk menghilangkan multikolinearitas tanpa kehilangan informasi penting.
- **Kumulatif Variansi:** **95.67%** dipertahankan dengan `n_components=0.95`.
- **Evaluasi Matrix Setelah PCA (114 Data Uji):**
  - **Terprediksi Benar:** 105 Data
  - **Kesalahan Prediksi:** 9 Data (5 False Positive, 4 False Negative)
  - **ROC-AUC:** **0.9785**

---

## 💻 Visualisasi & Interactive Web Dashboard

Proyek ini dilengkapi frontend dashboard interaktif (**OncoScan**):

- 🎨 **Minimalist UI Design:** Tata letak bersih, konsistensi *spacing*, dan tombol interaktif yang responsif.
- ⚡ **Real-Time Prediction Form:** Form parameter klinis interaktif dengan slider dan nilai indikator normal.
- 📁 **Multi-input Data Support:** 
  - Tombol **Preset Sample Pasien** (Jinak vs Ganas) untuk uji coba instan.
  - Fitur **Drag & Drop Upload File CSV/JSON** berbasis *FileReader API*.
- 📈 **Dynamic ROC-AUC Chart:** Kurva ROC interaktif dengan performa skor AUC 0.9785.

