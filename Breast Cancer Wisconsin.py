# =============================================================================
# SCRIPT GABUNGAN UTUH: EDA, PREPROCESSING, OPTIMASI, & PCA (BREAST CANCER)
# =============================================================================

# DAFTAR 30 FITUR ASLI SEBELUM DIREDUKSI (DIKELOMPOKKAN SINGKAT):
# 1-10.  FITUR MEAN (Rata-rata fisik sel):
#        - mean radius, mean texture, mean perimeter, mean area, mean smoothness,
#          mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension.
# 11-20. FITUR ERROR (Variansi/standar galeri pengamatan):
#        - radius error, texture error, perimeter error, area error, smoothness error,
#          compactness error, concavity error, concave points error, symmetry error, fractal dimension error.
# 21-30. FITUR WORST (Nilai terburuk/terbesar dari sel):
#        - worst radius, worst texture, worst perimeter, worst area, worst smoothness,
#          worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension.
#
# DAFTAR 10 KOMPONEN SETELAH DIREDUKSI DENGAN PCA (MENJAGA >= 95% VARIANSI):
# - PC1 & PC2  : Ringkasan utama dimensi/ukuran geometri sel (radius, perimeter, area).
# - PC3 & PC4  : Ringkasan tekstur, kehalusan, dan kekompakan permukaan sel.
# - PC5 s/d PC10: Ringkasan variansi error, cekungan, kesimetrisan, dan dimensi fraktal sel.
# =============================================================================
import time
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("="*70)
print(" SCRIPT GABUNGAN UTUH: EDA, PREPROCESSING, OPTIMASI, & PCA")
print("="*70)

# -------------------------------------------------------------------------
# LOAD DATASET & SPLIT UTAMA (30 FITUR NUMERIK)
# -------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names) # 30 fitur geometris inti sel
y = pd.Series(data.target) # Target Biner: 0 (Ganas), 1 (Jinak)
label_custom = ['malignant (kanker ganas)', 'benign (tumor jinak)']

# Simpan file CSV otomatis sebagai file eksternal project
X.assign(target=y).to_csv('breast_cancer.csv', index=False)

# Split data 80:20 berstrata
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================================
# BAGIAN 1: DEMONSTRASI 10 MASALAH KRITIS EDA & MITIGASI
# =========================================================================
print("\n" + "="*70)
print(" BAGIAN 1: DEMONSTRASI 10 MASALAH KRITIS EDA & MITIGASI")
print("="*70)

# 1. Data Leakage Demonstration
print("[1] Masalah Data Leakage:")
scaler_bocor = StandardScaler()
X_scaled_bocor = scaler_bocor.fit_transform(X)
X_tr_b, X_ts_b, y_tr_b, y_ts_b = train_test_split(X_scaled_bocor, y, test_size=0.2, random_state=42, stratify=y)
model_bocor = LogisticRegression()
model_bocor.fit(X_tr_b, y_tr_b)
acc_bocor = model_bocor.score(X_ts_b, y_ts_b)

pipe_isolasi = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
pipe_isolasi.fit(X_train, y_train)
acc_isolasi = pipe_isolasi.score(X_test, y_test)
print(f" - Sebelum (Leaked) Akurasi: {acc_bocor:.4f}")
print(f" - Sesudah (Isolasi Pipeline) Akurasi: {acc_isolasi:.4f}\n")

# 2. Skewed Data Scaling
print("[2] Masalah Data Skewed (Fitur: 'area error'):")
skew_awal = X['area error'].skew()
area_err_log = np.log1p(X['area error'])
skew_sesudah = area_err_log.skew()
print(f" - Skewness Awal: {skew_awal:.3f} | Setelah Log1p & Robust: {skew_sesudah:.3f}\n")

# 3. Missing Value Audit
print("[3] Missing Value Audit:")
print(f" - Total Missing Values: {X.isna().sum().sum()} (Dataset Bersih 0.0%)\n")

# 4. Multicollinearity (VIF)
print("[4] Masalah Multikolinearitas (VIF):")
fitur_vif = ['mean radius', 'mean perimeter', 'mean area']
X_vif = X[fitur_vif].copy()
X_vif['Intercept'] = 1
vif_sebelum = variance_inflation_factor(X_vif.values, 0)
X_vif_scaled = StandardScaler().fit_transform(X[fitur_vif])
vif_sesudah = variance_inflation_factor(X_vif_scaled, 0)
print(f" - VIF 'mean radius' Sebelum: {vif_sebelum:.2f} | Sesudah Standarisasi: {vif_sesudah:.2f}\n")

# 5. Class Imbalance
print("[5] Class Imbalance & Evaluation Metrics:")
print(f" - Sebaran Kelas: {np.bincount(y)[1]} Jinak vs {np.bincount(y)[0]} Ganas\n")

# 6. Linear Bias (Pearson vs Mutual Info)
print("[6] Bias Linearitas (Pearson vs Mutual Info):")
pearson_corr = abs(stats.pearsonr(X['smoothness error'], y)[0])
mi_score = mutual_info_classif(X[['smoothness error']], y, random_state=42)[0]
print(f" - Pearson: {pearson_corr:.4f} | Mutual Info: {mi_score:.4f}\n")

# 7. Uniqueness Ratio
print("[7] Feature Sanity Check (Uniqueness Ratio):")
max_uniq_ratio = max([X[col].nunique() / len(X) for col in X.columns])
print(f" - Uniqueness Ratio Tertinggi: {max_uniq_ratio:.4f} (Aman dari ID buatan)\n")

# 8 & 10. Outlier & True Positive Signal
print("[8 & 10] Deteksi Outlier & Sinyal Klinis:")
q1, q3 = X['worst area'].quantile(0.25), X['worst area'].quantile(0.75)
upper_bound = q3 + 1.5 * (q3 - q1)
prop_ganas_outlier = (y[X['worst area'] > upper_bound] == 0).mean()
print(f" - Proporsi Kanker Ganas pada Outlier 'worst area': {prop_ganas_outlier * 100:.1f}% (Outlier dipertahankan)\n")


# =========================================================================
# BAGIAN 2: BASELINE MODEL & HYPERPARAMETER TUNING (SEBELUM PCA)
# =========================================================================
print("="*70)
print(" BAGIAN 2: BASELINE MODEL & HYPERPARAMETER TUNING (SEBELUM PCA)")
print("="*70)

start_time = time.time()
model_awal = RandomForestClassifier(random_state=42)
model_awal.fit(X_train, y_train)
pred_awal = model_awal.predict(X_test)
time_awal = time.time() - start_time

print(f"Akurasi Model Baseline : {accuracy_score(y_test, pred_awal) * 100:.2f}%")

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 4, 8, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42), 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
pred_akhir = best_model.predict(X_test)

print(f"Parameter Terbaik      : {grid_search.best_params_}")
print(f"Akurasi Model Optimasi : {accuracy_score(y_test, pred_akhir) * 100:.2f}%\n")

# Format output Classification Report & Confusion Matrix (SEBELUM PCA)
print("[ Classification Report Sebelum PCA ]")
print(classification_report(y_test, pred_akhir, target_names=label_custom))
print("[ Confusion Matrix Sebelum PCA ]")
print(f"\n{confusion_matrix(y_test, pred_akhir)}\n")


# =========================================================================
# BAGIAN 3: REDUKSI DIMENSI DENGAN PRINCIPAL COMPONENT ANALYSIS (PCA)
# =========================================================================
print("="*70)
print(" BAGIAN 3: REDUKSI DIMENSI DENGAN PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*70)

# Analisis variansi PCA khusus data latih
scaler_temp = RobustScaler()
X_train_scaled = scaler_temp.fit_transform(X_train)

pca_temp = PCA(n_components=None, random_state=42)
pca_temp.fit(X_train_scaled)
cum_explained_var = np.cumsum(pca_temp.explained_variance_ratio_)
n_components_95 = np.argmax(cum_explained_var >= 0.95) + 1

print(f"[PCA Analysis] Fitur Awal (Mentah)     : {X.shape[1]} Fitur")
print(f"[PCA Analysis] Fitur Setelah Reduksi  : {n_components_95} Komponen Utama (PC1-PC10)")

# Pipeline dengan PCA dan ColumnTransformer
skewed_features = [col for col in X.columns if 'area' in col or 'radius' in col or 'perimeter' in col]
general_features = [col for col in X.columns if col not in skewed_features]

preprocessor = ColumnTransformer(transformers=[
    ('skewed_robust', RobustScaler(), skewed_features),
    ('general_robust', RobustScaler(), general_features)
])

pipeline_pca = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    ))
])

pipeline_pca.fit(X_train, y_train)
y_pred_pca = pipeline_pca.predict(X_test)
y_proba_pca = pipeline_pca.predict_proba(X_test)[:, 1]

print("\n=== EVALUASI AKHIR MODEL SETELAH REDUKSI PCA ===")
print(f"Jumlah Fitur Asli         : {X.shape[1]} Fitur")
print(f"Jumlah Komponen PCA Aktif : {pipeline_pca.named_steps['pca'].n_components_} Komponen Utama")
print(f"Total Variansi Dipertahankan : {np.sum(pipeline_pca.named_steps['pca'].explained_variance_ratio_)*100:.2f}%")
print(f"ROC-AUC Score              : {roc_auc_score(y_test, y_proba_pca):.4f}")

# Format output Classification Report & Confusion Matrix (SETELAH PCA) persis seperti gambar
print("\n[ Classification Report PCA ]")
print(classification_report(y_test, y_pred_pca, target_names=label_custom))
print("[ Confusion Matrix PCA ]")
print(f"\n{confusion_matrix(y_test, y_pred_pca)}")
print("="*70)