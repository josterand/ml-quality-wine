# Wine Quality Classification

[ English Version ](#english) | [ Versi Bahasa Indonesia ](#bahasa-indonesia)

---

<a name="english"></a>
## English

### Project Overview
A Supervised Machine Learning project to classify red wine quality based on physicochemical properties. The project addresses class imbalance using SMOTE and evaluates Random Forest against Support Vector Machine (SVM), optimized via GridSearchCV.

### Team Members (Group 2)
- Azis Al Risal
- Bagas Dwi Saputra
- Desta Ega Fatima
- Jonathan Steve Roland
- Nayla Khairusyi Shabrina

### Dataset & Methodology
- **Dataset**: `wine-data.csv` (1,599 records, 11 physicochemical features).
- **Target**: `WineCategory` with 3 classes (`Low`, `Average`, `High`).
- **Pipeline**:
  1. **EDA**: Class distribution analysis, correlation heatmap, and alcohol content boxplots.
  2. **Preprocessing**: Label encoding, stratified train-test split (80:20), feature standardization (`StandardScaler`), and SMOTE oversampling on training data.
  3. **Modeling**: Hyperparameter tuning via `GridSearchCV` (5-fold cross-validation) for Random Forest and SVM.
  4. **Evaluation**: Accuracy score, weighted F1-score, confusion matrix, and classification report.

### Results

| Model | Accuracy | Weighted F1-Score | Status |
|:---|:---:|:---:|:---|
| Random Forest | 85.62% | 86.27% | Best Model |
| Support Vector Machine (SVM) | 75.62% | 78.27% | Baseline Comparison |

Random Forest demonstrated superior performance in capturing non-linear relationships across wine quality grades.

### Setup & Local Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/josterand/ml-quality-wine.git
   cd ml-quality-wine
   ```

2. **Create and activate a virtual environment:**
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project:**
   - Execute the Python script directly:
     ```bash
     python main.py
     ```
   - Or open the Jupyter Notebook:
     ```bash
     jupyter notebook main.ipynb
     ```

---

<a name="bahasa-indonesia"></a>
## Bahasa Indonesia

### Gambaran Proyek
Proyek Machine Learning (Supervised Learning) untuk mengklasifikasikan mutu kualitas wine berdasarkan karakteristik fisikokimia. Proyek ini menangani ketidakseimbangan kelas (*class imbalance*) menggunakan SMOTE dan membandingkan performa Random Forest serta Support Vector Machine (SVM) dengan optimasi GridSearchCV.

### Anggota Kelompok (Kelompok 2)
- Azis Al Risal
- Bagas Dwi Saputra
- Desta Ega Fatima
- Jonathan Steve Roland
- Nayla Khairusyi Shabrina

### Dataset & Metodologi
- **Dataset**: `wine-data.csv` (1.599 baris data, 11 fitur fisikokimia).
- **Target**: `WineCategory` dengan 3 kelas (`Low`, `Average`, `High`).
- **Alur Kerja**:
  1. **EDA**: Analisis distribusi kategori, heatmap matriks korelasi, dan boxplot kadar alkohol.
  2. **Preprocessing**: Label encoding, stratified train-test split (80:20), standarisasi nilai (`StandardScaler`), dan penyeimbangan data latih dengan SMOTE.
  3. **Pemodelan**: Hyperparameter tuning menggunakan `GridSearchCV` (5-fold cross-validation) untuk Random Forest dan SVM.
  4. **Evaluasi**: Accuracy score, weighted F1-score, confusion matrix, dan classification report.

### Hasil Evaluasi

| Model | Akurasi | Weighted F1-Score | Keterangan |
|:---|:---:|:---:|:---|
| Random Forest | 85.62% | 86.27% | Model Terbaik |
| Support Vector Machine (SVM) | 75.62% | 78.27% | Model Pembanding |

Random Forest mencatatkan akurasi dan F1-score tertinggi dalam memprediksi kelas kualitas wine secara konsisten.

### Panduan Instalasi & Menjalankan Lokal

1. **Klon repositori:**
   ```bash
   git clone https://github.com/josterand/ml-quality-wine.git
   cd ml-quality-wine
   ```

2. **Buat dan aktifkan virtual environment:**
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```

3. **Pasang dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan program:**
   - Melalui skrip Python:
     ```bash
     python main.py
     ```
   - Melalui Jupyter Notebook:
     ```bash
     jupyter notebook main.ipynb
     ```
