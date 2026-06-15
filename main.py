import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Mematikan pesan warning yang mengganggu di terminal
warnings.filterwarnings("ignore")
data = pd.read_csv("wine-data.csv")

print("[INFO] 5 BARIS PERTAMA PADA DATASET")
print(data.head())
print("[INFO] INFORMASI FILE DATASET")
data.info()

# DIAGRAM DISTRIBUSI KATEGORI
print(f"[INFO] MENGANALISIS DATA DAN MEMBUAT DIAGRAM...")
plt.figure(figsize=(8,5))
plt.title("Distribusi Kategori Wine (WineCategory)")
plt.xlabel("Kategori Wine")
plt.ylabel("Jumlah Dataset")
plt.tight_layout()
sns.countplot(data=data, x="WineCategory", hue="WineCategory", order=["Low", "Average","High"], palette="viridis", legend=False)
plt.savefig("EDA-1-Distribusi_Kategori.png")
plt.close()

# DIAGRAM HEATMAP KORELASI
corr_matrix = data.drop(columns=["WineCategory"]).corr()
plt.figure(figsize=(12,8))
plt.title("Heatmap Korelasi Fitur Kualitas Wine")
plt.tight_layout()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.savefig("EDA-2-Heatmap_Korelasi_Fitur_Kualitas_Wine.png")
plt.close()

# DIAGRAM BOXPLOT ALKOHOL VS KATEGORI WINE
plt.figure(figsize=(8,5))
plt.title("Hubungan Kadar Alkohol vs Kategori Wine")
plt.xlabel("Kategori Wine")
plt.ylabel("Kadar Alkohol")
plt.tight_layout()
sns.boxplot(data=data, x="WineCategory", y="alcohol", hue="WineCategory", order=["Low", "Average", "High"], palette="Set2", legend=False)
plt.savefig("EDA-3-Boxplot_Alkohol.png")
plt.close()
print("[INFO] GRAFIK DIBUAT...")

# Pisahkan Fitur dan Target
# Fitur adalah X, dan Target adalah Y
X = data.drop(columns=["quality", "WineCategory"]) 
le = LabelEncoder()
y = le.fit_transform(data["WineCategory"]) 

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE (Menangani data yang tidak seimbang)
X_train_balanced, Y_train_balanced = SMOTE(random_state=42).fit_resample(X_train_scaled, Y_train)

print(f"[INFO] JUMLAH BARIS DATA TRAINING SEBELUM SMOTE : {len(Y_train)}")
print(f"[INFO] JUMLAH BARIS DATA TRAINING SESUDAH SMOTE : {len(Y_train_balanced)}")

# TRAINING & HYPERPARAMETER TUNING
print("[INFO] MEMULAI PENCARIAN HYPERPARAMETER TERBAIK (TUNING)...")
start_time = time.time()

# ALGORITMA 1: RANDOM FOREST
randomForestParamGrid = {
	"n_estimators": [50,100,200],
	"max_depth": [None, 10,20],
}
randomForestGrid = GridSearchCV(RandomForestClassifier(random_state=42), randomForestParamGrid, cv=5, scoring="accuracy", n_jobs=-1)
randomForestGrid.fit(X_train_balanced, Y_train_balanced)
bestRandomForestModel = randomForestGrid.best_estimator_

# ALGORITMA 2: SUPPORT VECTOR MACHINE (SVM)
supportVectorMachineParamGrid = {
	"C":[0.1,1,10],
	"kernel":["linear", "rbf"]
}
supportVectorMachineGrid = GridSearchCV(SVC(random_state=42), supportVectorMachineParamGrid, cv=5, scoring="accuracy", n_jobs=-1)
supportVectorMachineGrid.fit(X_train_balanced, Y_train_balanced)
bestSupportVectorMachineGridModel = supportVectorMachineGrid.best_estimator_

print(f"PROSES SELESAI DALAM WAKTU {time.time() - start_time:.2f} detik!")
print(f"[RANDOM FOREST] HYPERPARAMETER TERBAIK	: {randomForestGrid.best_params_}")
print(f"[SVM] HYPERPARAMETER TERBAIK			: {supportVectorMachineGrid.best_params_}")

# Tahap 5. Evaluasi & Kesimpulan
randomForestPrediction = bestRandomForestModel.predict(X_test_scaled)
supportVectorMachinePrediction = bestSupportVectorMachineGridModel.predict(X_test_scaled)

# Evaluasi Algoritma Random Forets
print("[INFO] EVALUASI RANDOM FOREST")
print("-"*40)
print(f"Accuracy Score		: {accuracy_score(Y_test, randomForestPrediction):.4f}")
print(f"F1-Score (Weighted)	: {f1_score(Y_test, randomForestPrediction, average='weighted'):.4f}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, randomForestPrediction))
print("\nClassification Report:\n", classification_report(Y_test, randomForestPrediction, target_names=le.classes_))

# Evaluasi Algoritma SVM
print("[INFO] EVALUASI ALGORITMA SUPPORT VECTOR MACHINE (SVM)")
print("-"*40)
print(f"Accuracy Score 		: {accuracy_score(Y_test, supportVectorMachinePrediction):.4f}")
print(f"F1-Score (Weighted) : {f1_score(Y_test, supportVectorMachinePrediction, average='weighted'):.4f}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, supportVectorMachinePrediction))
print("\nClassification Report:\n", classification_report(Y_test, supportVectorMachinePrediction, target_names=le.classes_))
