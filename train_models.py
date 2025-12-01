# =====================================================
# TRAINING MODEL: NAIVE BAYES vs ID3 (Decision Tree)
# Dataset: Wine Quality
# Model terbaik disimpan sebagai best_model.pkl
# =====================================================

import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ======================================
# 1. LOAD DATASET
# ======================================
df = pd.read_csv('WineQT.csv')
X = df.drop(columns=['quality', 'Id'])
y = df['quality']

print("Jumlah Data:", len(df))
print(df.info())
print(df['quality'].value_counts())

# ======================================
# 2. SPLIT DATA
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ======================================
# 3. TRAIN NAIVE BAYES
# ======================================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

print("\n=== Evaluasi Naive Bayes ===")
print("Akurasi:", nb_acc)
print(classification_report(y_test, nb_pred))

# ======================================
# 4. TRAIN DECISION TREE (ID3)
# ======================================
dt_model = DecisionTreeClassifier(criterion='entropy')  # agar benar-benar ID3
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("\n=== Evaluasi ID3 (Decision Tree) ===")
print("Akurasi:", dt_acc)
print(classification_report(y_test, dt_pred))

# ======================================
# 5. PILIH MODEL TERBAIK
# ======================================
if nb_acc > dt_acc:
    best_model = nb_model
    best_name = "Naive Bayes"
else:
    best_model = dt_model
    best_name = "ID3 (Decision Tree)"

print("\n=== MODEL TERBAIK ===")
print("Model :", best_name)
print("Akurasi :", max(nb_acc, dt_acc))

# ======================================
# 6. SIMPAN MODEL TERBAIK
# ======================================
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Simpan nama fitur → penting untuk Flask
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel berhasil disimpan sebagai best_model.pkl")
print("Daftar fitur disimpan sebagai feature_columns.pkl")
