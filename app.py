# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ----------------------------
# Load Naive Bayes Model
# ----------------------------
nb_model_path = "naive_wine_model.pkl"
features_path = "feature_columns.pkl"
nb_acc_path = "akurasi_wine.pkl"

# Load ID3 Model
id3_model_path = "id3_wine_model.pkl"
id3_acc_path = "id3_accuracy.pkl"

# Validasi file ada
required_files = [nb_model_path, features_path, nb_acc_path, id3_model_path, id3_acc_path]
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File '{f}' tidak ditemukan! Jalankan notebook pelatihan terlebih dahulu.")

# Muat model & data
nb_model = pickle.load(open(nb_model_path, "rb"))
id3_model = pickle.load(open(id3_model_path, "rb"))
feature_columns = pickle.load(open(features_path, "rb"))

# Muat akurasi dan ubah ke persen (dibulatkan 2 desimal)
nb_accuracy = round(pickle.load(open(nb_acc_path, "rb")) * 100, 2)
id3_accuracy = round(pickle.load(open(id3_acc_path, "rb")) * 100, 2)

# ----------------------------
# Routes
# ----------------------------

@app.route('/')
def index():
    # Redirect ke halaman Naive Bayes (atau buat home terpisah)
    return render_template("naive_wine.html", columns=feature_columns, akurasi=nb_accuracy)

@app.route('/naive')
def naive_page():
    return render_template("naive_wine.html", columns=feature_columns, akurasi=nb_accuracy)

@app.route('/id3')
def id3_page():
    return render_template("id3_wine.html", columns=feature_columns, akurasi=id3_accuracy)

# ----------------------------
# Prediksi: Naive Bayes
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = [float(request.form[col]) for col in feature_columns]
        input_array = np.array([input_values])
        prediction = nb_model.predict(input_array)[0]

        return render_template(
            "naive_wine.html",
            columns=feature_columns,
            prediction=int(prediction),
            akurasi=nb_accuracy
        )
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><p>Periksa apakah semua input diisi dengan angka.</p>"

# ----------------------------
# Prediksi: ID3 (Decision Tree)
# ----------------------------
@app.route('/predict_id3', methods=['POST'])
def predict_id3():
    try:
        input_values = [float(request.form[col]) for col in feature_columns]
        input_array = np.array([input_values])
        prediction = id3_model.predict(input_array)[0]

        return render_template(
            "id3_wine.html",
            columns=feature_columns,
            prediction=int(prediction),
            akurasi=id3_accuracy
        )
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><p>Periksa apakah semua input diisi dengan angka.</p>"

# ----------------------------
# Jalankan aplikasi
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)