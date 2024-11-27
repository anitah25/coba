from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("rfc_fix.pkl", "rb"))

app.config["DEBUG"] = True

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Mengambil data dari form dan mengonversinya ke tipe float
    nama = request.form['nama']
    age = float(request.form['umur'])  # Mengubah nama fitur 'umur' menjadi 'age'
    gender = 1 if request.form['gender'] == 'laki-laki' else 0
    hypertension = 1 if request.form['hipertensi'] == 'ya' else 0
    heart_disease = 1 if request.form['penyakit_jantung'] == 'ya' else 0
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['hba1c'])
    blood_glucose_level = float(request.form['gula_darah'])
    smoking_history = request.form['riwayat_merokok']
    
    # Encoding riwayat merokok
    if smoking_history == "tidak_pernah":
        smoking_history_encoded = 0
    elif smoking_history == "mantan":
        smoking_history_encoded = 1
    elif smoking_history == "pernah":
        smoking_history_encoded = 2
    elif smoking_history == "saat_ini":
        smoking_history_encoded = 3
    elif smoking_history == "bukan_saat_ini":
        smoking_history_encoded = 4
    elif smoking_history == "tidak_ada_info":
        smoking_history_encoded = -1
    else:
        smoking_history_encoded = -1
    # Membuat array dari data input
    features = np.array([[age, gender, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, smoking_history_encoded]])

    # Melakukan prediksi dengan model
    prediction = model.predict(features)
# Mapping hasil prediksi ke label teks
    if prediction[0] == 1:
        result = "Diabetes"
    else:
        result = "Tidak Terkena Diabetes"

    # Mengembalikan hasil prediksi ke template
    return render_template("index.html", prediction_text="Prediksi: {}".format(result), name=nama)


if __name__ == '__main__':
    app.run(debug=True)
