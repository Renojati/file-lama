from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

# Load model dari file .pkl
with open('naive_bayes_model_fix.pkl', 'rb') as file:
    model = pickle.load(file)

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)

# Route untuk halaman "index"
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk halaman "aplikasi"
@app.route('/aplikasi')
def aplikasi():
    return render_template('aplikasi.html')

# Route untuk halaman "informasi"
@app.route('/informasi')
def informasi():
    return render_template('informasi.html')

# Route untuk halaman "kelompok"
@app.route('/kelompok')
def kelompok():
    return render_template('kelompok.html')

# Route untuk prediksi curah hujan
@app.route('/predict', methods=['POST'])
def predict_rain():
    try:
        # Ambil data JSON dari permintaan
        data = request.get_json()
        Tn = float(data['Tn'])
        Tx = float(data['Tx'])
        Tavg = float(data['Tavg'])
        RH_avg = float(data['RH_avg'])
        ss = float(data['ss'])
        ff_avg = float(data['ff_avg'])

        # Format data untuk prediksi
        input_data = pd.DataFrame({
            'Tn': [Tn],
            'Tx': [Tx],
            'Tavg': [Tavg],
            'RH_avg': [RH_avg],
            'ss': [ss],
            'ff_avg': [ff_avg]
        })

        # Lakukan prediksi
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            result = "Ya"
            message = "Hari ini sedang hujan, pastikan ketika sedang beraktivitas sedia jas hujan dan payung yaa."
        else:
            result = "Tidak"
            message = "Cuaca cerah nih, selamat menjalankan aktivitas tanpa khawatir terkena hujan."

        return jsonify({
            'success': True,
            'prediction': result,
            'message': message
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
