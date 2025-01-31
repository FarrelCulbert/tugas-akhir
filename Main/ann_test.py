import numpy as np
from tensorflow.keras.models import load_model
import joblib  # Untuk memuat scaler

# Load model ANN dan scaler
model_path = "ann6.h5"  # Ganti dengan path model .h5 kamu
scaler_path = "ann6_skaler.pkl"  # Ganti dengan path scaler .pkl kamu

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Fungsi untuk memprediksi hazard level
def predict_hazard(fire_size, gas_level, fire_count):
    # Normalisasi input menggunakan scaler
    input_data = np.array([[fire_size, gas_level, fire_count]])
    input_scaled = scaler.transform(input_data)

    # Prediksi menggunakan model ANN
    prediction = model.predict(input_scaled)
    hazard_level = np.argmax(prediction, axis=1)[0]  # Ambil index prediksi tertinggi

    # Interpretasi hasil prediksi
    if hazard_level == 0:
        return "Aman"
    elif hazard_level == 1:
        return "Waspada"
    elif hazard_level == 2:
        return "Bahaya"
    else:
        return "Unknown"

# Program utama
if __name__ == "__main__":
    print("===== Manual Hazard Prediction =====")
    print('Ketik "q" kapan saja untuk keluar.\n')

    while True:
        try:
            # Input manual dari user
            fire_size_input = input("Enter Fire Size (m): ")
            if fire_size_input.lower() == "q":
                break
            fire_size = float(fire_size_input)

            gas_level_input = input("Enter Gas Level (ppm): ")
            if gas_level_input.lower() == "q":
                break
            gas_level = int(gas_level_input)

            fire_count_input = input("Enter Fire Count: ")
            if fire_count_input.lower() == "q":
                break
            fire_count = int(fire_count_input)

            # Prediksi hazard level
            hazard_level = predict_hazard(fire_size, gas_level, fire_count)
            print(f"\nPrediction Result: {hazard_level}\n")

        except Exception as e:
            print(f"Error: {e}")
            print("Silakan coba lagi.\n")

    print("Program selesai. Terima kasih!")
