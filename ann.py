import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib  # Import joblib untuk menyimpan scaler

# Contoh data dummy (fire size, gas level, fire count, label)
data = [
    [0.2, 100, 1, 0],  # Safe
    [1.5, 500, 1, 1],  # Medium
    [2.5, 1000, 1, 2],  # Danger
    [0.8, 200, 2, 1],  # Medium (karena lebih dari 1 api kecil)
    [1.8, 600, 1, 1],  # Medium
    [2.7, 1200, 1, 2],  # Danger
    [0.3, 150, 3, 1],  # Medium (lebih dari 2 api kecil)
    [1.6, 550, 1, 1],  # Medium
    [2.8, 1300, 1, 2],  # Danger
    [0.6, 250, 4, 2],  # Danger (banyak api kecil)
]

# Pisahkan fitur (X) dan label (y)
X = np.array([[row[0], row[1], row[2]] for row in data])  # fire size, gas level, fire count
y = np.array([row[3] for row in data])  # 0: Safe, 1: Medium, 2: Danger

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ubah label menjadi one-hot encoding
y_categorical = to_categorical(y, num_classes=3)

# Bangun model ANN
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))  # 3 input neurons: fire size, gas level, fire count
model.add(Dense(10, activation='relu'))  # Hidden layer 1
model.add(Dense(3, activation='softmax'))  # Output layer: 3 classes (Safe, Medium, Danger)

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model dengan data dummy
model.fit(X_scaled, y_categorical, epochs=500, batch_size=2)

# Simpan model dan scaler untuk digunakan di program utama
model.save('fire_hazard_ann_model_v2.h5')
joblib.dump(scaler, 'scaler_v2.pkl')
