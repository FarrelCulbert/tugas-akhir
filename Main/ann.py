import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib  # Untuk menyimpan scaler

# Load data dari file Excel
file_path = 'generated_hazard_data.xlsx'  # Ganti dengan path file Excel Anda
df = pd.read_excel(file_path)

# Pisahkan fitur (X) dan label (y)
X = df[['Fire Size', 'Gas Level', 'Fire Count']].values
y = df['Hazard Level'].values

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ubah label menjadi one-hot encoding
y_categorical = to_categorical(y, num_classes=3)

# Pisahkan data menjadi training, validation, dan testing (60%-20%-20%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_categorical, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Bangun model ANN
model = Sequential()
model.add(Dense(8, input_dim=3, activation='relu'))  # 3 input neurons: fire size, gas level, fire count
model.add(Dropout(0.1))  # Dropout untuk mengurangi overfitting
model.add(Dense(16, activation='relu'))  # Hidden layer 1
model.add(Dense(16, activation='relu'))  # Hidden layer 2
model.add(Dense(8, activation='relu'))  # Hidden layer 3
model.add(Dense(3, activation='softmax'))  # Output layer: 3 classes (Safe, Medium, Danger)

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Tambahkan EarlyStopping untuk menghentikan pelatihan jika tidak ada peningkatan
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Latih model dengan data training dan validasi
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=4,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluasi model pada data testing
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Testing Accuracy: {test_accuracy}, Testing Loss: {test_loss}")

# **Tambahan** evaluasi ulang pada training set untuk memastikan model sepenuhnya terkompilasi
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=1)
print(f"Training Accuracy: {train_accuracy}, Training Loss: {train_loss}")

# Visualisasi Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('training_validation_loss_from_excel.png')  # Save the loss plot
plt.show()

# Visualisasi Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('training_validation_accuracy_from_excel.png')  # Save the accuracy plot
plt.show()

# Simpan model dan scaler untuk digunakan di program utama
model.save('fire_hazard_ann_model_from_excel.h5')
joblib.dump(scaler, 'scaler_from_excel.pkl')

# **Tambahan** untuk memuat ulang model dan evaluasi untuk debugging
reloaded_model = load_model('fire_hazard_ann_model_from_excel.h5')
reloaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model reloaded and compiled successfully.")
