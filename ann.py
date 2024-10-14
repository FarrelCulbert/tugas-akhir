from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example data used to train the model
data = [
    [0.2, 500, "Safe"],
    [1.5, 1000, "Medium"],
    [2.5, 2000, "Danger"],
    [1.0, 800, "Medium"],
    [2.8, 1800, "Danger"],
    [0.3, 400, "Safe"],
    [0.5, 450, "Safe"],
    [1.7, 950, "Medium"],
    [2.6, 1950, "Danger"],
    [0.4, 600, "Safe"]
]

# Prepare the dataset (input features and labels)
X = [[row[0], row[1]] for row in data]  # fire size, gas level
y = [row[2] for row in data]  # hazard levels

# Scale the data (normalize input values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build and train the ANN model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(X_scaled, y)

# Function to predict hazard level
def predict_hazard(fire_size, gas_level):
    # Normalize the input using the same scaler as before
    input_data = scaler.transform([[fire_size, gas_level]])
    
    # Predict hazard level
    prediction = mlp.predict(input_data)
    
    # Return the predicted hazard level
    return prediction[0]

# Main loop to accept user input and predict hazard level
while True:
    # Get user input for fire size and gas level
    fire_size = float(input("Enter fire size (e.g., 0.5 for small, 2.5 for large): "))
    gas_level = float(input("Enter gas level (e.g., 500 for low gas, 2000 for high gas): "))

    # Predict hazard level
    hazard_level = predict_hazard(fire_size, gas_level)

    # Output the hazard level
    print(f"Predicted Hazard Level: {hazard_level}")

    # Option to exit the loop
    cont = input("Do you want to enter another value? (yes/no): ")
    if cont.lower() != 'yes':
        break
