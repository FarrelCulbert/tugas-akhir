import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler
import serial  # For serial communication with Arduino
from collections import Counter  # For majority voting

# Load YOLOv8 model pre-trained on detecting fire
model = YOLO('yolo11n.pt')  # Use your custom-trained YOLOv8 model for fire

# Load pre-trained ANN model and scaler
mlp_model = load_model('ann6.h5')
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Recompile model to avoid warning
scaler = joblib.load('ann6_skaler.pkl')

# Set up serial communication with Arduino (for gas sensor)
arduino = serial.Serial('COM3', 9600, timeout=1)  # Change COM6 to your correct port

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the RealSense pipeline
pipeline.start(config)

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Fire size classification thresholds in meters
SMALL_THRESHOLD = 0.2
LARGE_THRESHOLD = 3.5

# Variables to store last values
last_hazard_level = "Safe"
last_gas_value = None
fire_count = 0

# Function to classify fire size based on real-world height
def classify_fire_size(real_height):
    if real_height < SMALL_THRESHOLD:
        return "Kecil"
    elif SMALL_THRESHOLD <= real_height < LARGE_THRESHOLD:
        return "Sedang"
    else:
        return "Besar"

# Function to calculate real-world height
def calculate_real_height(bounding_box_height, depth):
    CAMERA_FOV_VERTICAL = 37  # Field of view for vertical axis in degrees (adjust based on camera spec)
    FRAME_HEIGHT_PIXELS = 480  # Camera frame height in pixels
    fov_vertical_rad = np.radians(CAMERA_FOV_VERTICAL)
    real_world_height = 2 * depth * np.tan(fov_vertical_rad / 2) * (bounding_box_height / FRAME_HEIGHT_PIXELS)
    return real_world_height

# Function to read gas data from Arduino
def read_gas_data():
    if arduino.in_waiting > 0:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            try:
                gas_value = int(data.split(":")[1].strip())  # Assume gas value format: 'nilai_gas: 500'
                print(f"Gas Sensor Data: {gas_value}")  # Tambahkan ini untuk melihat data yang diterima
                return gas_value
            except (ValueError, IndexError):
                return None
    return None

# Function to predict hazard level using ANN model with additional variable (fire count)
def predict_hazard(fire_size, gas_level, fire_count):
    input_data = scaler.transform([[fire_size, gas_level, fire_count]])  # Normalize input
    prediction = mlp_model.predict(input_data)
    hazard_level = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
    if hazard_level == 0:
        return "Aman"
    elif hazard_level == 1:
        return "Waspada"
    else:
        return "Bahaya"

# VideoWriter setup to save the output in MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('uji10.mp4', fourcc, 20.0, (640, 480))  # Save output

# Main loop to process frames and predict hazard
try:
    while True:
        # Get frames from the RealSense pipeline
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Perform YOLOv8 inference on the color image
        results = model(color_image)

        # Initialize variables for this frame
        current_fire_count = 0
        fire_sizes = []
        current_hazard_level = None

        # Get gas sensor data from Arduino
        gas_value = read_gas_data()
        if gas_value is not None:
            last_gas_value = gas_value  # Update the last gas value if new data is available

        # Loop through detection results
        for result in results:
            boxes = result.boxes  # Get bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
                class_id = int(box.cls[0])  # Get the class ID

                # Check if the detected object is fire
                if class_id == 0:  # Assuming class_id == 0 is fire
                    current_fire_count += 1  # Increment fire count

                    # Draw bounding box around the fire
                    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Calculate fire size (real-world height)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    distance = depth_frame.get_distance(center_x, center_y)
                    bounding_box_height = y2 - y1
                    fire_height = calculate_real_height(bounding_box_height, distance)

                    # Classify fire size category
                    fire_size = classify_fire_size(fire_height)
                    fire_sizes.append(fire_size)  # Append fire size to the list

        # Determine the overall fire size using majority voting
        if fire_sizes:
            overall_fire_size = Counter(fire_sizes).most_common(1)[0][0]  # Get the most common size
        else:
            overall_fire_size = "Tidak Ada"

        # Update fire count and hazard level based on detections
        fire_count = current_fire_count
        if fire_count > 0 and last_gas_value is not None:
            # Predict hazard level using ANN
            current_hazard_level = predict_hazard(1.0 if overall_fire_size == "Besar" else 0.5, last_gas_value, fire_count)
            last_hazard_level = current_hazard_level  # Update hazard level
        else:
            last_hazard_level = "Aman"

        # Display the fire count, fire size, gas value, and hazard level statically at the top-left corner
        cv2.putText(color_image, f"Fire Count: {fire_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(color_image, f"Overall Fire Size: {overall_fire_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if last_gas_value is not None:
            cv2.putText(color_image, f"Gas Value: {last_gas_value}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(color_image, f"Hazard Level: {last_hazard_level}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save the frame to the MP4 file
        out.write(color_image)

        # Display the final image with bounding boxes and annotations
        cv2.imshow('Fire Detection and Hazard Classification', color_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Stop the pipeline and release resources
    pipeline.stop()
    arduino.close()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()
