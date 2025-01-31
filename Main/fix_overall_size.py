import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler
import serial  # For serial communication with Arduino

# Load YOLOv8 model pre-trained on detecting fire
model = YOLO('model10i.pt')  # Use your custom-trained YOLOv8 model for fire

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
SMALL_THRESHOLD = 0.6
LARGE_THRESHOLD = 3.5

# Variables to store last values
last_hazard_level = "Safe"
last_gas_value = None

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
    CAMERA_FOV_VERTICAL = 37  # Field of view for vertical axis in degrees
    FRAME_HEIGHT_PIXELS = 480  # Camera frame height in pixels
    fov_vertical_rad = np.radians(CAMERA_FOV_VERTICAL)
    real_world_height = 2 * depth * np.tan(fov_vertical_rad / 2) * (bounding_box_height / FRAME_HEIGHT_PIXELS)
    return real_world_height

# Function to calculate overall fire size based on average
def calculate_overall_fire_size(heights):
    if len(heights) == 0:
        return "Tidak Ada Api"  # Jika tidak ada deteksi
    avg_height = sum(heights) / len(heights)
    if avg_height < SMALL_THRESHOLD:
        return "Kecil"
    elif SMALL_THRESHOLD <= avg_height < LARGE_THRESHOLD:
        return "Sedang"
    else:
        return "Besar"

# Function to read gas data from Arduino
def read_gas_data():
    if arduino.in_waiting > 0:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            try:
                gas_value = int(data.split(":")[1].strip())  # Assume gas value format: 'nilai_gas: 500'
                print(f"Gas Sensor Data: {gas_value}")
                return gas_value
            except (ValueError, IndexError):
                return None
    return None

# Function to predict hazard level using ANN model with additional variable (fire count)
def predict_hazard(avg_fire_size, gas_level, fire_count):
    input_data = scaler.transform([[avg_fire_size, gas_level, fire_count]])  # Normalize input
    prediction = mlp_model.predict(input_data)
    hazard_level = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
    if hazard_level == 0:
        return "Aman"
    elif hazard_level == 1:
        return "Waspada"
    else:
        return "Bahaya"

# VideoWriter setup to save the output in MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('uji12.mp4', fourcc, 20.0, (640, 480))  # Save output

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
        detected_heights = []

        # Get gas sensor data from Arduino
        gas_value = read_gas_data()
        if gas_value is not None:
            last_gas_value = gas_value  # Update the last gas value if new data is available

        # Loop through detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])

                if class_id == 0:  # Assuming class_id == 0 is fire
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    bounding_box_height = y2 - y1
                    depth = depth_frame.get_distance(center_x, center_y)

                    # Calculate fire size
                    fire_height = calculate_real_height(bounding_box_height, depth)
                    detected_heights.append(fire_height)
                    fire_size_category = classify_fire_size(fire_height)

                    # Draw bounding box and display size
                    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Size: {fire_size_category} ({fire_height:.2f}m)", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate overall fire size
        overall_fire_size = calculate_overall_fire_size(detected_heights)

        # Predict hazard level if data is available
        fire_count = len(detected_heights)
        if fire_count > 0 and last_gas_value is not None:
            avg_fire_size = sum(detected_heights) / fire_count
            current_hazard_level = predict_hazard(avg_fire_size, last_gas_value, fire_count)
            last_hazard_level = current_hazard_level
        else:
            last_hazard_level = "Aman"

        # Display results on the screen
        cv2.putText(color_image, f"Fire Count: {fire_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(color_image, f"Overall Fire Size: {overall_fire_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if last_gas_value is not None:
            cv2.putText(color_image, f"Gas Value: {last_gas_value}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if last_hazard_level is not None:
            cv2.putText(color_image, f"Hazard Level: {last_hazard_level}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save the frame to the MP4 file
        out.write(color_image)

        # Display the final image with bounding boxes and annotations
        cv2.imshow('Fire Detection and Hazard Classification', color_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    arduino.close()
    out.release()
    cv2.destroyAllWindows()
