import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler
import serial  # For serial communication with Arduino

# Load YOLOv8 model pre-trained on detecting fire
model = YOLO('D:\\yolo\\runs\detect\\fire_test\\weights\\best.pt')  # Use your custom-trained YOLOv8 model for fire

# Load pre-trained ANN model and scaler
mlp_model = load_model('fire_hazard_ann_model_v2.h5')
scaler = joblib.load('scaler_v2.pkl')

# Set up serial communication with Arduino (for gas sensor)
arduino = serial.Serial('COM6', 9600, timeout=1)  # Change COM6 to your correct port

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
SMALL_THRESHOLD = 0.5
LARGE_THRESHOLD = 2.0

# Variables to store last values
last_fire_size = None
last_hazard_level = "Safe"
last_gas_value = None
fire_count = 0

# Function to classify fire size based on real-world height
def classify_fire_size(real_height):
    if real_height < SMALL_THRESHOLD:
        return "Small"
    elif SMALL_THRESHOLD <= real_height < LARGE_THRESHOLD:
        return "Medium"
    else:
        return "Large"

# Function to calculate real-world height
def calculate_real_height(bounding_box_height, depth):
    CAMERA_FOV_VERTICAL = 57  # Field of view for vertical axis in degrees (adjust based on camera spec)
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
        return "Safe"
    elif hazard_level == 1:
        return "Medium"
    else:
        return "Danger"

# VideoWriter setup to save the output in MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('output_fire_detection.mp4', fourcc, 20.0, (640, 480))  # Save output

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
        current_fire_size = None
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
                    current_fire_size = classify_fire_size(fire_height)

        # Update fire count and hazard level based on detections
        fire_count = current_fire_count
        if fire_count > 0 and last_gas_value is not None and current_fire_size:
            # Predict hazard level using ANN
            current_hazard_level = predict_hazard(fire_height, last_gas_value, fire_count)
            last_hazard_level = current_hazard_level  # Update hazard level
            last_fire_size = current_fire_size  # Update fire size
        else:
            # Reset if no fire is detected
            last_fire_size = None
            if last_gas_value is not None:
                if last_gas_value >= 150:
                    last_hazard_level = "Danger" if last_gas_value > 400 else "Medium"
                else:
                    last_hazard_level = "Safe"
            else:
                last_hazard_level = "Unknown"

        # Display the fire count, fire size, gas value, and hazard level statically at the top-left corner
        cv2.putText(color_image, f"Fire Count: {fire_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if last_fire_size is not None:
            cv2.putText(color_image, f"Fire Size: {last_fire_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if last_gas_value is not None:
            cv2.putText(color_image, f"Gas Value: {last_gas_value}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if last_hazard_level is not None:
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
