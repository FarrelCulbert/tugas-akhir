import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model pre-trained on detecting fire (after training)
model = YOLO('yolov8n.pt')  # Use your custom-trained YOLOv8 model for fire

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Fire size classification thresholds in meters
SMALL_THRESHOLD = 0.5
LARGE_THRESHOLD = 2.0

# RealSense camera specifications
CAMERA_FOV_VERTICAL = 57  # Field of view for vertical axis in degrees (adjust based on camera spec)
FRAME_HEIGHT_PIXELS = 480  # Camera frame height in pixels

# Function to classify fire size based on real-world height
def classify_fire_size(real_height):
    if real_height < SMALL_THRESHOLD:
        return "Small Fire"
    elif SMALL_THRESHOLD <= real_height < LARGE_THRESHOLD:
        return "Medium Fire"
    else:
        return "Large Fire"

# VideoWriter setup to save the output in MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('output_fire_detection3.mp4', fourcc, 20.0, (640, 480))  # Save output

# Function to calculate real-world height
def calculate_real_height(bounding_box_height, depth):
    # Convert the field of view (vertical) from degrees to radians
    fov_vertical_rad = np.radians(CAMERA_FOV_VERTICAL)
    
    # Calculate the real-world height using the depth and bounding box height in pixels
    real_world_height = 2 * depth * np.tan(fov_vertical_rad / 2) * (bounding_box_height / FRAME_HEIGHT_PIXELS)
    return real_world_height

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

        # Get the frame dimensions
        frame_height, frame_width = depth_image.shape

        # Loop through detection results
        for result in results:
            boxes = result.boxes  # Get bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
                class_id = int(box.cls[0])  # Get the class ID

                # Check if the detected object is fire
                if class_id == 0:
                    # Draw bounding box around the fire
                    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Calculate the center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Ensure the coordinates are within the frame dimensions
                    if center_x >= frame_width:
                        center_x = frame_width - 1
                    if center_y >= frame_height:
                        center_y = frame_height - 1

                    # Get the distance from the camera at the center of the bounding box
                    distance = depth_frame.get_distance(center_x, center_y)

                    # Calculate the real-world height of the fire using depth and bounding box height
                    bounding_box_height = y2 - y1
                    fire_height = calculate_real_height(bounding_box_height, distance)

                    # Classify the fire size based on the real-world height
                    fire_size_label = classify_fire_size(fire_height)

                    # Display the fire size and distance on the frame
                    label = f"{fire_size_label}, Height: {fire_height:.2f}m, Dist: {distance:.2f}m"
                    cv2.putText(color_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Save the frame to the MP4 file
        out.write(color_image)

        # Display the color image with bounding box and fire size
        cv2.imshow('RealSense YOLOv8 Fire Detection with Size', color_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Stop the pipeline and release resources
    pipeline.stop()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()
