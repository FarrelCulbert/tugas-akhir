import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd  # Library untuk menyimpan ke Excel

# Load YOLO model
model = YOLO('yolo11n.pt')  # Replace with your custom YOLO model if needed

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

# RealSense camera specifications
CAMERA_FOV_VERTICAL = 37  # Field of view for vertical axis in degrees
FRAME_HEIGHT_PIXELS = 480  # Camera frame height in pixels

# Actual height of the person (in meters)
ACTUAL_HEIGHT = 1.56

# Data storage for Excel
data = []

# Function to calculate real-world size using depth
def calculate_real_world_size(bounding_box_height, bounding_box_width, depth):
    fov_vertical_rad = np.radians(CAMERA_FOV_VERTICAL)
    real_height = 2 * depth * np.tan(fov_vertical_rad / 2) * (bounding_box_height / FRAME_HEIGHT_PIXELS)
    real_width = real_height * (bounding_box_width / bounding_box_height)  # Assuming aspect ratio consistency
    return real_height, real_width

# VideoWriter setup for recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('output8_uni.mp4', fourcc, 20.0, (640, 480))  # Save output

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

        # Perform YOLO inference
        results = model(color_image)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])

                # Assuming you're interested in a specific class, e.g., 'person'
                if class_id == 0:  # Replace '0' with the ID of the object you want to measure
                    bounding_box_height = y2 - y1
                    bounding_box_width = x2 - x1

                    # Calculate depth at the center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    depth = depth_frame.get_distance(center_x, center_y)

                    # Calculate real-world height and width
                    real_height, real_width = calculate_real_world_size(bounding_box_height, bounding_box_width, depth)

                    # Calculate the error
                    error = abs(ACTUAL_HEIGHT - real_height)
                    error_percentage = (error / ACTUAL_HEIGHT) * 100

                    # Draw bounding box
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Height: {real_height:.2f} m", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Error: {error:.2f} m ({error_percentage:.2f}%)", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(color_image, f"Depth: {depth:.2f} m", (x1, y1 - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Append data to the list
                    data.append({
                        "Height (m)": real_height,
                        "Depth (m)": depth,
                        #"Error (m)": error,
                        #"Error (%)": error_percentage
                    })

        # Write frame to video
        out.write(color_image)


        # Display the frame
        cv2.imshow("RealSense Error Measurement with YOLO", color_image)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Save data to Excel
    df = pd.DataFrame(data)
    df.to_excel("object_measurements_uni.xlsx", index=False)  # Save data to Excel file

    # Stop the pipeline and release resources
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
