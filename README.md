# Fire Detection with Intel RealSense D435 and MQ Sensor

"This repository contains my final project"

##Table of Contents
- [Installation](#Installation)

## Requirement
- Intel RealSense D435
- MQ Sensor
- Arduino
  
## Installation
Instructions on how to setup the project.

### Install Python Dependencies
```sh
pip install numpy opencv-python ultralytics tensorflow joblib pyserial pandas openpyxl
pip install pyrealsense2
```

## Preparing Model Files
Make sure all required model files are in the same directory as the program:
- YOLO Model: <br/>
  You can use YOLO Model from [Ultralytics](https://www.ultralytics.com/) or you can train your fire model. <br/>
  This is my fire model you can use it [best.pt](https://github.com/FarrelCulbert/tugas-akhir/blob/main/Main/best.pt)

- ANN Model: <br/>
Train ANN Model from reference data and you will get file.h5 and file.pkl <br/>
[ANN Model Program](https://github.com/FarrelCulbert/tugas-akhir/blob/main/Main/ann.py) <br/>
or use my ANN model at [here](https://github.com/FarrelCulbert/tugas-akhir/tree/main/Main)

## Program
### Arduino 
Upload MQ Sensor program to your Arduino. 
```sh
#define MQ_SENSOR A0  // Sensor gas MQ dihubungkan ke pin A0

void setup() {
    Serial.begin(9600);  // Mulai komunikasi serial
}

void loop() {
    int gas_value = analogRead(MQ_SENSOR);  // Baca nilai sensor gas
    Serial.println(gas_value);  // Kirim nilai ke Serial Monitor
    delay(1000);  // Tunggu 1 detik sebelum membaca lagi
}

```

### Final Program
After you have all model files and Arduino with MQ Sensor program , you can run the program<br/>
[Program](https://github.com/FarrelCulbert/tugas-akhir/blob/main/Main/fix.py)<br/>
if your camera and arduino access errors, ensure the correct port is used
