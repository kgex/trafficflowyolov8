# TrafficFlow 

![TrafficFlow Logo](https://github.com/kgex/trafficflowyolov8/logo.png)

The Traffic Flow project at KGISL utilizes cutting-edge technologies like DeepSort for precise object tracking and ANPR/OCR for license plate recognition. This revolutionizes vehicle monitoring on campus by extracting vehicle images with DeepSort, analyzing them through ANPR/OCR for accurate vehicle numbers and counts, and ensuring real-time data processing and precise movement tracking. Integrating MQTT and Thingsboard establishes a swift data transmission system, facilitating instant vehicle counts, Vehicle type, Number plate value, and Vehicle color along with a timestamp. This significantly enhances safety measures and optimizes traffic flow at KGISL, contributing to a smarter, more secure educational campus.

## Table of Contents
- [Getting Started](#getting-started)
- [Features](#features)
- [Overall Architecture](#overall-architecture)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)


## Getting Started

Follow these simple steps to get started with TrafficFlow:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/kgex/trafficflowyolov8

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt

3. **Run the Code:**
   ```bash
   python scripts/final.py

## Features

1. DeepSort Integration for Object Tracking: Utilizing DeepSort technology to achieve precise
   object tracking within video feeds, allowing for accurate monitoring of vehicle movements.

   ![Example from our model](https://github.com/kgex/trafficflowyolov8/logo.png)

3. ANPR/OCR for License Plate Recognition: Implementing Automatic Number Plate Recognition
   (ANPR) and Optical Character Recognition (OCR) to accurately extract and process license
   plate information in real time.

   ![example from our model](https://github.com/kgex/trafficflowyolov8/logo.png)

5. Dynamic Dashboard for Insights: Creating a dynamic dashboard that visualizes and provides
   insights derived from the tracked data, offering real-time information on vehicle counts,
   types, and movement patterns.

   ![example from model](https://github.com/kgex/trafficflowyolov8/logo.png)

## Overall Architecture

1. **CCTV Camera Stream:**
   - Provides the video feed capturing cars, trucks, motorcycles, and buses.

2. **Frame Processing Module:**
   - Receives and processes frames from the camera stream.

3. **Vehicle Detection Module (Object Detection):**
   - Identifies vehicles within the defined Region of Interest (ROI).

4. **Vehicle Classification Module:**
   - Classifies detected vehicles into different types (cars, trucks, motorcycles, and buses) within the ROI.

5. **Number Plate and Vehicle Attributes Detection Module:**
   - Performs number plate detection using OCR and extracts vehicle attributes within the ROI.

6. **ROI Management:**
   - Defines and manages the Region of Interest within the frame for specific area analysis.

7. **Decision Logic and Database Interaction:**
   - Manages decision-making based on detected vehicles, number plate data, and validation outcomes within the ROI.
   - Stores vehicle-related data in the database.

8. **Database:**
   - Stores collected data including vehicle timestamps, IDs, types, attributes, and number plates within the ROI.

9. **MQTT Broker:**
   - Facilitates communication between the database and Thingboard Dashboard for ROI-related data transfer.

10. **Thingboard Dashboard:**
    - Visualizes data received through MQTT related to the ROI for monitoring and analysis.

11. **Storage System for Unsuccessful Images:**
    - Stores images that failed validation or processing within the ROI for further analysis.

## Workflow:

1. **Camera Stream Input:**
   - The system continuously receives frames capturing vehicles.

2. **Frame Processing and ROI Analysis:**
  
