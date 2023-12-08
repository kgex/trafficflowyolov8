# TrafficFlow 

TrafficFlow, developed by [KGXperience](https://github.com/kgex), is an innovative project designed for the automated surveillance of a given region. This sophisticated system facilitates the tracking of vehicles entering and exiting the area by capturing and analyzing their license plates. Additionally, TrafficFlow offers comprehensive data, including the precise times of arrival and departure for each vehicle within the specified region.

<p align="center">
  <img src="https://github.com/kgex/trafficflowyolov8/assets/83204531/3426b516-66ae-425b-af1f-21766814d3c4" alt="Sample">
</p>

<p align="center">
  <img src="https://github.com/kgex/trafficflowyolov8/assets/83204531/ad252a50-f3b9-4960-807e-5b52b679c656" alt="KGX_Logo" width = 100 height = 54>
</p>

## Table of Contents
- [Getting Started](#getting-started)
- [Architecture](#Architecture)


## Getting Started

Follow these simple steps to get started with TrafficFlow:

1. *Clone the Repository:* 
   ```
   git clone https://github.com/kgex/trafficflowyolov8
   ```

2. *Install Requirements:*
   ```
   pip install -r requirements.txt
   ```

3. *Run the Code:*
   ```
   python scripts/main.py
   ```

## Architecture

![Architecture](https://github.com/kgex/trafficflowyolov8/assets/83204531/ea5dfe51-8483-46f6-8eb9-0256a6f491fe)

### YOLOv8 Vehicle Detection System

The code architecture is centered around YOLOv8, a state-of-the-art object detection model, specifically tailored for vehicle detection. The process involves two sequential YOLOv8 models, each serving a distinct purpose.

1. **Vehicle Detection Model:**
    - The first YOLOv8 model is dedicated to detecting vehicles within a given scene or region.
    - Upon successful detection, the identified vehicles are then forwarded to the next stage for further analysis.

2. **Number Plate Detection Model:**
    - The second YOLOv8 model takes the previously detected vehicles and focuses on locating and extracting number plates from each vehicle.
    - This enables a detailed analysis of vehicle-specific information.

### PaddleOCR for Text Extraction

After the successful identification of number plates, the architecture integrates PaddleOCR, a powerful Optical Character Recognition (OCR) tool. PaddleOCR is employed to extract text from the detected number plates, enabling the retrieval of alphanumeric details.

### Integration with Thingsboard Server using MQTT Protocol

The final step involves forwarding the extracted vehicle details to a Thingsboard server using the MQTT (Message Queuing Telemetry Transport) protocol. This ensures seamless and efficient communication between the detection system and the Thingsboard server, facilitating real-time updates and storage of pertinent vehicle information.

This comprehensive architecture enhances the capabilities of the system, providing a robust solution for automated vehicle surveillance, number plate recognition, and efficient data integration with the Thingsboard platform.

