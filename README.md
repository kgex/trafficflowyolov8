# TrafficFlow 

TrafficFlow, developed by [KGXperience](https://github.com/kgex), is an innovative project designed for the automated surveillance of a given region. This sophisticated system facilitates the tracking of vehicles entering and exiting the area by capturing and analyzing their license plates. Additionally, TrafficFlow offers comprehensive data, including the precise times of arrival and departure for each vehicle within the specified region.

<p align="center">
  <img src="https://github.com/kgex/trafficflowyolov8/assets/83204531/3426b516-66ae-425b-af1f-21766814d3c4" alt="Sample">
</p>


## Table of Contents
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Models Used](#models-used)


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
3. *Download Sample Video*
   ```
   gdown https://drive.google.com/uc?id=1_Lsve_tfnv5weRMnYE93K1aANWyoRk8E
   ```
4. *Run the Code:*
   ```
   python scripts/main.py --input # 0 for webcam or path/to/video.mp4
   ```

## Architecture

![Architecture](https://github.com/kgex/trafficflowyolov8/assets/83204531/ea5dfe51-8483-46f6-8eb9-0256a6f491fe)


## Models Used

### YOLOv8 Vehicle Detection System

The code architecture is centered around [YOLOv8](https://github.com/ultralytics/ultralytics), a state-of-the-art object detection model, specifically tailored for vehicle detection. The process involves two sequential YOLOv8 models, each serving a distinct purpose.

1. **Vehicle Detection Model:**
    - The first YOLOv8 model is dedicated to detect the incoming vehicles within a given region.
    - The trained model will predict these particular classes:
      * Bike
      * Car
      * Bus
   - The model was trained using the incoming traffic of the institute and the following table illustrates the results of the training process.

   
   | Train Loss | Validation loss |    MaP     | Precision |  Recall  |
   |------------|-----------------|------------|-----------|----------|
   |  0.84073   |      1.0158     |   0.92726  |  0.88056  | 0.88054  |

   ![Vehicle_model_metrics]()


2. **Number Plate Detection Model:**
    - The second YOLOv8 model takes the cropped output of the detected vehicles and focuses on detecting the number plates from each vehicle.
    - The following table illustrated the results of the training process.


   | Train Loss | Validation loss |    MaP      | Precision |  Recall    |
   |------------|-----------------|-------------|-----------|------------|
   |  0.99308   |      1.1624     |   0.95279   |  0.95196  |  0.94387   |

   ![Numberplate_model_metrics]()

### Integration with Thingsboard Server using MQTT Protocol

The final step involves forwarding the extracted vehicle details to a Thingsboard server using the MQTT (Message Queuing Telemetry Transport) protocol. This ensures seamless and efficient communication between the detection system and the Thingsboard server, facilitating real-time updates and storage of pertinent vehicle information.

This comprehensive architecture enhances the capabilities of the system, providing a robust solution for automated vehicle surveillance, number plate recognition, and efficient data integration with the Thingsboard platform.

<p align="center">
  <img src="https://github.com/kgex/trafficflowyolov8/assets/83204531/c6831336-7c03-4f02-9971-e1e96dffa526" alt="Dashboard">
</p>


## Developed by 
<p align="left">
  <img src="https://github.com/kgex/trafficflowyolov8/assets/83204531/ad252a50-f3b9-4960-807e-5b52b679c656" alt="KGX_Logo" width = 100 height = 54>
</p>
