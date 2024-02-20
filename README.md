# <div align="center">TrafficFlow</div> 

* TrafficFlow is an open-source project developed by [KGXperience](https://kgx.nivu.me) designed for the automated vehicle surveillance of a given region. 
* The project consists of an object detection pipeline that tracks the entry and exit of vehicles by keeping track of their numberplate.
* The object detection pipeline is built using YOLOv8 and the dashboard is built using Thingsboard server.
 

   <div style="display: flex; justify-content: space-between;">
      <img src="https://github.com/kgex/trafficflowyolov8/blob/5aeb7fb1502605762303f4af2d4a70e2287352bb/assets/sample_image.png" alt="Left Image" style="width: 45%;">
      <img src="https://github.com/kgex/trafficflowyolov8/blob/5aeb7fb1502605762303f4af2d4a70e2287352bb/assets/Screenshot%202024-01-23%20184317.png" alt="Right Image" style="width: 45%;">
   </div>

## Table of Contents
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Models](#models)
- [Notebooks](#notebooks)
- [Contact Us](#developed-by)


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

## <div align="center">Documentation</div>


![Architecture](https://github.com/kgex/trafficflowyolov8/assets/83204531/ea5dfe51-8483-46f6-8eb9-0256a6f491fe)


## <div align="center">Models</div> 

 [YOLOv8](https://github.com/ultralytics/ultralytics), a state-of-the-art object detection model, is specifically trained for vehicle detection and numberplate detection respectively. Both models will run sequentially in the pipeline and the resultant image will be passed to PaddlePaddleOCR model for numberplate extraction.

1. **Vehicle Detection Model:**
    - The first YOLOv8 model is dedicated to detect the incoming vehicles within a given region.
    - The trained model will predict these particular classes:
      * Bike
      * Car
      * Bus
2. **Number Plate Detection Model:**
    - The second YOLOv8 model takes the cropped output of the detected vehicles and focuses on detecting the number plates from each vehicle.
 
The models were trained using indigenous dataset gathered by the incoming traffic of the KGiSL Campus.

### <div align="center">Metrics</div>

   
   |   Models   | Train Loss | Validation loss |    MaP     | Precision |  Recall  |
   |------------|------------|-----------------|------------|-----------|----------|
   |   Vehicle Detection Model        |  0.84073   |      1.0158     |   0.92726  |  0.88056  | 0.88054  |
   |Numbeplate Detection Model  |  0.99308   |      1.1624     |   0.95279   |  0.95196  |  0.94387   |



#### <div align="center">Vehicle Detection Model</div>
   ![Vehicle_model_metrics](https://github.com/kgex/trafficflowyolov8/blob/7a4392e7b8611efc53cd71b22b46ff2937c1a364/assets/vehicle_dataset_metrics.png)

#### <div align="center">Vehicle Detection Model</div>

   ![Numberplate_model_metrics](https://github.com/kgex/trafficflowyolov8/blob/7a4392e7b8611efc53cd71b22b46ff2937c1a364/assets/numberplate_model_metrics.png)


## Notebooks


| **Description**         | **Link**           |
|-------------------------|--------------------|
|  Vehicle Detection      | <a href="https://colab.research.google.com/drive/1GKBfgJgSN4GaUiJBrRYfB95wKsbeqtdG?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>      |
|  Number Plate Detection | <a href="https://colab.research.google.com/drive/1-VDVWyfE80405bOSMJnc0fYwJBDfaeOM?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>        |


## Developed by 
<p align="center">
  <img src="https://github.com/kgex/trafficflowyolov8/assets/83204531/ad252a50-f3b9-4960-807e-5b52b679c656" alt="KGX_Logo" width = 100 height = 54>
</p>
<div align="center">
  <a href="https://github.com/kgex"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="KGX GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/kgx/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="KGX LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.instagram.com/kgxperience/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="KGX Instagram"></a>

</div>
</div>
