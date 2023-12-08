# TrafficFlow 

![TrafficFlow Logo](https://github.com/kgex/trafficflowyolov8/logo.png)

The Traffic Flow project at KGISL utilizes cutting-edge technologies like DeepSort for precise object tracking and ANPR/OCR for license plate recognition. This revolutionizes vehicle monitoring on campus by extracting vehicle images with DeepSort, analyzing them through ANPR/OCR for accurate vehicle numbers and counts, and ensuring real-time data processing and precise movement tracking. Integrating MQTT and Thingsboard establishes a swift data transmission system, facilitating instant vehicle counts, Vehicle type, Number plate value, and Vehicle color along with a timestamp. This significantly enhances safety measures and optimizes traffic flow at KGISL, contributing to a smarter, more secure educational campus.

## Table of Contents
- [Getting Started](#getting-started)
- [Features](#features)
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

