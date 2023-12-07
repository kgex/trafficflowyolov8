"""Program to detect vehicles and number plates in a video using YOLOv8 and PaddleOCR"""
from collections import defaultdict
import asyncio
import socket
import math
import pickle
import time
import sys
import os
import re

root = os.path.join(os.getcwd())
sys.path.append(root)
models_dir = os.path.join(root, "models/")
input_dir = os.path.join(root, "input/")

from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
import paho.mqtt.client as mqtt
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans
from pandas import DataFrame



# Load the YOLOv8 model
model = YOLO(models_dir + "yolov8n.pt")
model2 = YOLO(models_dir + "best.pt")
tracker = DeepSort(max_age=30)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# cfg = get_config()
# cfg.merge_from_file(os.path.join(root, "deep_sort/configs/deep_sort.yaml"))
# deepsort = DeepSort("osnet_x0_25",
#                     max_dist=cfg.DEEPSORT.MAX_DIST,
#                     max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
#                     use_cuda=True)

# Define MQTT parameters
topic_pub = 'v1/devices/me/telemetry'
client = mqtt.Client()

# client.username_pw_set("0tdnwetjCarO9XFvDGVh")
# # client.connect('192.168.229.20', 1883, 1)
client.connect("broker.hivemq.com", 1883, 60)

# Define colors for different vehicles
color_dict = {2: (0, 255, 0),  # car
              3: (255, 0, 0),  # bike
              5: (0, 0, 255),  # bus
              7: (255, 255, 0)  # truck
              }

# Track history dictionary
track_history = defaultdict(lambda: [])

# Open the video file
VIDEO_PATH = "/home/nawin/Projects/kgx/trafficflowyolov8/input/input.mp4"
# VIDEO_PATH = "/home/nawin/Projects/kgx/trafficflowyolov8/input/video_20231201_102444.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

# Get video information (width, height, frames per second)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

roi_line = [(0, 700), (width, 700)]
roi_line_color = (0, 255, 0)  # Green color

MAX_LENGTH = 65000
HOST = "127.0.0.1"
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def Regex(Numplate):
    """Function to check if the number plate is valid or not"""
    # Defining the regular expression
    patt = "^[A-Z]{2}[ -]?[0-9]{2}[ -]?[A-Z]{1,2}[ -]?[0-9]{4}$"

    # When the string is empty
    if not Numplate:
        return False

    if len(Numplate) == 0:
        return False

    # Return the answer after validating the number plate
    if re.match(patt, Numplate):
        return True
    else:
        return False

def extract_dominant_color(image, k=1):
    """Convert the image from BGR to RGB"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = image_rgb.reshape((-1, 3))

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the dominant color(s)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    return dominant_colors
    
def convert_rgb_to_names(rgb_tuple):
    """Function to convert the rgb colors to its respective names."""    
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}' 

async def function_async2(plate_crop_img):
    """Function to extract the number plate from the cropped image and pass it to the OCR model"""
    print(plate_crop_img.shape)
    if plate_crop_img.shape[0] == 0 or plate_crop_img.shape[1] == 0:
        result = 'No Plate Detected'
        return result
    else:
        data = Image.fromarray(cv2.cvtColor(plate_crop_img, cv2.COLOR_BGR2RGB))
        data.save('Non.png')
        text = ocr.ocr('Non.png')
        result = ''
        for idx, value in enumerate(text):
            res = value
            if res is not None:
                for line in res:
                    result += (line[1][0])
                    confidence = line[1][1]
                    print('Numberplate: ', result, "Confidence", confidence)
        return result
    
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    COUNT = 0
    if success:
        cv2.line(frame, roi_line[0], roi_line[1], roi_line_color, 3)
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, classes=[2, 3, 5, 7], persist = True, tracker='botsort.yaml')

        # Check if boxes are not None before accessing attributes
        if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            for i in range(len(results[0].boxes)):
                conf = results[0].boxes.conf[i].int().cpu().tolist()
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy.cpu()[i])
                cls = results[0].boxes.cls[i].item()
                # Draw bounding box based on the detected class
                if cls in color_dict:
                    color = color_dict[cls]
                    THICKNESS = 2  # thickness of the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
                    # Display class name in the top of the bounding box
                    if cls == 2:
                        CLASS_NAME = "Car"
                    elif cls == 3:
                        CLASS_NAME = "Bike"
                    elif cls == 5:
                        CLASS_NAME = "Bus"
                    elif cls == 7:
                        CLASS_NAME = "Truck"
                    else:
                        CLASS_NAME = "Unknown"
                    cv2.putText(
                        frame, CLASS_NAME, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                          THICKNESS
                    )

                if y2 >700:
                # Numberplate Identification
                    cropped_img = frame[y1:y2, x1:x2]
                    dominant_color = extract_dominant_color(cropped_img, k=1)[0]
                    color = convert_rgb_to_names(dominant_color)
                    result_new = model2(cropped_img)
                    if results[0].boxes is None and len(results[0].boxes.xyxy) == 0:
                        RESULT = "{" + "vehicle_id" + ":" + str(COUNT) + "," + "vehicle_type" + ":" + str(CLASS_NAME) + "," + "color" + ":" + str(color) + "," "numplate" + ":" + "None" + "}"    
                        client.publish(topic_pub, RESULT)
                        time.sleep(1)
                        continue
                    for i in range(len(result_new[0].boxes)):
                        if track_history.get(track_ids[i]) is None:
                            if result_new[0].boxes is not None and len(result_new[0].boxes.xyxy) > 0:
                                x, y, w, h = result_new[0].boxes.xyxy[0]
                                x = x.cpu().numpy().astype(np.int32)
                                y = y.cpu().numpy().astype(np.int32)
                                w = w.cpu().numpy().astype(np.int32)
                                h = h.cpu().numpy().astype(np.int32)
                                # cropping the image to pass it to the ocr model.
                                crop_img = cropped_img[y:h, x:w]
                                loop = asyncio.get_event_loop()
                                plate_text = loop.run_until_complete(function_async2(crop_img))
                                plate_text_send = 'None' if plate_text == '' else plate_text
                            # if Regex(plate_text_send):
                                print(type(plate_text), plate_text)                          
                                track_history.update({track_ids[i]: [x1, y1, x2, y2]})
                                RESULT = "{" + "vehicle_id" + ":" + str(track_ids[i]) + "," + "vehicle_type" + ":" + str(CLASS_NAME) + "," + "color" + ":" + str(color) + "," "numplate" + ":" + str(plate_text_send) + "}"    
                                print("Data published in mqtt \n", RESULT)
                                client.publish(topic_pub, RESULT)
                                time.sleep(1)
                                # Draw number plate information on the annotated frame
                                cv2.putText(frame, f'Plate: {plate_text}', (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                COUNT += 1

            # Display the annotated frame
            resized_frame = cv2.resize(frame, (900, 600))  # Adjust the window size as needed
            cv2.imshow("YOLOv8 Tracking", resized_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, close the display window, and release the output video
cap.release()
