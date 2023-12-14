"""Vehicle and Number Plate Detection Program

This program uses YOLOv8 for vehicle detection and PaddleOCR for number plate 
recognition in a video stream.

The detected information is then published to a MQTT server.

Requirements:
- YOLOv8 (Ultralytics implementation)
- PaddleOCR
- OpenCV
- Webcolors
- Paho MQTT
- Scikit-learn
- Pandas

Note: Make sure the required libraries are installed using
requirements.txt file before running the program.

"""
from collections import defaultdict
import asyncio
import socket
import math
import pickle
import time
import sys
import os
import re
import argparse
from datetime import datetime
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
import paho.mqtt.client as mqtt
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans

# Setting up the root directory
root = os.path.join(os.getcwd())
sys.path.append(root)
models_dir = os.path.join(root, "models/")
input_dir = os.path.join(root, "input/")
output_dir = os.path.join(root, "output/")
dataset_dir = os.path.join(output_dir, "Dataset/")

# Load the YOLOv8 model
model = YOLO(models_dir + "yolov8n.pt")
model2 = YOLO(models_dir + "best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Define MQTT parameters
TOPIC = 'v1/devices/me/telemetry'
client = mqtt.Client()


client.username_pw_set("Z2pU7qUneWGK070mryjO")  
client.connect("mqtt.thingsboard.cloud", 1883, 60)


def regex(numplate):
    """Function to check if the number plate is valid or not"""
    # Defining the regular expression
    patt = "^[A-Z]{2}[ -]?[0-9]{2}[ -]?[A-Z]{1,2}[ -]?[0-9]{4}$"

    # When the string is empty
    if not numplate:
        return False

    if len(numplate) == 0:
        return False

    # Return the answer after validating the number plate
    if re.match(patt, numplate):
        return True
    else:
        return False
    
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

    return convert_rgb_to_names(dominant_colors[0])
    


async def function_async2(plate_crop_img):
    """Function to extract the number plate from the cropped image and pass it to the OCR model"""
    ocr_output = ocr.ocr(plate_crop_img) # ocr output format = [[[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [text, confidence]]]
    if ocr_output is None:
        return None
    confidence = ocr_output[0][0][1][1]
    numplate = ocr_output[0][0][1][0]
    print(numplate, confidence)
    return numplate

def detect(cap):   
    """Function to detect vehicles and number plates in a video stream"""
    # Define colors for different vehicles
    color_dict = {2: (0, 255, 0),  # car
                3: (255, 0, 0),  # bike
                5: (0, 0, 255),  # bus
                7: (255, 255, 0)  # truck
                }

    # to display the vehicle count
    COUNT = 0

    # Get video information (width, height, frames per second)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    roi_line = [(0, 700), (width, 700)]
    roi_line_color = (0, 255, 0)  # Green color

    # Track history dictionary
    track_history = defaultdict(lambda: [])
    track_ids = [] 


    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            cv2.line(frame, roi_line[0], roi_line[1], roi_line_color, 3)
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # results = model(frame, classes=[2, 3, 5, 7])
            results = model.track(frame, classes = [2, 3, 5, 7], persist = True, tracker='botsort.yaml')

            # Check if boxes are not None before accessing attributes
            if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                for i in range(len(results[0].boxes)):
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

                if y2 > 700 and i < len(track_ids) and track_history.get(track_ids[i]) is None:
                    cropped_img = frame[y1:y2, x1:x2]
                    cv2.imwrite(f"{dataset_dir}vehicle/vehicle_img{COUNT}.jpg", cropped_img)
                    #The code to extract the dominant color from the cropped image
                    color = extract_dominant_color(cropped_img, k=1)
                                        
                    # Numberplate Identification
                    result_new = model2(cropped_img)

                    # If no numberplate is detected
                    if len(result_new[0].boxes) == 0:
                        COUNT += 1
                        RESULT = "{" + "vehicle_id" + ":" + str(COUNT) + "," + "vehicle_type" + ":" + str(CLASS_NAME) + "," + "color" + ":" + str(color) + "," "numplate" + ":" + "None" + "}"
                        track_history.update({track_ids[i]: [x1, y1, x2, y2]})                  
                        print("Data published in mqtt \n", RESULT)
                        client.publish(TOPIC, RESULT)
                        continue

                    # Processing numberplate detections
                    for i in range(len(result_new[0].boxes)):
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
                            cv2.imwrite(f"{dataset_dir}numberplate/numberplate_img{COUNT}.jpg", crop_img)

                            # If the numberplate is valid                
                            if plate_text is not None and regex(plate_text) == True:
                                COUNT += 1
                                print(type(plate_text), plate_text)                          
                                track_history.update({track_ids[i]: [x1, y1, x2, y2]})
                                RESULT = "{" + "vehicle_id" + ":" + str(COUNT) + "," + "vehicle_type" + ":" + str(CLASS_NAME) + "," + "color" + ":" + str(color) + "," "numplate" + ":" + str(plate_text) + "}"    
                                print("Data published in mqtt \n", RESULT)
                                client.publish(TOPIC, RESULT)

                # Draw number plate information on the annotated frame
                cv2.putText(frame, f'Time: {datetime.now()}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Display the annotated frame
                resized_frame = cv2.resize(frame, (900, 600))  # Adjust the window size as needed
                cv2.imshow("YOLOv8 Tracking", resized_frame)
                normalized_image = resized_frame.astype(np.float32) / 255.0
                blob = cv2.dnn.blobFromImage(
                    normalized_image, 1 / 255.0, (416, 416), 
                    swapRB = True, crop = False)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, close the display window, and release the output video
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=0, help='0 for webcam or path/to/video/file.mp4')
    args = parser.parse_args()
    if args.input == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)
    detect(cap)

if __name__ == '__main__':
    main()


# End-of-file (EOF)
