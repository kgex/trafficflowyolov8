import cv2
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import asyncio
import paho.mqtt.client as mqtt
import imagezmq
import socket
import time
import math
import pickle
import sys
import os
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
model2 = YOLO('best.pt')
ocr = PaddleOCR(use_angle_cls=True, lang="en")


max_length = 65000
host = "127.0.0.1"
port = 5000


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_frame(frame):
    # compress frame
    retval, buffer = cv2.imencode(".jpg", frame)

    if retval:
        # convert to byte array
        buffer = buffer.tobytes()
        # get size of the frame
        buffer_size = len(buffer)

        num_of_packs = 1
        if buffer_size > max_length:
            num_of_packs = math.ceil(buffer_size / max_length)

        frame_info = {"packs": num_of_packs}

        # send the number of packs to be expected
        print("Number of packs:", num_of_packs)
        sock.sendto(pickle.dumps(frame_info), (host, port))

        left = 0
        right = max_length

        for i in range(num_of_packs):
            print("left:", left)
            print("right:", right)

            # truncate data to send
            data = buffer[left:right]
            left = right
            right += max_length

            # send the frames accordingly
            sock.sendto(data, (host, port))

# Define MQTT parameters
client = mqtt.Client("trafficflowyolov8")
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
video_path = "/home/nawin/Projects/kgx/trafficflowyolov8/input/PXL_20230417_082118392.mp4"

cap = cv2.VideoCapture(video_path)

# Get video information (width, height, frames per second)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output\output_video.mp4', fourcc, fps, (width, height))

async def function_async2(plate_crop_img):
    print(plate_crop_img.shape)
    if plate_crop_img.shape[0] == 0 or plate_crop_img.shape[1] == 0:
        result = 'No Plate Detected'
        return result
    else:
        global plate_count
        data = Image.fromarray(cv2.cvtColor(plate_crop_img, cv2.COLOR_BGR2RGB))
        data.save('Non.png')
        text = ocr.ocr('Non.png')
        result = ''
        print('Numberplate: ', text)
        for idx in range(len(text)):
            res = text[idx]
            if res is not None:
                for line in res:
                    result += (line[1][0])
                print('Numberplate: ', result)
        return result

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Check if boxes are not None before accessing attributes
        if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

            for i in range(len(results[0].boxes)):
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy.cpu()[i])
                cls = results[0].boxes.cls[i].item()
                # Draw bounding box based on the detected class
                if cls in color_dict:
                    color = color_dict[cls]
                    thickness = 2  # Thickness of the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    # Display class name in the top of the bounding box
                    if cls == 2:
                        class_name = f"Car"
                    elif cls == 3:
                        class_name = f"Bike"
                    elif cls == 5:
                        class_name = f"Bus"
                    elif cls == 7:
                        class_name = f"Truck"
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                    print(f"Vehicle detected: {class_name}")
                    cropped_img = frame[y1:y2, x1:x2]
                    # cv2.imshow("cropped", cropped_img)
                    result_new = model2(cropped_img)
                    # Check if boxes are not None before accessing elements
                    for i in range(len(result_new[0].boxes)):
                        if result_new[0].boxes is not None and len(result_new[0].boxes.xyxy) > 0:
                            x, y, w, h = result_new[0].boxes.xyxy[0]
                            x, y, w, h = x.cpu().numpy().astype(np.int32), y.cpu().numpy().astype(np.int32), w.cpu().numpy().astype(
                                np.int32), h.cpu().numpy().astype(np.int32)
                            annotated_frame = result_new[0].plot()
                            crop_img = cropped_img[y:h, x:w]
                            cv2.imshow("cropped", crop_img)
                            plate_text = asyncio.run(function_async2(crop_img))
                            print('Plate Text:', plate_text)
                            result = "{class_name}, {plate_text}".format(class_name=class_name, plate_text=plate_text)
                            client.publish('trafficflowyolov8', result)

                            # Draw number plate information on the annotated frame
                            cv2.putText(frame, f'Plate: {plate_text}', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the annotated frame
            resized_frame = cv2.resize(frame, (800, 600))  # Adjust the window size as needed
            send_frame(resized_frame)
            cv2.imshow("YOLOv8 Tracking", resized_frame)

            # Write the frame to the output video
            output_video.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, close the display window, and release the output video
cap.release()
output_video.release()
cv2.destroyAllWindows()
