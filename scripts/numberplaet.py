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
import sys
import os
import argparse
from ultralytics import YOLO
import cv2

# Setting up the root directory
root = os.path.join(os.getcwd())
sys.path.append(root)

models_dir = os.path.join(root, "models/")
input_dir = os.path.join(root, "input/")
output_dir = os.path.join(root, "output/")
dataset_dir = os.path.join(root, "Dataset/")

# Load the YOLOv8 model
model = YOLO(models_dir + "yolov8n.pt")

    
def detect(cap):   
    """Function to detect vehicles and number plates in a video stream"""
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
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # results = model(frame, classes=[2, 3, 5, 7])
            results = model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker='botsort.yaml')
            # cv2.line(frame, roi_line[0], roi_line[1], roi_line_color, 3)

            # Check if boxes are not None before accessing attributes
            if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                for i in range(len(results[0].boxes)):
                    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy.cpu()[i])
                    cls = results[0].boxes.cls[i].item()

                    if y2 > 700 and i < len(track_ids) and track_history.get(track_ids[i]) is None and cls == 2:
                        track_history.update({track_ids[i]: [x1, y1, x2, y2]})
                        cropped_img = frame[y1:y2, x1:x2]
                        cv2.imwrite(dataset_dir + f"cropped_img{COUNT}.jpg", cropped_img)
                        COUNT += 1
                        
            # cv2.imshow("Frame", frame)

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
