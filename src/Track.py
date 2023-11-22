import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "input\demo.mp4"
cap = cv2.VideoCapture(video_path)
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Define colors for different vehicles
color_dict = {2: (0, 255, 0),  # car
              3: (255, 0, 0),  # bike
              5: (0, 0, 255),  # bus
              7: (255, 255, 0)  # truck
              }

# Track history dictionary
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Check if boxes are not None before accessing attributes
        if results[0].boxes is not None:
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
                    if cls==2:
                        class_name = f"Car: {cls}"
                    elif cls==3:
                        class_name = f"Bike: {cls}"
                    elif cls==5:
                        class_name = f"Bus: {cls}"
                    elif cls==7:
                        class_name = f"Truck: {cls}"
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                    print(f"Vehicle detected: {class_name}")

        # Display the annotated frame
        resized_frame = cv2.resize(frame, (3000, 1600))  # Adjust the window size as needed
        cv2.imshow("YOLOv8 Tracking", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
