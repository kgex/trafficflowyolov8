from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('yolov8n.pt')

# Read the input image
frame = cv2.imread("traffic.jpg")

# Run the tracker
results = model.track(source="traffic.jpg")

# Loop through the detected objects and draw bounding boxes
for i in range(len(results[0].boxes)):
    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy.cpu()[i])
    cls = results[0].boxes.cls[i].item()

    # Draw bounding box based on the detected class
    if cls == 2:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("Car detected")
    elif cls == 3:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("Bike detected")
    elif cls == 5:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("Bus detected")
    elif cls == 7:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("Truck detected")

# Display the annotated frame
cv2.imshow("YOLOv8 Tracking", frame)

# Wait for a key press and close the window if 'q' is pressed
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()
