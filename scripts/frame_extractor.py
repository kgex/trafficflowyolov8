import cv2 

cap = cv2.VideoCapture("C:\Users\Nawin\Projects\trafficflowyolov8\input\Morning Footage.mp4")


while cap.isOpened():
    success, frame = cap.read()
    if success:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break