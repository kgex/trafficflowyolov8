import cv2
from ultralytics import YOLO
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import asyncio

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
video_path = "input\demo.mp4"
cap = cv2.VideoCapture(video_path)

# Get video information (width, height, frames per second)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output\output_video.mp4', fourcc, fps, (width, height))

# OCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

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
        results = model(frame)
        x, y, w, h = results[0].boxes.xyxy[0]
        x, y, w, h = x.cpu().numpy().astype(np.int32), y.cpu().numpy().astype(np.int32), w.cpu().numpy().astype(
            np.int32), h.cpu().numpy().astype(np.int32)
        annotated_frame = results[0].plot()
        crop_img = frame[y:h, x:w]
        plate_text = asyncio.run(function_async2(crop_img))
        print('Plate Text:', plate_text)

        # Draw number plate information on the annotated frame
        cv2.putText(annotated_frame, f'Plate: {plate_text}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Save the annotated frame to the output video
        output_video.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

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
