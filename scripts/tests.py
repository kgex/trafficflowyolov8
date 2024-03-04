import pytest
import cv2
import numpy as np
from paddleocr import PaddleOCR
from main import regex, convert_rgb_to_names, extract_dominant_color, function_async2

ocr = PaddleOCR(use_angle_cls=True, lang="en")

@pytest.fixture
def sample_plate_image():
    # Create a sample plate image for testing
    plate_image = cv2.imread("/home/kgx/Projects-Nawin/Thingsboard/trafficflowyolov8/docs/plate_img.jpeg")
    ocr_output = ocr.ocr(plate_image)  # ocr output format: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, confidence)] 
    if ocr_output is None:
        return None
    for idx in range(len(ocr_output)):
        res = ocr_output[idx]
        for line in res:
            numberplate = line
            assert numberplate[1][0] == "MH12DE1433"


def test_regex():
    # Test cases for regex function
    assert regex("AB12CD3456") == True
    assert regex("A1B2C345") == False
    assert regex("") == False

def test_convert_rgb_to_names():
    # Test cases for convert_rgb_to_names function
    assert convert_rgb_to_names((255, 255, 255)) == "white"
    assert convert_rgb_to_names((0, 0, 0)) == "black"

def test_extract_dominant_color(sample_plate_image):
    # Test cases for extract_dominant_color function
    vehicle_image = cv2.imread("/home/kgx/Projects-Nawin/Thingsboard/trafficflowyolov8/docs/vehcile.png")
    dominant_color = extract_dominant_color(vehicle_image)
    assert dominant_color == "red"  # Since the sample plate image is filled with white color



# Additional tests can be added for other functions as needed
