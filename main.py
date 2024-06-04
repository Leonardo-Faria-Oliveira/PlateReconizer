import sys
import cv2 as cv
import pytesseract
import time
import numpy as np
from ultralytics import YOLO
import re

#Ubuntu
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

#Win
#pytesseract.pytesseract.tesseract_cmd = 'C:/User/<nome>/AppData/tesseract'


# Load the models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

def crop_license_plate(image):

    # Detect license plates
    results = license_plate_detector.predict(image)
    
    # Assuming there's at least one detection, get the first detection (highest confidence)
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get the bounding box coordinates of the first detected license plate
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Crop the image using the bounding box coordinates
        cropped_image = image[y1:y2, x1:x2]
        
        return cropped_image

    else:
        print('No license plate detected')
        return false

def show_img(img):
    cv.imshow('Imagem', img)
    cv.waitKey(2000)
    sys.exit()


def main():

    if(len(sys.argv) < 1):
        print("Erro: caminho não especificado")
        sys.exit()

    img_path = sys.argv[1]
    img = cv.imread(img_path)
    img_cut = crop_license_plate(img)

    img_cut_gray = cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY)

    img_cut_gray = cv.bilateralFilter(img_cut_gray, 11, 17, 17)

    _, img_otsu = cv.threshold(img_cut_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    img_erode = cv.erode(img_otsu, kernel, iterations=1)

    custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_char_blacklist=-/, '
    text = pytesseract.image_to_string(img_erode, config=custom_config, lang='eng')
    print(text)
    

main()
 