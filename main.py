import sys
import cv2 as cv
import pytesseract
import time
import numpy as np

#Ubuntu
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def show_img(img):
    cv.imshow('Imagem', img)
    cv.waitKey(2000)
    sys.exit()

def main():

    if(len(sys.argv) < 1):
        print("Erro: caminho não especificado")
        sys.exit()

    #Lê a imagem
    img_path = sys.argv[1]
    img = cv.imread(img_path)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_gauss_blur = cv.GaussianBlur(img_gray, (7,7), 0)

    img_canny = cv.Canny(img_gauss_blur, 50, 150)

    img_inverted = cv.bitwise_not(img_canny)

    __, img_binary = cv.threshold(img_inverted, 1, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)
        
        if len(approx) == 4:  # Supõe que o conteúdo tem quatro lados
            cv.drawContours(img_binary, [approx], 0, (0, 255, 0), 2)

    bigger_contour = None
    bigger_area = 0
    for contour in contours:

        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)
        
        if len(approx) == 4:  # Supõe que o conteúdo tem quatro lados
            area = cv.contourArea(contour)
            if area > bigger_area:
                bigger_contour = approx
                bigger_area = area

    mask = np.zeros_like(img_gray)

    cv.drawContours(mask, [bigger_contour], 0, 255, -1)

    img_cut = cv.bitwise_and(img, img, mask=mask)

    img_cut_gray = cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY)

    _, img_otsu = cv.threshold(img_cut_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)

    img_otsu_erode = cv.erode(img_otsu, kernel, 18)

    # show_img(img_otsu_erode)

    text= pytesseract.image_to_string(img_otsu_erode, config='--psm 11', lang='eng')
    print("detected " + text)

main()