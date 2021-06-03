import cv2
import numpy as np
from matplotlib import pyplot as plt

def count(file_name):
    img = cv2.imread(file_name)

    img_final = cv2.imread(file_name) 
    
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img2gray = cv2.medianBlur(img2gray,5)
    cv2.imwrite("12-gray.png",img2gray)

    th3 = cv2.adaptiveThreshold(img2gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    cv2.imwrite("12-threshold.png",th3)

    image_final = cv2.bitwise_and(img2gray, img2gray, mask=th3)
    cv2.imwrite("12-imagefinal.png",image_final)

    ret, new_img = cv2.threshold(image_final, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("12-new_img.png",new_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("12-closed.png",closed)    

    (contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    

    total = 0
    rect_list = []
    for contour in contours:

        [x, y, w, h] = cv2.boundingRect(contour)

        # remove conotours with smaller and larger dimensions.
        if w < 60 or h < 100:
            continue

        if w > 1000 or h > 1000:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 9)
        total+=1
    cv2.imwrite('12-boxed.png', img)
    print(total)


file_name = 'Bookshelf.jpg'
count(file_name)