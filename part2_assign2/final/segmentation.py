# To Run the code first run the command 
# wget "https://pjreddie.com/media/files/yolov3.weights" to get yolov3 weights

import cv2
import numpy as np
import argparse
import os


def Segmentation(Img):
    Image = "images/{}".format(Img)
    ##Object Localization ie to get a bounding box around the image
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_name = net.getLayerNames()
    output_layers = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread(Image)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if boxes==[]:
        boxes=[[77,30,270,110]]

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # GrabCut algorithm to get the outputMask
    image = cv2.imread(Image)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    if boxes==[]:
        rect=(77,30,270,110)
    else:
        rect = (boxes[0][0],boxes[0][1]-20,boxes[0][2],boxes[0][3]+20)

    # img1=cv2.rectangle(image, rect, (216,255,12), 2)
    # cv2.imwrite("{}".format(Img.replace(".jpg",''))+"_BB.png", img1)
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
    	fgModel, iterCount = 10, mode = cv2.GC_INIT_WITH_RECT)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    output = cv2.bitwise_and(image, image, mask=outputMask)

    cv2.imwrite("{}".format(Img.replace(".jpg",''))+"_seg.png", outputMask)
    # cv2.imshow("OutputMask",outputMask)
    # cv2.waitKey(0)
    print("Segmented image {}_seg.png is saved in present working directory".format(Img.replace(".jpg",'')))

images = os.listdir('images')
for img in images:
    if(img.endswith('.jpg')):
        print(img)
        Segmentation(img)
# Segmentation("Image2.jpg")
