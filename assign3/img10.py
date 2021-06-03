import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt 

image = cv2.imread("new.jpg")
rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


light = (60, 90, 60)
dark = (200, 180, 170)

mask = cv2.inRange(rgb, light, dark)
result = cv2.bitwise_and(rgb, rgb, mask=mask)

plt.imshow(mask, cmap="gray")
plt.show()
 