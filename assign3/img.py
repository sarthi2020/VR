
# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
import imutils


image=cv2.imread("Bookshelf.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite("blurred.png",blurred)


thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imwrite("Thresh.png", thresh)

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite("Output.png", output)

# we apply erosions to reduce the size of foreground objects
# mask = thresh.copy()
# mask = cv2.erode(mask, None, iterations=5)
# cv2.imwrite("Eroded.png", mask)


# # similarly, dilations can increase the size of the ground objects
# mask = thresh.copy()
# mask = cv2.dilate(mask, None, iterations=5)
# cv2.imwrite("Dilated.png", mask)

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

print(len(cnts))

# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (240, 0, 159), 25)
	cv2.imwrite("Contours.png", output)


   
# cv2.waitKey(0)