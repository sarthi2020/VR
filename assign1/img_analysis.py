import cv2 

image=cv2.imread("img_analysis.jpg")

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.png",gray)
# cv2.imshow('Grayscale', gray) 
# cv2.waitKey()

smooth=cv2.blur(image,(5,5))
cv2.imwrite("smooth.png",smooth)
# cv2.imshow('Smooth', smooth) 
# cv2.waitKey()

Gsmooth=cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
cv2.imwrite("GaussianBlur.png",Gsmooth)
# cv2.imshow('GSmooth', Gsmooth) 
# cv2.waitKey()

sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
cv2.imwrite("SobelX.png",sobelx)
cv2.imwrite("SobelY.png",sobely)
# cv2.imshow("Sobel X",sobelx)
# cv2.waitKey()
# cv2.imshow("Sobel Y",sobely)
# cv2.waitKey()

cannyedge = cv2.Canny(image,50,100)
cv2.imwrite("Canny.png",cannyedge)
# cv2.imshow("Canny",cannyedge)
# cv2.waitKey()

cv2.destroyAllWindows()