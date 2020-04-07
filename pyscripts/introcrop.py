import cv2
import numpy as np
def thres(img):
	grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
	ret,thresh1 = cv2.threshold(grayscaled,127,255,cv2.THRESH_BINARY)
	return thresh1
smaller=947904723979349319439844
#img=cv2.imread("C:/Users/Vedant/Desktop/tren0.jpg")
path="C:/Users/Vedant/Desktop/image_6.jpg"
imig=cv2.imread(path)
img=thres(imig)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		if (img[i,j].any()==0) and j<smaller:
			smaller=j
		else: pass
#cv2.imshow('og',img)
#cropped=img[300:800, 400:800]
croppie=img[(smaller-120):(smaller+420), (smaller-2):(smaller+420)]
cv2.imwrite("C:/Users/Vedant/Desktop/vss.jpg",croppie)
cv2.imshow('crop',croppie)
cv2.waitKey(0)

cv2.destroyAllWindows()