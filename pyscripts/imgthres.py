import cv2
import numpy as np
import os

img = cv2.imread("C:/Users/Vedant/Desktop/train_0.jpg")
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
ret,thresh1 = cv2.threshold(grayscaled,127,255,cv2.THRESH_BINARY)
cv2.imwrite("C:/Users/Vedant/Desktop/hec.jpg",thresh1)