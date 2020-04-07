import cv2
import numpy as np
trait=[]
# load image as YUV (or YCbCR) and select Y (intensity)
# or convert to grayscale, which should be the same.
# Alternately, use L (luminance) from LAB.
img = cv2.imread("C:/Users/Vedant/Desktop/tren_1141.jpg")
Y = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# compute min and max of Y
min = np.min(Y)
max = np.max(Y)
a=max-min
b=int(max)+int(min)
# compute contrast
contrast = a/b
#print(contrast)
if contrast<0.35:
	trait.append("reliable35")
elif contrast<0.75 and contrast>0.35:
	trait.append("reliable75")
else:
	trait.append("reliable100")
print(trait)