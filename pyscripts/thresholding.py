import cv2
import numpy as np
import os
def thres(imig,newpath,nem):
	img = cv2.imread(imig)
	grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#previously 127
	ret,thresh1 = cv2.threshold(grayscaled,132,255,cv2.THRESH_BINARY)
	cv2.imwrite(newpath+nem,thresh1)

path="C:/Users/Vedant/Desktop/warmfriend/"
newpath="C:/Users/Vedant/Desktop/wthres/"
j=1
for i in range(1,149):
	nem="train_"+str(j)+".jpg"
	imig=path+nem
	thres(imig,newpath,nem)
	j+=1