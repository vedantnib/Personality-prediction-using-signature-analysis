import numpy as np
import math
import cv2

def compute_skew(file_name):
    
    #load in grayscale:
    src=cv2.cvtColor(file_name,cv2.COLOR_BGR2GRAY)
    #src = cv2.imread(img)
    height, width = src.shape[0:2]
    
    #invert the colors of our image:
    cv2.bitwise_not(src, src)
    
    #Hough transform:
    minLineLength = width/2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src,1,np.pi/180,100,minLineLength,maxLineGap)
    
    #calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)
    
    
    for line in lines:
        angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0);
    
    angle /= nb_lines*1.0
    
    return angle* 180.0 / np.pi

file_path ='C:/Users/Vedant/Desktop/image_6.jpg'
src=cv2.imread(file_path)  
angel = compute_skew(src)
print(angel)
if angel<(-20.00):
    print("confident")
#cv2.waitKey(0)