import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import math
personality={}
def printpersonality():
	print(personality)
#thresholding the image 
def thres(img):
	grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	ret,thresh1 = cv2.threshold(grayscaled,127,255,cv2.THRESH_BINARY)
	return thresh1
#crop the image into square to feed to introextro neural network
def introcrop(img):
	smaller=947904723979349319439844

	path=img
	imig=cv2.imread(path)
	img=thres(imig)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if (img[i,j].any()==0) and j<smaller:
				smaller=j
			else: pass

	croppie=img[(smaller-120):(smaller+420), (smaller-2):(smaller+420)]
	return(croppie)
#introvert or extrovert
def introextro(img,path):
	train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/introxx.csv')
	train_image=[]
	model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/intmodel.h5')
	img1=image.img_to_array(introcrop(path))

	classes = np.array(train.columns[1:])
	proba = model.predict(img1.reshape(1,464,465,3))	
	for i in range(len(classes)):
		print("{}".format(classes[i])+" ({})".format(proba[0][i]))
	introextro={classes[0]:proba[0][0],classes[1]:proba[0][1]}

	if introextro[classes[0]]>introextro[classes[1]]:
		personality.update({classes[0]:introextro[classes[0]]})
	else:
		personality.update({classes[1]:introextro[classes[1]]})

#skew detection algorithm
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
#calculate confidence by skew detection
def confidence(img):
	file_path =img
	src=cv2.imread(file_path)  
	angle = compute_skew(thres(src))
	if angle<(-20.00):
		personality.update({"confident":angle})
	else:
		personality.update({"not_confident":angle})
#calculating penpressure
def reliable(img):
	img = cv2.imread(img)
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
		personality.update({"reliable35":contrast})
	elif contrast<0.75 and contrast>0.35:
		personality.update({"reliable75":contrast})
	else:
		personality.update({"reliable100":contrast})


def firm(img):
	train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/firm.csv')
	train_image=[]
	model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/firmmodel.h5')
	img = image.load_img(img,target_size=(280,498,3))
	img = image.img_to_array(img)
	classes = np.array(train.columns[1:])
	proba = model.predict(img.reshape(1,280,498,3))
	for i in range(len(classes)):
		print("{}".format(classes[i])+" ({})".format(proba[0][i]))
	firm={classes[0]:proba[0][0],classes[1]:proba[0][1]}
	if firm[classes[0]]>firm[classes[1]]:
		personality.update({classes[0]:firm[classes[0]]})
	else:
		personality.update({classes[1]:firm[classes[1]]})
def overthink(img):
	train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/overthink.csv')
	train_image=[]
	model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/ovethink.h5')
	img = image.load_img(img,target_size=(280,498,3))
	img = image.img_to_array(img)
	classes = np.array(train.columns[1:])
	proba = model.predict(img.reshape(1,280,498,3))
	for i in range(len(classes)):
		print("{}".format(classes[i])+" ({})".format(proba[0][i]))
	firm={classes[0]:proba[0][0],classes[1]:proba[0][1]}
	if overthink[classes[0]]>overthink[classes[1]]:
		personality.update({classes[0]:overthink[classes[0]]})
	else:
		personality.update({classes[1]:overthink[classes[1]]})


def warmfriendly(img):
	train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/warmfriend.csv')
	train_image=[]
	model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/warmfriend.h5')
	img = image.load_img(img,target_size=(280,498,3))
	img = image.img_to_array(img)
	classes = np.array(train.columns[1:])
	proba = model.predict(img.reshape(1,280,498,3))
	for i in range(len(classes)):
		print("{}".format(classes[i])+" ({})".format(proba[0][i]))
	warmfriendly={classes[0]:proba[0][0],classes[1]:proba[0][1]}
	if warmfriendly[classes[0]]>warmfriendly[classes[1]]:
		personality.update({classes[0]:warmfriendly[classes[0]]})
	else:
		personality.update({classes[1]:warmfriendly[classes[1]]})

def commercial_instincts(img):
	train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/commercial.csv')
	train_image=[]
	model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/commercial.h5')
	img = image.load_img(img,target_size=(280,498,3))
	img = image.img_to_array(img)
	classes = np.array(train.columns[1:])
	proba = model.predict(img.reshape(1,280,498,3))
	for i in range(len(classes)):
		print("{}".format(classes[i])+" ({})".format(proba[0][i]))
	commercial_instincts={classes[0]:proba[0][0],classes[1]:proba[0][1]}
	if commercial_instincts[classes[0]]>commercial_instincts[classes[1]]:
		personality.update({classes[0]:commercial_instincts[classes[0]]})
	else:
		personality.update({classes[1]:commercial_instincts[classes[1]]})
def job_completion(img):
	train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/jobcomplete.csv')
	train_image=[]
	model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/jobcomplete.h5')
	img = image.load_img(img,target_size=(280,498,3))
	img = image.img_to_array(img)
	classes = np.array(train.columns[1:])
	proba = model.predict(img.reshape(1,280,498,3))
	for i in range(len(classes)):
		print("{}".format(classes[i])+" ({})".format(proba[0][i]))
	job_completion={classes[0]:proba[0][0],classes[1]:proba[0][1]}
	if job_completion[classes[0]]>job_completion[classes[1]]:
		personality.update({classes[0]:job_completion[classes[0]]})
	else:
		personality.update({classes[1]:job_completion[classes[1]]})

def influential(img):
	train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/influencer.csv')
	train_image=[]
	model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/influencer.h5')
	#img = image.load_img('C:/Users/Vedant/Desktop/tpax/tren0.jpg',target_size=(280,498,3))
	img = image.img_to_array(img)

	classes = np.array(train.columns[1:])
	proba = model.predict(img.reshape(1,280,498,3))
	top_3 = np.argsort(proba[0])[:-4:-1]

	for i in range(len(classes)):
		print("{}".format(classes[i])+" ({})".format(proba[0][i]))
	influential={classes[0]:proba[0][0],classes[1]:proba[0][1]}
	if influential[classes[0]]>influential[classes[1]]:
		personality.update({classes[0]:influential[classes[0]]})
	else:
		personality.update({classes[1]:influential[classes[1]]})

path="C:/Users/Vedant/Desktop/image_6.jpg"
rawimage = image.load_img(path,target_size=(280,498,3))
thresim=thres(rawimage)
finalimage = image.img_to_array(thresim)
introextro(thresim,path)
confidence(path)
reliable(path)
firm(finalimage)
overthink(finalimage)
warmfriendly(finalimage)
commercial_instincts(finalimage)
job_completion(finalimage)
influential(finalimage)
printpersonality()