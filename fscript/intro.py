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
personality={}
def pp():
	print(personality)
train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/introxx.csv')
train_image=[]
model = keras.models.load_model('C:/Users/Vedant/Desktop/beproject/neumod/intmodel.h5')
img = image.load_img('C:/Users/Vedant/Desktop/tpax/ks.jpg',target_size=(464,465,3))
img = image.img_to_array(img)
#img = img/255
classes = np.array(train.columns[1:])
proba = model.predict(img.reshape(1,464,465,3))
top_3 = np.argsort(proba[0])[:-4:-1]
#for i in range(2):
#    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
for i in range(len(classes)):
	print("{}".format(classes[i])+" ({})".format(proba[0][i]))
introextro={classes[0]:proba[0][0],classes[1]:proba[0][1]}

print(introextro)
#trait=introextro[classes[0]]
if introextro[classes[0]]>introextro[classes[1]]:
	personality.update({classes[0]:introextro[classes[0]]})
else:
	personality.update({classes[1]:introextro[classes[1]]})
pp()
#plt.imshow(img)
#plt.show() 