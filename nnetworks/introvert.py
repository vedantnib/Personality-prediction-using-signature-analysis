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
train = pd.read_csv('C:/Users/Vedant/Desktop/beproject/csvs/introxx.csv')
train_image=[]
ignore = ['id']
for i in tqdm(range(train.shape[0])):
    img = image.load_img('C:/Users/Vedant/Desktop/beproject/datasets/introvert/'+str(train['id'][i])+'.jpg',target_size=(464,465,3))
    img = image.img_to_array(img)
#    img = img/255
    train_image.append(img)
X = np.array(train_image)
#plt.imshow(X[2])
#plt.show()
#print(train['breed'][0])
y = np.array(train.drop(['id'],axis=1))
#y.shape=y.shape-2
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.08)
model = Sequential()
model.add(Conv2D(filters=5, kernel_size=(5, 5), activation="relu", input_shape=(464,465,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(Conv2D(filters=4, kernel_size=(5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(filters=4, kernel_size=(5, 5), activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(filters=4, kernel_size=(5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(filters=9, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=15, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
#model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=32, validation_data=(X_test, y_test), batch_size=4)
model.save('C:/Users/Vedant/Desktop/neumod/intmodel.h5')
del model
model = keras.models.load_model('C:/Users/Vedant/Desktop/neumod/intmodel.h5')
img = image.load_img('C:/Users/Vedant/Desktop/nss.jpg',target_size=(464,465,3))
img = image.img_to_array(img)
#img = img/255
classes = np.array(train.columns[1:])
proba = model.predict(img.reshape(1,464,465,3))
top_3 = np.argsort(proba[0])[:-4:-1]
#for i in range(2):
#    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
for i in range(len(classes)):
	print("{}".format(classes[i])+" ({})".format(proba[0][i]))
#plt.imshow(img)
#plt.show() 