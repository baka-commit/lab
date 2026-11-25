# -*- coding: utf-8 -*-
# Experiment 9 - Image Classification Model using CNN for CIFAR-10
# =================================================================

from tensorflow.keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

#First column is related to Red, Second column is realted to Green and
#Third column is related to Blue
print(x_train[0])

#These RGB pixels values are spanning from 0 to 255
#Since there is much variation in the values, it is better to normalize
#them by fiiting them between 0 to 1
#This can be done by simply dividing the train and test by 255 as:
x_train,x_test = x_train/255.0, x_test/255.0
print(x_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()

arr=y_train[0]
print(arr)

import matplotlib.pyplot as plt
plt.imshow(x_train[10])
plt.show()

arr=y_train[10]
print(arr)

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(10,10)) #10x10 inches of width and height of the plot
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])

from tensorflow.keras import utils
y_train = utils.to_categorical(y_train,10)
y_test = utils.to_categorical(y_test,10)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
#Build our CNN model
model = Sequential(
    [
        Dense(64,input_shape=(32,32,3)),
        Conv2D(64,kernel_size=(3,3),activation='relu'),
        Conv2D(64,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128,kernel_size=(3,3),activation='relu'),
        Conv2D(64,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.5),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(10,activation='softmax')
    ]
)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train=model.fit(x_train,y_train,batch_size=32,epochs=10,verbose=1,validation_data=(x_test,y_test))

model.summary()

val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss)
print(val_acc)

#Make Prediction
arr = model.predict([x_train[0].reshape(1,32,32,3)])
print(arr)

import numpy as np
label = np.argmax(arr)
print(label)
