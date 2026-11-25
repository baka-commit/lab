# -*- coding: utf-8 -*-
# Experiment 8 - CNN MODEL
# =========================

#Hand written digit image recognition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,MaxPooling2D
import numpy as np
#Load MNIST data for digit recognition from keras datasets
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Display the shape of data
print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_test)

#Reshape the data into 4 dimensions
x_train = x_train.reshape(60000, 28, 28, 1) #1 - no. of channels
x_test = x_test.reshape(10000, 28, 28, 1)
#Convert targets (class vectors) to binary class matrices
from tensorflow.keras import utils
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
#Display a sample train image array and label
arr = y_train[100]
print(arr)

label = np.argmax(arr)
print(label)

#display the image
import matplotlib.pyplot as plt
plt.imshow(x_train[100], cmap='gray')

#Build CNN model
model = Sequential()
input_layer = Dense(32, input_shape = (28, 28, 1))
model.add(input_layer)
conv_layer1 = Conv2D(32, kernel_size = (3,3), activation='relu') #3x3 convolution grid
model.add(conv_layer1)
conv_layer2 = Conv2D(64, kernel_size = (3,3), activation='relu') #3x3 convolution grid
model.add(conv_layer2)
pool_layer = MaxPooling2D(pool_size = (2,2))
model.add(pool_layer)
drop_layer = Dropout(0.5) #To avoid overfitting
model.add(drop_layer)
flat_layer = Flatten()
model.add(flat_layer)
output_layer = Dense(10, activation='softmax')
model.add(output_layer)
#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
#Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=10,
verbose=1, validation_data=(x_test, y_test))

#Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

#Make prediction with 100th test data
arr = model.predict([x_train[100].reshape(1, 28, 28, 1)])
print(arr)

label = np.argmax(arr)
print(label)
