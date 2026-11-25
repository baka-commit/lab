# -*- coding: utf-8 -*-
# Experiment 10 - Image Classification Model using VGG-16 for Dog vs Cat
# =======================================================================

import requests
import zipfile
import io

# Kaggle download URL
url = "https://storage.googleapis.com/kaggle-data-sets/8537391/13450138/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20261103%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20261103T110051Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1c0cf9fd3660d0fc723eb05092a4965cf56c5aaae8f09a7110ce1be1195207e061e5908d494b3626c206b943c93d628b9c791f29a98d6b1a622abba2cfc4006864ad7c7b3b88c7e45782fc0c8837caa99ebd0a4172e4a11bc11271ba1f2dc10bca8a884611a89a8d853c369cea41fca2a38b5f2b35cdd89dbb362f510063a760a19e79d80740f6631e5bb4664bceb6f77595ebf33727838285fd1e54b865adf810be4817c0982a9c91a03287ce80f3b344c7cd68d1c6a4c4a07245ca77edbae49034494bf946b8e836e699fb0121e95b30da32a7d4d0a6cdfc140bbeb51106c6c9ca33c51fd3e4a20595b9169744fad994ffa993c69a58850ee40178127a99e3"

# Download ZIP file
response = requests.get(url)
response.raise_for_status()

# Extract ZIP file
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall("/content/dogs-vs-cats")

print("Download and extraction completed!")

import zipfile
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

#with zipfile.ZipFile('/content/archive.zip', 'r') as zip_ref:
#     zip_ref.extractall('/content/dogs-vs-cats')

train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/dogs-vs-cats/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset='training',
    seed=123
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='/content/dogs-vs-cats/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Normalization
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# create CNN model

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=5,validation_data=validation_ds)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

#Validation on new image
import cv2
test_img = cv2.imread('/content/cat.jpg')
plt.imshow(test_img)

test_img.shape

# Convert BGR to RGB
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Resize and normalize
test_img = cv2.resize(test_img, (256, 256)) / 255.0

# Reshape into 4D
test_input = test_img.reshape(1, 256, 256, 3)

# Predict
prediction = model.predict(test_input)

# Display result
print("Prediction:", "Dog" if prediction[0][0] >= 0.5 else "Cat")
