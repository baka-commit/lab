# -*- coding: utf-8 -*-
# Experiment 7 - ANN model for classify the house price prediction
# =================================================================

import pandas as pd
df=pd.read_csv('/content/housepricedata.csv')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Take independent features as x and target as y
x=df.iloc[:,0:10]
y=df.iloc[:,10]
print(x)
print(y)

#Scale the data from 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_scale=scaler.fit_transform(x)
print(x_scale)

#Let us split the dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=1)

#Creating ANN model
model = Sequential()
input_layer = Dense(20, input_shape=(10,))
model.add(input_layer)
hidden_layer1 = Dense(20, activation='relu')
model.add(hidden_layer1)
hidden_layer2 = Dense(20, activation='relu')
model.add(hidden_layer2)

#Create Output layer
output_layer = Dense(1, activation='sigmoid')
model.add(output_layer)

model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=10)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

#Make prediction
lst = [[8450, 7, 5, 856, 2, 1, 3, 8, 0, 548]]
scaled_data = scaler.transform(lst)
print(scaled_data)

result = model.predict(scaled_data)
if result >= 0.5:
    print('1')
else:
    print('0')

#fit() method returns the total history of what happend during tarining
hist = model.fit(x_train, y_train, batch_size=20, epochs=100, validation_data=(x_test, y_test))

#Visulaize the accuracy in each epoch
import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'], label='Acc')
plt.plot(hist.history['val_accuracy'], label='Val_acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Visulaize the Loss in each epoch
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.title('Model Loss')
plt.xlabel('Accuracy')
plt.legend()
plt.show()
