# -*- coding: utf-8 -*-
# Experiment 6 - ANN model to IRIS Flower dataset
# ================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()

print("Feature Names :", iris.feature_names)
print("Target Names :", iris.target_names)
print("Target Array : \n", iris.target)
print("Data :\n", iris.data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create ANN
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Accuracy :", accuracy)
print("\nTest Loss :", loss)

# Predict
pred = model.predict(X_test, verbose=0)
# print("Predicted:", tf.argmax(pred, axis=1).numpy())

print("\nPredicted Classes :")
print(pred.argmax(axis=1))
print("\nActual Classes :")
print(y_test)
