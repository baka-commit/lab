# -*- coding: utf-8 -*-
# Experiment 5 - Backpropagation Neural Network using Python
# ===========================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=load_iris()

x=data.data
y=data.target

y=pd.get_dummies(y, dtype='int').values
y[:3]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

input_size = 4
hidden_size = 2
output_size = 3
learning_rate = 0.05
iterations = 350

results = pd.DataFrame(columns=["mse", "accuracy"])

np.random.seed(10)

w1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

w2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

#feedforward using test data
z1 = np.dot(X_test, w1)
a1 = sigmoid(z1)

z2 = np.dot(a1, w2)
a2 = sigmoid(z2)

acc = accuracy(a2, y_test)
print("Accuracy: {}".format(acc))

# Backpropagation neural network
for itr in range(iterations):

    # Feedforward propagation
    # Hidden layer
    x1 = np.dot(X_train, w1)
    a1 = sigmoid(x1)

    # Output layer
    x2 = np.dot(a1, w2)
    y_pred = sigmoid(x2)

    # Calculate error
    mse = mean_squared_error(y_train, y_pred)
    results.loc[itr, "mse"] = mse

    # Calculate accuracy
    acc = accuracy(y_train, y_pred)
    results.loc[itr, "accuracy"] = acc

    # Backpropagation
    e1 = y_pred - y_train
    dw1 = e1 * y_pred * (1 - y_pred)

    e2 = np.dot(dw1, w2.T)
    dw2 = e2 * a1 * (1 - a1)

    # Weight updating
    w2_update = np.dot(a1.T, dw1)
    w1_update = np.dot(X_train.T, dw2)

    w2 = w2 - learning_rate * w2_update / len(X_train)
    w1 = w1 - learning_rate * w1_update / len(X_train)

# Test the trained network
a1_test = sigmoid(np.dot(X_test, w1))
y_test_pred = sigmoid(np.dot(a1_test, w2))

print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test Accuracy:", accuracy(y_test, y_test_pred))

# Plot MSE and accuracy
plt.figure(figsize=(7, 10))

# Plot MSE
plt.subplot(2, 1, 1)
plt.plot(results["mse"], label="Loss (MSE)", color="red")
plt.title("Training Loss over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.legend()

# Plot Accuracy
plt.subplot(2, 1, 2)
plt.plot(results["accuracy"], label="Accuracy", color="blue")
plt.title("Training Accuracy over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
