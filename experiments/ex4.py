# -*- coding: utf-8 -*-
# Experiment 4 - Activation Functions in Building Neural Network
# ===============================================================

import numpy as np

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x0 = 2
x1 = 3

# First hidden layer
x2 = relu((x0 * 1) + (x1 * 1))
x3 = relu((x0 * -1) + (x1 * 1))

# Second hidden layer
x4 = relu((x2 * -2) + (x3 * 1))
x5 = relu((x2 * 1) + (x3 * 0))

# Output layer
x6 = sigmoid((x4 * 1) + (x5 * -1))

print("\nActivation Functions in NEURAL NETWORK ")

print("\nInput Layer:")
print("x0 =", x0)
print("x1 =", x1)

print("\nHidden Layer 1: (RELU Activation)")
print("x2 =", x2)
print("x3 =", x3)

print("\nHidden Layer 2: (RELU Activation)")
print("x4 =", x4)
print("x5 =", x5)

print("\nOutput Layer:")
print("x6 =", x6)
