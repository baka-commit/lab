# -*- coding: utf-8 -*-
# Experiment 3 - Artificial Neural Network
# =========================================

##exp 3.1 - Implement coding our first neurons

###exp 3.1.1 - Single Neuron

import numpy as np

# Single Neuron
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

###exp 3.1.2 - Neuron with n Inputs

import numpy as np

# Neuron with 4 Inputs
print("\nNeuron with 4 Inputs")
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = (inputs[0]*weights[0] +
          inputs[1]*weights[1] +
          inputs[2]*weights[2] +
          inputs[3]*weights[3] + bias)

print(output)

#done no need for rest of the code
# Neuron with 5 Inputs
# print("\nNeuron with 5 Inputs")

# inputs = [1, 2, 3, 2.5, 4]
# weights = [0.2, 0.8, -0.5, 1.0, 0.4]
# bias = 2

# output = (inputs[0]*weights[0] +
#           inputs[1]*weights[1] +
#           inputs[2]*weights[2] +
#           inputs[3]*weights[3] +
#           inputs[4]*weights[4] +
#           bias)

# print(output)

# # Neuron with 6 Inputs
# print("\nNeuron with 6 Inputs")

# inputs = [1, 2, 3, 2.5, 4, 5]
# weights = [0.2, 0.8, -0.5, 1.0, 0.4, -0.2]
# bias = 2

# output = (inputs[0]*weights[0] +
#           inputs[1]*weights[1] +
#           inputs[2]*weights[2] +
#           inputs[3]*weights[3] +
#           inputs[4]*weights[4] +
#           inputs[5]*weights[5] +
#           bias)

# print(output)

###exp 3.1.3 - Layer of Neurons with Numpy

import numpy as np

# Layer of Neurons
inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)

###exp 3.1.4 - Layer of Neurons with for loop

import numpy as np

# Layer of Neurons without numpy
inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]

outputs = []

for i in range(len(weights)):
    output = 0

    for j in range(len(inputs)):
        output += inputs[j] * weights[i][j]

    output += biases[i]
    outputs.append(output)

print(outputs)

###exp 3.1.5 - Dot product using Numpy

import numpy as np
# dot product using numpy
a = [1, 2, 3]
b = [4, 5, 6]
print("\nDot product using numpy: ",np.dot(a,b))

###exp 3.1.6 - Dot product: Layer of Neurons

import numpy as np

# Dot product: Layer of Neurons
x = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6]
]
w = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9],
  [1.0, 1.1, 1.2]
]

output = np.dot(x,w)
print("\nDot product of Layer of Neurons :\n",output)

###exp 3.1.7 - Layer of Neurons & Batch of Data with ReLU Activation

import numpy as np

# Layer of Neurons & Batch of Data
x = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6]
]
w = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9],
  [1.0, 1.1, 1.2]
]
biases = [1, 2, 0.5]

z = np.dot(x, w) + biases

a = np.maximum(0,z)

print("\nInput batch :")
print(x)

print("\nWeights :")
print(w)

print("\nBiases :")
print(biases)

print("\nOutput before activation :")
print(z)

print("\nOutput after Relu activation :")
print(a)

##exp 3.2 - Single Layered Neural Network with 2 input values

import numpy as np

x0 = 1
x1 = 2

x2 = (x0 * 1) + (x1 * 1)

x3 = (x0 * -1) + (x1 * 1)

x4 = (x2 * 2) + (x3 * -1)

print("\nNEURAL NETWORK ")

print("\nInput Layer:")
print("  x0 =", x0)
print("  x1 =", x1)

print("\nHidden Layer:")
print("  x2 =", x2)
print("  x3 =", x3)

print("\nOutput Layer:")
print("  x4 =", x4)

##exp 3.3 - Multi Layered Neural Network with 2 input values

import numpy as np

x0 = 2
x1 = 3

# First hidden layer
x2 = (x0 * 1) + (x1 * 1)
x3 = (x0 * -1) + (x1 * 1)

# Second hidden layer
x4 = (x2 * -2) + (x3 * 1)
x5 = (x2 * 1) + (x3 * 0)

# Output layer
x6 = (x4 * 1) + (x5 * -1)

print("\nNEURAL NETWORK ")

print("\nInput Layer:")
print("x0 =", x0)
print("x1 =", x1)

print("\nHidden Layer 1:")
print("x2 =", x2)
print("x3 =", x3)

print("\nHidden Layer 2:")
print("x4 =", x4)
print("x5 =", x5)

print("\nOutput Layer:")
print("x6 =", x6)
