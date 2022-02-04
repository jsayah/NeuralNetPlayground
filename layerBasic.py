import numpy as np
import nnfs
from nnfs.datasets import spiral_data  # import of dataset

nnfs.init()  # Setting the random for the weights and set the data type

'''X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]'''

X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):  # This is defining the shape as n_inputs and n_neurons
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # Setting the shape in the initialization allows
        # us to skip transpose
        self.biases = np.zeros((1, n_neurons))  # This needs to be a toople by using double brackets

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)  # first input is defined by inputs in this case the spiral data set has 2, X and Y
''' layer2 = Layer_Dense(5, 2)  First input is determined by the second value of the previous layer.'''
activation1 = Activation_ReLU()  # Will take the number of inputs and do the activation function for each neuron

layer1.forward(X)  # uses the initial input batch for the first layer then uses the forward pass
#print(layer1.output)
activation1.forward(layer1.output)

print(activation1.output)

'''
Below is basic layering

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.15, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
'''
