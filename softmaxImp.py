import numpy as np
import nnfs
from nnfs.datasets import spiral_data  # import of dataset

nnfs.init()  # Setting the random for the weights and set the data type


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


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)  # Passes in the X variable that is spiral_data
activation1.forward(dense1.output)  # .forward passes in the output of the dense1 function.

dense2.forward(activation1.output)  # Passes result of the activation1() to the dense2 layer
activation2.forward(dense2.output)

print(activation2.output[:5])

''' layer1 = Layer_Dense(2, 5)  # first input is defined by inputs in this case the spiral data set has 2, X and Y
layer2 = Layer_Dense(5, 2)  First input is determined by the second value of the previous layer.
activation1 = Activation_ReLU()  # Will take the number of inputs and do the activation function for each neuron

layer1.forward(X)  # uses the initial input batch for the first layer then uses the forward pass
#print(layer1.output)
activation1.forward(layer1.output)

print(activation1.output)'''
