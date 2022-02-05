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

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]  # confidence for scaller values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)  # confidence for one hot | inf 0 prot

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

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

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)  # Loss calculation using the output of the last activation()

print("Loss:", loss)

'''import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)'''