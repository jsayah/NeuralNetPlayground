import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)  # This replaces the loop below
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Runs NP sum at axis 1 and keep dimensions True

print(norm_values)
#print(sum(norm_values))

'''for output in layer_outputs:  # For loop to exponentiate the layer outputs
    exp_values.append(E**output)'''

'''norm_values = exp_values / np.sum(exp_values)  # Replaces function below'''

'''norm_base = sum(exp_values)  # Normalization happens after exponentiation
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)'''

