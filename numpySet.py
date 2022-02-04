import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)

'''
The first list type determines the shape of the output list.
Since inputs is a 1D Array or Vector using it as the first value will make it expect a Vector
Using weights since it is a 2D Matrix will set the output as a matrix. 
Running otherwise will give a "type" error. 

np.dot(weights, inputs) = [np.dot(weights[0], inputs), np.dot(weights[1], inputs),
 np.dot(weights[2], inputs)] = [2.8, -1.79, 1.885] + [2.0, 3.0, 0.5] = [4.8, 1.21, 2.385]
 
 output = np.dot(weights, inputs) + biases
 >>> array([4.8, 1.21, 2.385])
'''