inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

'''This is the structure of a basic neuron in a neural network. It multiplies the input and weights.
Adds them up and then adds the bias for the output.
Added a fourth input to simulate farther down in the layers.'''

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
print(output)

