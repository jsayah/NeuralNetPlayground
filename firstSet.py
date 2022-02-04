inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []  # Output of current layer. Creating empty list
for neuron_weights, neuron_bias in zip(weights, biases):  # Zip creates a list of list. Zipping weights and biases.
    neuron_output = 0  # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):  # For loops to do the calculation for each value in the list
        neuron_output += n_input*weight  # Creates input for use in functions
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)  # Append function to add output to final outputs list

print(layer_outputs)

'''
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

This utilizes the structure of the neurons we built and uses them in a list to add layering. This simulates the last leg.
This takes 4 inputs to 3 neurons to create a list of values based on the neurons calculation.
This is commented out as the code is shown more efficiently restructured.

output = [ inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
           inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
           inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)
'''
