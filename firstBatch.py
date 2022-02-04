import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
'''
In the back end the np.dot converts these values to an array. 
Since we need to transpose the the weights we first convert it to an array.
Then we use the .T function to transpose this array. 
We need to transpose so the inputs and weights calculate properly. 
'''
output = np.dot(inputs, np.array(weights).T) + biases
print(output)

'''
Nothing needs to be changed in the weights and biases aka our layer. 
This is because altho we added a batch the number of actual neurons did not change.

How this functions in order
1. Converts the weights matrix to an array and transposes them
2. Does the .dot function on the now transposed weights and inputs
[ 0.2, 0.8, -0.5, 1.0 ]                 [ 0.2, 0.5 , -0.26 ]  
[ 0.5, -0.91, 0.26, -0.5 ]       --->   [ 0.8, -0.91, -0.27 ]
[ -0.26, -0.27, 0.17, 0.87 ]            [ -0.5, 0.26, 0.17 ]
                                        [ 1.0, -0.5, 0.87 ]
                                      
This allows for the .dot to work properly 

[ 1, 2, 3, 2.5 ]                   [ 0.2, 0.5 , -0.26 ]  
[ 2.0, 5.0, -1.0, 2.0 ]     *      [ 0.8, -0.91, -0.27 ]
[ -1.5, 2.7, 3.3, -0.8 ]           [ -0.5, 0.26, 0.17 ]
                                   [ 1.0, -0.5, 0.87 ]

Sum of this and then we add the biases for the result

[ 2.8, -1.79, 1.885 ]
[ 6.9, -4.81, -0.3 ]        +      [ 2.0, 3.0, 0.5 ]
[ -0.59, -1.949, -0.474 ]

Final result will be 

[ 4.8, 1.21, 2.385 ]
[ 8.9, -1.81, -0.2 ]
[ 1.41, 1.051, 0.026 ]
'''