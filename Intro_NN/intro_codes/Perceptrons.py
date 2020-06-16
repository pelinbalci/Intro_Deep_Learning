import pandas as pd
import numpy as np

'''
AND Operations 
'''
# TODO: Set weight1, weight2, and bias
weight1 = 1.0
weight2 = 1.0
bias = -1.5

print('AND Operations')

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
result = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    activation_output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if activation_output == correct_output else 'No'
    result.append([test_input[0], test_input[1], linear_combination, activation_output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in result if output[4] == 'No'])
output_frame = pd.DataFrame(result, columns=['Input 1', '  Input 2', '  Linear Combination',
                                              '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


'''
OR Operations 
'''

# TODO: You can increase the weights or you can decrease the magnitude of bias.
# {or_weight1:2, or_weight2:2, or_bias:-1.5}
# {or_weight1:1, or_weight2:1, or_bias:-0.5}

or_weight1 = 2
or_weight2 = 2
or_bias = -1.5

print('OR Operations')
# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
or_test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
or_correct_outputs = [False, True, True, True]
or_result = []

# Generate and check output
for test_input, correct_output in zip(or_test_inputs, or_correct_outputs):
    linear_combination = or_weight1 * test_input[0] + or_weight2 * test_input[1] + or_bias
    activation_output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if activation_output == correct_output else 'No'
    or_result.append([test_input[0], test_input[1], linear_combination, activation_output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in or_result if output[4] == 'No'])
output_frame = pd.DataFrame(or_result, columns=['Input 1', '  Input 2', '  Linear Combination',
                                              '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


'''
NOT Operations
'''

import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 1
weight2 = -2
bias = 0.5

print('NOT Operations')
# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))