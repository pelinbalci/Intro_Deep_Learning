import torch
from torch import nn

logsoftmax_func = nn.LogSoftmax()
softmax_func = nn.Softmax()

input = torch.randn(2, 3)

log_softmax_output = logsoftmax_func(input)
softmax_output = softmax_func(input)

print(input)
print(softmax_output)  # between 0 and 1
print(log_softmax_output)  # negative values.

'''
tensor([[ 0.4081, -1.4856,  0.2882],
        [ 1.8226,  1.3984,  0.6116]])
        
ln(e^x1 / e^x1 + e^x2 + e^x3)

tensor([[-0.7118, -2.6054, -0.8316],
        [-0.6690, -1.0931, -1.8799]])
'''