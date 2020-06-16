import torch
from torch import nn

m = nn.LogSoftmax()
input = torch.randn(2, 3)
output = m(input)

print(m)
print(input)
print(output)

'''
tensor([[ 0.4081, -1.4856,  0.2882],
        [ 1.8226,  1.3984,  0.6116]])
        
ln(e^x1 / e^x1 + e^x2 + e^x3)

tensor([[-0.7118, -2.6054, -0.8316],
        [-0.6690, -1.0931, -1.8799]])
'''