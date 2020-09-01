import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

input = torch.randn(20, 16, 50, 100)

# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=1)

# 16 input, 33 output, filter is 3*3. 20 number of inputs it will remain the same.
# 20 (number of input), 33 (output), 50-3 / 1 +1 = 48, 100-3/1 +1 = 98
output = m(input)  # 20,33,48,98

# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
output = m(input) # 20, 33, 28, 100

# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
output = m(input) # 20, 33, 26, 100