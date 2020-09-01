
import numpy as np
import matplotlib.pyplot as plt


import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from common.save_fig import save_fig

data = torch.randn(1, 3, 23, 29)  # torch.Size([1, 3, 32, 32])

pool = nn.MaxPool2d(2, 2)

# With square kernels and equal stride
m = nn.Conv2d(3, 16, 3, padding=1)
output1 = m(data)  # torch.Size([1, 16, 32, 32])

m2 = nn.Conv2d(16, 32, 3, padding=1)
output2 = m2(output1)  # torch.Size([1, 32, 32, 32])

m3 = nn.Conv2d(32, 64, 3, padding=1)
output3 = m3(output2) # torch.Size([1, 64, 32, 32])

print(data)
