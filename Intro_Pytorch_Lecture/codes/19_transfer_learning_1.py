# Ref: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%207%20-%20Loading%20Image%20Data%20(Exercises).ipynb
# Ref: https://pytorch.org/docs/0.3.0/torchvision/models.html


import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F

from Intro_Pytorch.common import helper, nn_model

data_dir = 'Cat_Dog_data'

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

test_transform = transforms.Compose([transforms.RandomRotation(30),
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])


train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

model = models.densenet121(pretrained=True)

print(model)

'''
DenseNet(
  (features): Sequential(
    (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu0): ReLU(inplace=True)
    (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (denseblock1): _DenseBlock(
      (denselayer1): _DenseLayer(
        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu1): ReLU(inplace=True)
        (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu2): ReLU(inplace=True)
        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )...
      (denselayer6):_DenseLayer( ... )
      
      (transition1): _Transition(
      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    (denseblock2): _DenseBlock(
      
      (denselayer1): _DenseLayer(
        (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu1): ReLU(inplace=True)
        (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu2): ReLU(inplace=True)
        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      ) ...
      (denselayer12):
    (transition2):
    
    (denseblock3):
      (denselayer1):
      (denselayer24):
    (transition3):
    
    (denseblock4):
      (denselayer1):
      (denselayer16):..)
    
      (classifier): Linear(in_features=1024, out_features=1000, bias=True)

model includes sequential layers and classifier. 
Sequential layers have 4 dense block and 3 transitions. 
Dense layers in blocks are: [6, 12, 24, 16]
In each dense layers there are: [norm1, relu1, conv1, norm2, relu2, conv2]

We need to refactor classifier part, since we don't need 1000 output. 
'''

# Freeze parameeters, we don't backprop trhrough them:
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('lin1', nn.Linear(1024, 500)),
                                        ('relu', nn.ReLU()),
                                        ('lin2', nn.Linear(500, 2)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))

model.classifier = classifier

'''
However, now we're using a really deep neural network. 
If you try to train this on a CPU like normal, it will take a long, long time. 

Instead, we're going to use the GPU to do the calculations. 

The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. 
It's also possible to train on multiple GPUs, further decreasing training time.

GPU: Graphics proccessing unit
CPU: Central proccessing unit

'CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical 
processing units (GPUs). 
With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.
(https://developer.nvidia.com/cuda-zone)'


'''

import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for device in ['cpu']:

# I do't have cuda
# TODO: cuda?

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(train_loader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii == 3:
            break

    print(f"Device = {device}; Time per batch: {(time.time() - start) / 3:.3f} seconds")

print('done')