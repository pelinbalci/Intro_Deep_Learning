from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

features = images.view(images.shape[0], 784)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_1 = nn.Linear(784, 128)  # Linear Transformation.
        self.hidden_2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Pass hidden and output layer into sigmoid and softmax functions.
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.softmax(self.output(x))
        return x


model = Network()
y = model.forward(features)

print('done')