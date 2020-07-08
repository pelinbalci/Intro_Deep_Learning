from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

'''
Model Requirements:
initialize weight, bias
input layer --> 784 units convert 28 x 28 image to 784 unit image.
image shape is [64, 1, 28, 28] --> [64, 784]
hidden layer --> 256 units
output layer --> 10 units (one for each number)
'''

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)  # [64, 1, 28, 28] 64 images, 1 colorchannels: greyscale, 28 x 28 pixels.
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()


def activation(x):
    return 1/(1+torch.exp(-x))


features = images.view(images.shape[0], 784)
# inputs = images.view(images.shape[0], -1)  #  --> 64,784

# Define the size of (number of neurons) each layer:
n_input = features.shape[1]  # same shape with features
n_hidden = 256
n_output = 10

w1 = torch.randn(n_input, n_hidden)
w2 = torch.randn(n_hidden, n_output)

b1 = torch.randn(1, n_hidden)
b2 = torch.randn(1, n_output)


h = activation(torch.mm(features, w1) + b1)
out = torch.mm(h, w2) + b2


def softmax(x):
    y3 = torch.exp(x) / torch.sum(torch.exp(out), dim=1).view(-1, 1)
    return y3

y3 = softmax(out)
print(y3[1])

# Explanation:

exp_x = torch.exp(out)  # 64,10

# exp_x has 64 row and 10 column. first row includes 10 outputs for 10 numbers.
# we need to divide each element in the first row / sum first row.
# sum shape should be 64 row and 1 column.
sum_exp_x = torch.sum(exp_x, dim=1)  # shape: 1 row, 64 column.


# shape_sum_exp_x = sum_exp_x.view(1, -1).T --> same.
shape_sum_exp_x = sum_exp_x.view(-1, 1)  # shape: 64 row, 1 column


y2 = exp_x / shape_sum_exp_x
