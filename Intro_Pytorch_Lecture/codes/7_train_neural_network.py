'''
The gradient is the slope of the loss function.

Forward pass: calculate the loss with given weights.
Backward pass: calculate the gradient (update the weights) with respect to loss.


nn.CrossEntropyLoss --> pass the raw output into the loss, not output of the softmax function.
nn.LogSoftmax() --> nn.NLLLoss() --> negative log likelihood loss.

'''

import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Get Data
dataiter = iter(trainloader)
images, labels = dataiter.next()
# Flatten Images to 64,784
images = images.view(images.shape[0], -1)

# Define model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))


# Define loss
criterion = nn.CrossEntropyLoss()

# Forward Pass
logits = model(images)  # we haven't define softmax yet.

# Calculate the loss with logits and labels.
loss = criterion(logits, labels)

print('loss with cross entropy: ', loss)


'''
Use LogSoftmax at the end. 

If you use the probabilities to calculate the loss function then you should use NLLLoss. 

While calculating the softmax, remember the architecture of matrices:

    There are 64 rows (observations) and 10 columns.
    Each column represent the output for a number. 
    For observation 1 (row), there are 10 outputs. 
    We would like to calculate the  probability of each number
     by dividing exp of one output to sum of exp of each output.

dim=0 takes the sum of the columns. 
dim=1 takes the sum of the row. 

We didn't use dim argument in nn.Softmax or F.Softmax.
'''

# TODO: difference btw nn.Softmax and nn.LogSoftmax.

# Define model
model_2 = nn.Sequential(nn.Linear(784, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        nn.LogSoftmax(dim=1))  # dim=1 --> calculates softmax across the columns so each row sums to 1.

# Define loss
criterion_2 = nn.NLLLoss()

# Forward Pass
logits_2 = model_2(images)

# Calculate the loss with logits and labels.
loss_2 = criterion_2(logits_2, labels)

print('loss with nlllos: ', loss_2)



