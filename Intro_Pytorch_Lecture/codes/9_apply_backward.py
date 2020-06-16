import torch
from torch import nn
from torchvision import datasets, transforms

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Get Data
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
images, labels = next(iter(trainloader))

# Flatten Images to 64,784
images = images.view(images.shape[0], -1)

# Define model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))  # dim=1 --> calculates softmax across the columns so each row sums to 1.

# Define loss
criterion = nn.NLLLoss()

# Forward Pass, logits are output of the model.
logits = model(images)

# Calculate the loss with logits and labels.
loss = criterion(logits, labels)

print('loss with nlllos: ', loss)

print('before backward pass: ', model[0].weight.grad)  # gives none
loss.backward()
print('after backward pass:', model[0].weight.grad)  # gives gradients.


