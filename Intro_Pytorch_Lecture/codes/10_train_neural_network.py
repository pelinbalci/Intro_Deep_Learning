import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Get Data
images, labels = next(iter(trainloader))

# Flatten Images to 64,784
images = images.view(images.shape[0], -1)
# images.resize_(64,784)

# Define model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))  # dim=1 --> calculates softmax across the columns so each row sums to 1.

# Define loss
criterion = nn.NLLLoss()

'''
Training the network. 
How can we update the weights with gradients? We need to define this function. 
Pytorch has optim package. 
Stochastic gradient descent --> optim.SGD
'''

# Define Optimizer
optimizer = optim.SGD(model.parameters(), lr=100)


print('initial weights:', model[0].weight)

# When you do multiple backwards with the same parameters the gradients are accumulated.
# You need to zero the gradients on each training pass.
# Otherwise, you'll retain gradients from previous training batches.

# Clear the gradients.
optimizer.zero_grad()

# Forward Pass, calculate loss, backward pass on loss, update weights with optimizer step.
logits = model.forward(images)
loss = criterion(logits, labels)
loss.backward()

print('gradient: ', model[0].weight.grad)

optimizer.step()
print('updated weights: ', model[0].weight)


# Initial weight = -0.0319
# learning rate = 100
# gradient = -0.0017
# Updated weight:
print(-0.0319 - (100 * -0.0017))  # 0.1381 -->  0.1430
print(0.0231 - (100 * -0.0017))  # 0.1931 -->  0.1980







