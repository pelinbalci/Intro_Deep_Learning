import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as F
from common import helper


# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# Define model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))  # dim=1 --> calculates softmax across the columns so each row sums to 1.

# Define loss
criterion = nn.NLLLoss()


# Define Optimizer, How can we update the weights with gradients? We need to define this function.
optimizer = optim.SGD(model.parameters(), lr=0.003)


epochs = 1
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten Images to 64,784
        images = images.view(images.shape[0], -1)  # 64,784

        # Clear grads
        optimizer.zero_grad()

        # Forward pass
        logits = model.forward(images)

        # Calculate loss
        loss = criterion(logits, labels)
        print('loss', loss)

        # sum the loss
        running_loss += loss.item()

        # Backward --> find gradient
        loss.backward()

        # Update weights
        optimizer.step()
    else:
        print(f'training loss: {running_loss/len(trainloader)}')


# plot
img = images[0].view(1, 784)
with torch.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)




