import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as F
from common import helper
import time

start = time.time()

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# Define model:
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax(dim=1)
                      )

# Define Loss:
criterion = nn.NLLLoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1
for e in range(epochs):
    epoch_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)

        # Forward pass, find the output
        logits = model.forward(images)

        # Calculate loss with the output and actual values
        loss = criterion(logits, labels)

        # Clear grad and update weights:
        optimizer.zero_grad()  # clear the gradient
        loss.backward()  # calculate grad
        optimizer.step()  # update weights

        epoch_loss += loss.item()
    else:
        print('epoch loss is', epoch_loss)

print('Duration: {} seconds'.format(time.time() - start))

img = images[0].view(1, 784)
with torch.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')
