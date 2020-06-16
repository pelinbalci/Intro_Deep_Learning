import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as F
from Intro_Pytorch.common import helper
import time

start = time.time()

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training data:
trainset = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128,64)
        self.hidden3 = nn.Linear(64,10)

    def forward(self,x):
        #images = images.view(images.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.log_softmax(self.hidden3(x))

        return x

# Define model:
model = Classifier()

# Define Loss:
criterion = nn.NLLLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5
for e in range(epochs):
    epoch_loss = 0
    for images, labels in trainloader:

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

# test
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]

ps = torch.exp(model(img))
helper.view_classify(img, ps, version='Fashion')
