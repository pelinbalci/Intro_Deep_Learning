import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F


# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training & test data:
trainset = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# model architecture
class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(784, 256)
        self.hidden_2 = nn.Linear(256, 128)
        self.hidden_3 = nn.Linear(128, 64)
        self.hidden_4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], 784)

        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.log_softmax(self.hidden_4(x), dim=1)

        return x


# 1.define model
# 2. define loss
# 3. define optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# data iteraions
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
images, labels = next(iter(trainloader))
features = images.view(images.shape[0], 784)

# 4. Forward pass, find the output (model.forward(images))
# 5. Calculate loss with the output and actual values (criterion(logits, labels))
# 6. Clear gradients (optimizer.zero_grad())
# 7. Calculate grad (loss.backward())
# 8. Update weights (optimizer.step())
logits = model.forward(images)
loss = criterion(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

