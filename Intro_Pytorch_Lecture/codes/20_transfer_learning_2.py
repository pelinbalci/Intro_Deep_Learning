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


def validation(model, criterion, testloader):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        # Get the output
        logits_test = model(images)

        # Calculate loss
        test_loss += criterion(logits_test, labels)

        # probability of the output
        ps = torch.exp(logits_test)

        # top probabilities and classes
        top_p, top_class = ps.topk(1, dim=1)

        # compare the predictions and labels
        equals = top_class == labels.view(*top_class.shape)

        # calculate accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    return test_loss, accuracy

device = 'cpu'

# train the model:
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

steps = 0
train_loss = 0
epochs = 1
total_steps = 40

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        steps += 1

        optimizer.zero_grad()
        logits = model.forward(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if steps % total_steps == 0:  # in each 40 steps
            # set model to evaluation mode
            model.eval()  # This sets the model to evaluation mode where the dropout probability is 0.

            with torch.no_grad():  # we won't do any backward operation for the validation part.
                test_loss, accuracy = validation(model, criterion, test_loader)

            print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss / total_steps),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
                  "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

            train_loss = 0
            model.train()
