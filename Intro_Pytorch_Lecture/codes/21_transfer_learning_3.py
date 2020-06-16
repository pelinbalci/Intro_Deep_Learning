# Solution : https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F

from Intro_Pytorch.common import helper, nn_model


data_dir = 'Cat_Dog_data'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=100)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(2048, 512),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(512, 2),
                      nn.LogSoftmax(dim=1))

model.fc = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images_test, labels_test in testloader:

                    images_test, labels_test = images_test.to(device), labels_test.to(device)

                    logps_test = model(images_test)
                    batch_loss = criterion(logps_test, labels_test)
                    test_loss += batch_loss

                    ps = torch.exp(logps_test)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))
                    accuracy = accuracy.item()

            print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

            running_loss = 0
            model.train()


'''
Epoch: 1/1..  Training Loss: 1.891..  Test Loss: 1.086..  Test Accuracy: 0.515
Epoch: 1/1..  Training Loss: 0.755..  Test Loss: 0.211..  Test Accuracy: 0.493
Epoch: 1/1..  Training Loss: 0.448..  Test Loss: 0.140..  Test Accuracy: 0.499
Epoch: 1/1..  Training Loss: 0.230..  Test Loss: 0.113..  Test Accuracy: 0.492
Epoch: 1/1..  Training Loss: 0.217..  Test Loss: 0.070..  Test Accuracy: 0.502
Epoch: 1/1..  Training Loss: 0.228..  Test Loss: 0.070..  Test Accuracy: 0.502
Epoch: 1/1..  Training Loss: 0.179..  Test Loss: 0.067..  Test Accuracy: 0.504
Epoch: 1/1..  Training Loss: 0.144..  Test Loss: 0.067..  Test Accuracy: 0.497
Epoch: 1/1..  Training Loss: 0.000..  Test Loss: 2.040..  Test Accuracy: 0.416
Epoch: 1/1..  Training Loss: 0.000..  Test Loss: 2.147..  Test Accuracy: 0.427


Epoch 1/1.. Train loss: 0.577.. Test loss: 0.241.. Test accuracy: 0.898
Epoch 1/1.. Train loss: 0.316.. Test loss: 0.104.. Test accuracy: 0.963
There is a mistake here. test accuracy is not changing over time. 

'''
