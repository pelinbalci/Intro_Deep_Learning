# Ref: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%207%20-%20Loading%20Image%20Data%20(Exercises).ipynb

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
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


# # Run this to test your data loader
# images, labels = next(iter(train_loader))
# helper.imshow(images[0], normalize=False)
# plt.show()


# images_shape : [32,3,224,224] --> 32 images per batch, 3 color channel, 224x224 images.

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def validation(model, criterion, testloader):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 50176)

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


#def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5, total_steps=40):



model = Network(50176, 2, [500, 200, 100])
print('Original model_3', model)

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
        images = images.resize_(images.size()[0], 50176)

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

            model.train()  # back to training mode


# # Save the model: The parameters of network are stored in a model's state_dict.
# torch.save(model.state_dict(), 'cat_dog_checkpoint.pth')

# # Load and reuse it:
# my_state_dict = torch.load('cat_dog_checkpoint.pth')
# model_reuse = nn_model.Network(784, 10, [512, 256, 128])
# model_reuse.load_state_dict(my_state_dict)
# print('Reuse model:', model_reuse)

# test
dataiter = iter(test_loader)
images, labels = dataiter.next()
img = images[1]

ps = torch.exp(model(img))
helper.view_cat_dog(img, ps, version='catdog')
plt.show()




