# Ref:https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/fc_model.py


import torch
from torch import nn
import torch.nn.functional as F


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

# model = nn_model.Network(784, 10, [512, 256, 128])

# class Classifier(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.hidden_1 = nn.Linear(784, 256)
#         self.hidden_2 = nn.Linear(256, 128)
#         self.hidden_3 = nn.Linear(128, 64)
#         self.hidden_4 = nn.Linear(64, 10)
#
#         self.dropout = nn.Dropout(p=0.2)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], 784)
#
#         x = self.dropout(F.relu(self.hidden_1(x)))
#         x = self.dropout(F.relu(self.hidden_2(x)))
#         x = self.dropout(F.relu(self.hidden_3(x)))
#
#         # no dropout for the final layer. It should have 10 neurons.
#         x = F.log_softmax(self.hidden_4(x), dim=1)
#
#         return x


def validation(model, criterion, testloader):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 784)

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


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5, total_steps=40):
    steps = 0
    train_loss = 0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            steps += 1
            images.resize_(images.size()[0], 784)

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