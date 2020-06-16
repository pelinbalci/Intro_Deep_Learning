from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 256)  # Linear Transformation.
        self.output = nn.Linear(256,10)

        self.sigmoid = nn.Sigmoid()  # After linear transformation we apply sigmoid.
        self.softmax = nn.Softmax()  # After the output we apply softmax.

    def forward(self, x):
        # Pass the input x through each of the operations:
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


model = Network()
print('done')


import torch.nn.functional as F


class Network_alternative(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 256)  # Linear Transformation.
        self.output = nn.Linear(256,10)

    def forward(self, x):
        # Pass hidden and output layer into sigmoid and softmax functions.
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x))

        return x


model_alternative = Network_alternative()
print('done')

