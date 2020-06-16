import torch


def activation(x):
    return 1/(1+torch.exp(-x))


torch.manual_seed(7)

features = torch.randn((1, 3))

# Define the size of (number of neurons) each layer:
n_input = features.shape[1] # same shape with features
n_hidden = 2
n_output = 1

w1 = torch.randn(n_input, n_hidden)
w2 = torch.randn(n_hidden, n_output)

b1 = torch.randn(1, n_hidden)
b2 = torch.randn(1, n_output)

h = activation(torch.mm(features, w1) + b1)
y = activation(torch.mm(h, w2) + b2)

y_one = activation(torch.mm(activation(torch.mm(features, w1) + b1), w2) + b2)

print('done')