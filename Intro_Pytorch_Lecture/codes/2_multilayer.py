import torch


def activation(x):
    return 1/(1+torch.exp(-x))


torch.manual_seed(7)

features = torch.randn((1, 3))  # tensor([[-0.1468,  0.7861,  0.9468]]) shape: 1,3

# Define the size of (number of neurons) each layer:
n_input = features.shape[1]  # same shape with features
n_hidden = 2
n_output = 1

w1 = torch.randn(n_input, n_hidden)  # shape: 3 rows, 2 columns. First column is for first neuron, second col is for second neuron.
w2 = torch.randn(n_hidden, n_output) # shape: 2 rows 1 column. 2 hidden nueron to one output.

b1 = torch.randn(1, n_hidden) # shape 1,2:
b2 = torch.randn(1, n_output)

h = activation(torch.mm(features, w1) + b1)
y = activation(torch.mm(h, w2) + b2)

y_one = activation(torch.mm(activation(torch.mm(features, w1) + b1), w2) + b2)
print('done')