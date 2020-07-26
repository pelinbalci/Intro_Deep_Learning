# Ref: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/time-series/Simple_RNN.ipynb

"""
In ths notebook, we're going to train a simple RNN to do time-series prediction. Given some set of input data,
it should be able to generate a prediction for the next time step

First, we'll create our data
Then, define an RNN in PyTorch
Finally, we'll train our network and see how it performs
"""


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from common.save_fig import save_fig

plt.figure(figsize=(8,5))

# how many time steps/data pts are in one batch of data
seq_length = 20

# generate evenly spaced data pts
time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data.resize((seq_length + 1, 1))  # size becomes (seq_length+1, 1), adds an input_size dimension

x = data[:-1]  # all but the last piece of data
y = data[1:]  # all but the first

# display the data
plt.plot(time_steps[1:], x, 'r.', label='input, x') # x
plt.plot(time_steps[1:], y, 'b.', label='target, y') # y

plt.legend(loc='best')
fig = plt.gcf()
name = 'data'
save_fig('RNN/images', fig, name)


"""
Define RNN
- input_size - the size of the input we have 1 feature --> x
- hidden_dim - the number of features in the RNN output and in the hidden state
- n_layers - the number of layers that make up the RNN, typically 1-3; greater than 1 means that you'll create a stacked RNN
- batch_first - whether or not the input/output of the RNN will have the batch_size as the first dimension 
(batch_size, seq_length, hidden_dim)
- batch_first=True means that the first dimension of the input and output will be the batchsize. 
- batch_first=True : shaping the input such that the batch size will be the first dimension. 
"""


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # final fc layer.
        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, x, hidden):
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)

        # shape output to be batch_size * seq_length, hidden_dim
        r_out = r_out.view(-1, self.hidden_dim)

        output = self.fc(r_out)

        return output, hidden


# check the shape
# test that dimensions are as expected
test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)

# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))  # (20,1)

test_input = torch.Tensor(data).unsqueeze(0)  # give it a batch_size of 1 as first dimension
print('Input size: ', test_input.size())  # torch.Size([1, 20, 1])  batch_size=1, seq_length = 20, input number of features = 1

# test out rnn sizes
test_out, test_h = test_rnn(test_input, None)
print('Output size: ', test_out.size())  # torch.Size([20, 1])  batch_size*seq=20, output_size=1
print('Hidden state size: ', test_h.size())  # torch.Size([2, 1, 10])  n_layers=2, batch_size =1, hidden_dim=10

print('done')


# check the shape
# test that dimensions are as expected
test_rnn = RNN(input_size=2, output_size=2, hidden_dim=7, n_layers=3)

# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, seq_length)
data_1 = np.sin(time_steps)
data_2 = np.cos(time_steps)
data_1.resize((seq_length, 1))
data_2.resize((seq_length, 1))
data = np.concatenate((data_1, data_2), axis=1)

test_input = torch.Tensor(data).unsqueeze(0)  # give it a batch_size of 1 as first dimension
print('Input size: ', test_input.size())  # torch.Size([1, 20, 2])  batch_size=1, seq_length = 20, input number of features = 2

# test out rnn sizes
test_out, test_h = test_rnn(test_input, None)
print('Output size: ', test_out.size())  # torch.Size([20, 2])  batch_size*seq=20, output_size=2
print('Hidden state size: ', test_h.size())  # torch.Size([3, 1, 7])  n_layers=3, batch_size =1, hidden_dim=7

print('done')