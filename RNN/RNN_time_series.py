import os
import pandas as pd
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from common.save_fig import save_fig


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden


def train(rnn, iterations, print_every, x_tensor, y_tensor, time_steps, criterion, optimizer):

    # initialize hidden state
    hidden = None

    for batch_i, iteration in enumerate(range(iterations)):

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i % print_every == 0:
            print('batch_i: {}, Loss: {}'.format(batch_i,loss.item()))

    return rnn, prediction


##############
# decide on hyperparameters for construct RNN
group = 'group_1'

input_size = 1
output_size = 1
hidden_dim = 64
n_layers = 2
lr = 0.01
divide = 100000

# train the rnn and monitor results:

iterations = 1000
print_every = 100

##############
# instantiate an RNN with hyperparameters
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

##############
# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)


##############
# prepare data and loop over groups

df = pd.read_csv('/Users/pelin.balci/PycharmProjects/Intro_Deep_Learning/RNN/time_series_df.csv')
time_steps = range(19)
groups = list(df['group'].unique())


#for group in groups:

df_group = df[df['group'] == group]
df_crop = df_group[:20]
df_ar = df_crop['impression']
df_ar = df_ar.reset_index()
df_final = df_ar['impression']


df_final = df_final/divide

x = df_final[:-1]  # all but the last piece of data
y = df_final[1:]

x = x.to_numpy()
y = y.to_numpy()

x_reshape = x.reshape(-1,1)
y_reshape = y.reshape(-1,1)

# convert data into Tensors
x_tensor = torch.Tensor(x_reshape).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
y_tensor = torch.Tensor(y_reshape)

trained_rnn, prediction = train(rnn, iterations, print_every, x_tensor, y_tensor, time_steps, criterion, optimizer)

plt.figure(figsize=(12, 8))
plt.title(str(group) + ' divided by' + str(divide) + ' lr: ' + str(lr) + ' hidden_dim: ' + str(hidden_dim) +
          ' n_layers:' + str(n_layers))
plt.plot(time_steps, x_reshape, 'r.', label='input')  # input
plt.plot(time_steps, x_reshape, 'r-')
plt.plot(time_steps, prediction.data.numpy().flatten(), 'b.', label='prediction')
plt.plot(time_steps, prediction.data.numpy().flatten(), 'b-')  # predictions
plt.plot(time_steps, y_reshape, 'g.', label='real')
plt.plot(time_steps, y_reshape, 'g-')
plt.legend()
fig = plt.gcf()
name = 'pred_' + group
save_fig('RNN/images', fig, name)

print('rnn is completed')