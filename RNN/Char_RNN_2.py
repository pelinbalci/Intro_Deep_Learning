# Ref: https://classroom.udacity.com/courses/ud188/lessons/a8fc0724-37ed-40d9-a226-57175b8bb8cc/concepts/04794f8a-7728-4c0a-a808-aa7972bc7738
# Ref: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/char-rnn/Character_Level_RNN_Solution.ipynb


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# open text file and read in data as `text`
with open('anna.txt', 'r') as f:
    complete_text = f.read()


#text = complete_text[:48]
text = complete_text[:100000]
text = complete_text
print(text)

# encode the text and map each character to an integer and vice versa

# we create two dictionaries:
# 1. int2char, which maps integers to characters
# 2. char2int, which maps characters to unique integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])

print('encoded', encoded)
print('char2int', char2int)

def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

# check that the function works as expected
test_seq = np.array([[2, 1, 0]])
one_hot = one_hot_encode(test_seq, 3)

print(one_hot)


def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    # Get the number of batches we can make
    n_batches = len(arr) // (batch_size * seq_length)

    batch_size_total = batch_size * seq_length

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]

    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # Iterate over the batches using a window of size seq_length
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

batch_size = 2
seq_length = 2
batches = get_batches(encoded, batch_size, seq_length)
x, y = next(batches)

# printing out the first 10 items in a sequence
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])


# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')


# RNN Model:

class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ## TODO: define the layers of the model
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## TODO: Get the outputs and the new hidden state from the lstm

        # get RNN outputs
        r_out, hidden = self.lstm(x)

        r_out = self.dropout(r_out)

        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.contiguous().view(-1, self.n_hidden)

        # get final output
        out = self.fc(r_out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))  # 43
    data, val_data = data[:val_idx], data[val_idx:]

    print('train_data', data)
    print('validation data', val_data)


    if (train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):

            #print('x', x)
            #print('y', y)

            counter += 1

            # One-hot encode our data and make them Torch tensors
            x_one_hot = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x_one_hot), torch.from_numpy(y)

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # clip the gradient with a treshold.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x_one_hot = one_hot_encode(x, n_chars)
                    x_one_hot, y = torch.from_numpy(x_one_hot), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x_one_hot, y
                    if (train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length).long())

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterationg through validation data

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


# define and print the net
n_hidden = 512
n_layers = 2

net = CharRNN(chars, n_hidden, n_layers)
print(net)

batch_size = 128
seq_length = 100
n_epochs = 4 # start small if you are just testing initial behavior

# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.01, print_every=1)



# change the name, for saving multiple files
model_name = 'rnn_new_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)


############
# Prediction
# To actually get the next character, we apply a softmax function,
# which gives us a probability distribution that we can then sample to predict the next character.


def predict(net, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if (train_on_gpu):
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    if (train_on_gpu):
        p = p.cpu()  # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):
    if (train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)   # 1 means that 1 char at a time.
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(net, 100, prime='Anna', top_k=5))


##### Other way of loading the model

# Here we have loaded in a model that trained over 20 epochs `rnn_20_epoch.net`
with open('rnn_new_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])


# Sample using a loaded model
print(sample(loaded, 100, top_k=5, prime="the"))
