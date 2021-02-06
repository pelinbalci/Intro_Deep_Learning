# Ref: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/sentiment-rnn/Sentiment_RNN_Exercise.ipynb

"""
We need an embedding layer because we have tens of thousands of words, so we'll need a more efficient representation
for our input data than one-hot encoded vectors.

In this case, the embedding layer is for dimensionality reduction, rather than for learning semantic representations.

After input words are passed to an embedding layer, the new embeddings will be passed to LSTM cells.

Finally, the LSTM outputs will go to a sigmoid output layer. We're using a sigmoid function because positive and
negative = 1 and 0, respectively, and a sigmoid will output predicted, sentiment values between 0-1.
"""

import numpy as np

#########
# read data
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()

with open('data/labels.txt', 'r') as f:
    labels = f.read()

print('REVIEWS:', reviews[:2000])
print('LABELS:', labels[:20])

##########
# get rid of punctuation
from string import punctuation

print(punctuation)  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])

print('W/O PUNC: ', all_text[:2000])

reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# split by new line
words = all_text.split()  # split the words to different elements in one list.

########
# Encoding the words

# feel free to use this import
from collections import Counter

# Build a dictionary that maps words to integers
word_count = dict(Counter(words))
sorted_word = sorted(word_count.keys(), key=word_count.get, reverse=True)
vocab_to_int = {word: i + 1 for i, word in enumerate(sorted_word)}

# use the dict to tokenize each review in reviews_split store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
print()

# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])

########
# Encoding the labels
labels_string = labels.split('\n')
encoded_labels = [1 if x == 'positive' else 0 for x in labels_string]

review_lengths = Counter([len(x) for x in reviews_ints])

# TODO: make a plot.

print('zero length: {}'.format(review_lengths[0]))
print('max length: {}'.format(max(review_lengths)))

print('length of rev. before removing outliers: ', len(reviews_ints))


non_zero = []
for idx in range(len(reviews_ints)):
    review = reviews_ints[idx]
    if len(review) != 0:
        non_zero.append(idx)

reviews_ints = [reviews_ints[ii] for ii in non_zero]  # type is list. in pad function it will be turned into numpy array.
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero])


print('length of rev. after removing outliers: ', len(reviews_ints))

# Padding
def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''
    ## implement function

    features = []

    for idx in range(len(reviews_ints)):
        review = reviews_ints[idx]
        if len(review) > seq_length:
            features.append(review[:seq_length])
        else:
            review.extend([0]*(seq_length-len(review)))
            features.append(review)

    features = np.array(features)
    return features


def pad_features_solution(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


# Test your implementation!

seq_length = 200

features = pad_features(reviews_ints, seq_length=seq_length)

features_2 = pad_features_solution(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features) == len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0]) == seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches
print(features[:30, :10])
print(features_2[:30, :10])
print('done')

# Training test sets:

split_frac = 0.8

split_idx = int(len(features)*split_frac)

train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]


# now, half of the remaining part will be test set:
test_idx = int(len(remaining_x)*0.5)
test_x, val_x = remaining_x[:test_idx], remaining_x[test_idx:]
test_y, val_y = remaining_y[:test_idx], remaining_y[test_idx:]

# print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# Turn numpy to tensor, that way you can easily feed the training loop with batches:

import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets:
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# Take a sample:
dataiter = iter(train_loader)

# dataiter.next() gives us a list. 2 elements: x values, y values.
# dataiter.next()[0].shape = torch.Size([50,200])
# dataiter.next()[1].shape = torch.Size([50])

sample_x, sample_y = dataiter.next()
print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

# MODEL

# check if gpu is available
train_on_gpu = torch.cuda.is_available()

if(train_on_gpu):
    print('Yeeeyy GPU')
else:
    print('no gpu available, training on cpu')

# define the model
import torch.nn as nn

class Sentiment_RNN(nn.module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob = 0.5):

        super(Sentiment_RNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        # get RNN outputs

        batch_size = x.size(0)

        embedding_out = self.embedding(x)
        lstm_out, hidden = self.lstm(embedding_out, hidden)

        # shape output to be (batch_size*seq_length, hidden_dim)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # get final output
        drop_out = self.dropout(lstm_out)
        linear_out = self.fc(drop_out)
        sig_out = self.sig(linear_out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
