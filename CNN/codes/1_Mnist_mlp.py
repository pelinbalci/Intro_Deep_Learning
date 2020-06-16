# Ref: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/mnist-mlp/mnist_mlp_solution.ipynb

import torch
import numpy as np
import time

from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import topk

from common.save_fig import save_fig

start = time.time()

num_workers = 0  # number of subprocrss to use for data loading
batch_size = 20  # how many samples per batch to load


######
# convert data to tensor #
######
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


######
# visualize the data #
######
dataiter = iter(trainloader)  # one batch of the data
images, labels = dataiter.next()  # next() gives us the  data and targets.

images = images.numpy()

# plot the images in the batch with corresponding labels
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]))
    ax.set_title(str(labels[idx].item()))
fig = plt.gcf()
name = 'example_mnist_images'
save_fig(fig, name)


# view one image in more detail
img = np.squeeze(images[1])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.imshow(img)
width, height = img.shape
treshold = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if (img[x][y], 2) != 0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<treshold else 'black')
fig = plt.gcf()
name = 'one_mnist_image'
save_fig(fig, name)


######
# define the network #
######
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten the input
        x = x.view(-1, 28*28)

        # hidden layer with activation  function.
        # The outputs will become positive numbers.
        x = F.relu(self.fc1(x))

        # add dropout layer
        x = self.dropout(x)

        # hidden layer
        x = F.relu(self.fc2(x))

        # add dropout layer
        x = self.dropout(x)

        # add output layer
        x = F.log_softmax(self.fc3(x), dim=1)

        return x  # list of class scores


######
# initialize NN #
######
model = Network()
print(model)


######
# define loss and optimization #
######
# Cross entropy loss is a combination of Logsoftmax + NLLoss.
# It takes the output, calculae the logsoftmax, then apply NLLoss.
# The losses are averaged across observations for each minibatch
# for example if a batch contains 20 images; return loss will be average loss over 20 images.

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model_name = 'Adam'


######
# training loop #
######
# epochs are how many times we want the model to iterate through entire dataset.
# One epoch sees every training image just once.
# start with 20 epochs, try different models, then increase the epoch after find the best model.

n_epochs = 1
epoch_loss_list = []

model.train() # prep model for training

for epoch in range(n_epochs):
    train_loss = 0

    for data, target in trainloader:

        # clear the gradients
        optimizer.zero_grad()

        # forward_pass --> output is predicted class score.
        output = model(data)

        # calculate cross entropy loss (average) for 20 items in one batch.
        loss = criterion(output, target)

        # backward pass --> compute gradient of the loss with respect to model parameters (weights)
        loss.backward()

        # add this loss to train_loss --> will give the total loss in one epoch.
        # we need to multiply the loss with data size.
        # the loss is average of 20 images.
        train_loss += loss.item()*data.size()[0]

        # apply parameter update.
        optimizer.step()

    epoch_loss = train_loss/len(trainloader)
    print('epoch: {} \TrainingLoss: {:.6f}'.format(epoch+1, epoch_loss))

    epoch_loss_list.append(epoch_loss)

print(f"Model = {model_name}; Total time: {(time.time() - start) / 3:.3f} seconds")
print(str(model_name) + ' for ' + str(n_epochs) + ' epochs, ' + ' training loss: ' + str(epoch_loss_list[-1]))


# plot the loss
epochs_list = [i+1 for i in range(n_epochs)]
fig = plt.figure(figsize=(25,4))
plt.plot(epochs_list, epoch_loss_list)
plt.plot(epochs_list, epoch_loss_list, 'bo')
plt.xticks(np.arange(min(epochs_list), max(epochs_list), step=1))
plt.title(str(model_name) + ' for ' + str(n_epochs) + ' epochs ' + ' training loss is ' + str(epoch_loss_list[-1]))

name = str(model_name) + '_' + str(n_epochs) + '_epochs'
fig = plt.gcf()
save_fig(fig, name)


#####
# test the model on test data #
#####

test_loss = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()  # prep model for evaluation

for data, target in testloader:
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size()[0]

    # Prediction Accuracy: convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare pred to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(testloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)')

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


#####
# plot test images and predictions #
#####

# obtain one batch of test images
dataiter = iter(testloader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)

# convert output probabilities to predicted class
_, preds = torch.max(output, 1)

# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))

name = 'one_batch_test_data_predictions'
fig = plt.gcf()
save_fig(fig, name)

# TODO: the epoch number is enough?
# TODO: Should we add one more layer?