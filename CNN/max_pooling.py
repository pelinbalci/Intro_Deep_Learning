# https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/maxpooling_visualization.ipynb

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.save_fig import save_fig

###########
# prepare data #
img_path = 'images/udacity_sdc.png'

# load color image
original_img = cv2.imread(img_path)  # shape: 213, 320, 3
# convert to grayscale
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)  # shape 213, 320

# normalize, rescale entries to lie in [0,1]
gray_img_normalized = gray_img.astype("float32")/255

# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img_normalized).unsqueeze(0).unsqueeze(1)  # shape 1,1,213,320

###########
# define filters #
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])  # shape: 4,4,4

###########
# define weights #
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)  # shape: 4, 1, 4, 4

############
# define nn with a single conv layer with four filters #
class Net(nn.Module):

    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # assumes there are 4 grayscale filters
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

        # define pool layer
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)  # A ReLu function turns all negative pixel values in 0's (black).

        pooled_x = self.pool(activated_x)

        # returns both layers
        return conv_x, activated_x, pooled_x


############
# instantiate the model and set the weights 3
model = Net(weight)

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)



############
# Outputs
print('model: \n', model)
#  Net((conv): Conv2d(1, 4, kernel_size=(4, 4), stride=(1, 1), bias=False)
#      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

# helper function for visualizing the output of a given layer, default number of filters is 4
def viz_layer(layer, name, n_filters=4):
    fig = plt.figure(figsize=(20, 20))

    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i + 1)
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))

    fig = plt.gcf()
    save_fig('CNN/images', fig, name)


# plot original image
plt.imshow(gray_img, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'max_pool_car_gray')

# visualize all filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')

fig = plt.gcf()
save_fig('CNN/images', fig, 'max_pool_car_filters')


# visualize the output
viz_layer(conv_layer, 'max_pool_car_conv')

viz_layer(activated_layer, 'max_pool_car_activated')

viz_layer(pooled_layer, 'max_pool_car_pooled')



