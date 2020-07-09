"""
Original code: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/custom_filters.ipynb

Instructions
- Define your own convolutional filters and apply them to an image of a road
- See if you can define filters that detect horizontal or vertical edges
- This notebook is meant to be a playground where you can try out different filter sizes and weights and see the resulting,
filtered output image!

TODO: Create a custom kernel
Below, you've been given one common type of edge detection filter: a Sobel operator.

The Sobel filter is very commonly used in edge detection and in finding patterns in intensity in an image.
Applying a Sobel filter to an image is a way of taking (an approximation) of the derivative of the image in the x or y
direction, separately. The operators look as follows.


It's up to you to create a Sobel x operator and apply it to the given image.

For a challenge, see if you can put the image through a series of filters: first one that blurs the image
(takes an average of pixels), and then one that detects the edges.

"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

from common.save_fig import save_fig

# Read in the image
image = mpimg.imread('images/curved_lane.jpg')
plt.imshow(image)
plt.show()

# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'curved_lane_gray')


# 3x3 array for edge detection
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

sobel_x_left = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

sobel_x_right = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

blur = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])

emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

outline = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
blur_image = cv2.filter2D(gray, -1, blur)
plt.imshow(blur_image, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'curved_lane_blur')

emboss_image = cv2.filter2D(gray, -1, emboss)
plt.imshow(emboss_image, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'curved_lane_emboss')

outline_image = cv2.filter2D(gray, -1, outline)
plt.imshow(outline_image, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'curved_lane_outline')

sobel_x_left_im = cv2.filter2D(gray, -1, sobel_x_left)
plt.imshow(sobel_x_left_im, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'curved_lane_sobelx_left')

sobel_x_right_im = cv2.filter2D(gray, -1, sobel_x_right)
plt.imshow(sobel_x_right_im, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'curved_lane_sobelx_right')

sobel_y_image = cv2.filter2D(gray, -1, sobel_y)
plt.imshow(sobel_y_image, cmap='gray')
fig = plt.gcf()
save_fig('CNN/images', fig, 'curved_lane_sobely')