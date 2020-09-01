# Ref: https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment1/cs231n/classifiers/softmax.py


from builtins import range
import numpy as np
from random import shuffle
from builtins import range

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        # calculate loss
        scores = X[i].dot(W)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        loss -= np.log(probs[y[i]])

        # calculate gradient
        # we are calculating: prob (y_pred) - y_actual.
        # y actual is 1 , that's why we subtract 1.
        dscores = probs.reshape(1, -1)
        dscores[:, y[i]] -= 1

        # since scores = X.dot(W), get dW by multiplying X.T and dscores
        # W is D x C so dW should also match those dimensions
        # X.T x dscores = (D x 1) x (1 x C) = D x C
        dW += np.dot(X[i].T.reshape(X[i].shape[0], 1), dscores)

    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[range(num_train), y])
    data_loss = np.sum(correct_logprobs) / num_train
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss

    dscores = probs
    dscores[range(num_train), y] -= 1
    dscores /= num_train
    dW = np.dot(X.T, dscores)
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
