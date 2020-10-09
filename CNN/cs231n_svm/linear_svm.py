from builtins import range
import numpy as np
from random import shuffle
from builtins import range


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    ############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]  #10
    num_train = X.shape[0]  # 500
    loss = 0.0
    delta = 1.0
    scores = X.dot(W)

    # convert correct_class_scores to a column vector
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    margin = np.maximum(0, scores - correct_class_scores + delta)  # shape: 500,10
    # you need to sum the values except the correct class values to calculate the loss.
    margin[np.arange(num_train), y] = 0  # correct y values are 0 now.

    loss = np.sum(margin)
    loss /= num_train
    loss += reg * np.sum(W*W)  # 10.1549
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM loss, storing the result in dW #                                                                         #
    # Hint: Instead of computing the gradient from scratch, it may be easier to reuse some of the intermediate values that you used to compute the loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    binary = margin
    binary[margin > 0] = 1  # all values bigger than 0 are 1
    row_sum = np.sum(binary, axis=1)
    # Subtract in correct class (-s_y)
    binary[np.arange(num_train), y] -= row_sum  # binary y. column'dan row sum'ı çıkar. correct column'dan kaç kex x çıkarılacağını buluyoruz.
    dW = np.dot(X.T, binary) / num_train
    # Regularization gradient
    dW += reg * 2 * W

    # Other Version
    scores_new = X.dot(W)
    N = X.shape[0]
    correct_class_scores_new = scores_new[np.arange(N), y]
    margins = np.maximum(0, scores_new - correct_class_scores_new[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0  # correct y values are 0 now.
    loss = np.sum(margins) / N   # 9.15
    num_pos = np.sum(margins > 0, axis=1)
    dW_new = np.zeros_like(scores_new)
    dW_new[margins > 0] = 1  # instead of correct classes all values are 1 now.
    dW_new[np.arange(N), y] -= num_pos
    dW_new /= N
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
