# Ref: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
import numpy as np


class RNN:
    # define RNN
    def step(self, x):
        # update the hidden state

        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y


rnn = RNN()
y = rnn.step(x)