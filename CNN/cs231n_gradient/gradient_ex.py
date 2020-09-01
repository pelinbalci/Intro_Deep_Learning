import numpy as np

np.random.seed(0)

# forward pass
W = np.random.randn(3, 2)
X = np.random.randn(2, 1)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)

print(W)

print(X)

print(dD)