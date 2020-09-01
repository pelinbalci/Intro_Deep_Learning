import numpy as np

stride = 2
padding = 0

np.random.seed(0)

X1 = np.random.randint(0, 5, size=(9, 9))
X2 = np.random.randint(0, 5, size=(9, 9))
X3 = np.random.randint(0, 5, size=(9, 9))

W1 = np.random.randint(0, 3, size=(3, 3))
W2 = np.random.randint(0, 3, size=(3, 3))
W3 = np.random.randint(0, 3, size=(3, 3))

output_shape = ((X1.shape[0] - W1.shape[0] + 2 * padding) / stride) +1
V = np.zeros((int(output_shape),int(output_shape)))
print('Input:', X1, X2, X3)
print('Filter:', W1, W2, W3)

# we are not using dot product. simply product the elements and take sum of them.
V[0, 0] = np.sum(X1[:3, :3] * W1 + X2[:3, :3] * W2 + X3[:3, :3] * W3)
V[1, 0] = np.sum(X1[2:5, :3] * W1 + X2[2:5, :3] * W2 + X3[2:5, :3] * W3)
V[2, 0] = np.sum(X1[4:7, :3] * W1 + X2[4:7, :3] * W2 + X3[4:7, :3] * W3)
V[3, 0] = np.sum(X1[6:9, :3] * W1 + X2[6:9, :3] * W2 + X3[6:9, :3] * W3)
print('Output:', V)

print('done')

