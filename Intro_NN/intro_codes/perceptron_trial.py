import numpy as np

# Setting the random seed, feel free to change it and see different solutions.

# Example of AND operator, as described above
lr = 0.5
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [0, 0, 0, 1]
weights = [0, 0]
bias = 0
num_epochs = 5


for i in range(num_epochs):
    # In each epoch, we apply the perceptron step.
    for i in range(len(data)):
        input = data[i]
        y_observed = target[i]

        for j in input:
            X = input[j]
            W = weights[j]
            lin_output = X*W + bias

            y_pred = int(lin_output > 0)

        if (y_pred == 0) & (target[i] == 1):
            W[0] += lr * X[i][0]
            W[1] += lr * X[i][1]
            bias += lr

        if (y_pred == 1) & (target[i] == 0):
            W[0] -= lr * X[i][0]
            W[1] -= lr * X[i][1]
            bias -= lr
