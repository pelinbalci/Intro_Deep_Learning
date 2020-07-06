import numpy as np

# https://stackoverflow.com/questions/10593100/how-do-you-do-natural-logs-e-g-ln-with-numpy-in-python
# np.log is ln, whereas np.log10 is your standard base 10 log.


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def calculate_cross_entropy(Y, P):
    cross_entropy = 0
    for i in range(len(Y)):
        cross_entropy += Y[i] * np.log(P[i]) + (1-Y[i]) * np.log(1-P[i])
    return cross_entropy*-1


Y = [1, 0, 1, 1]
P = [0.4, 0.6, 0.1, 0.5]
cross_entropy = calculate_cross_entropy(Y, P)
print('quiz', cross_entropy)

Y = [1, 0, 1, 1]
P = [0.9, 0.1, 0.8, 0.8]
cross_entropy = calculate_cross_entropy(Y, P)
print('good model', cross_entropy)


Y = [1, 0, 1, 1]
P = [0.1, 0.3, 0.4, 0.2]
cross_entropy = calculate_cross_entropy(Y, P)
print('bad model', cross_entropy)


def cross_entrpoy_solution(Y,P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))