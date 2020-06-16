import numpy as np
from matplotlib import pyplot as plt

X = 2 * np.random.rand(10, 1)
y = 3*X + 4 + np.random.rand(10,1)

plt.plot(X, y, "bo")
plt.show()

onesX = np.ones(10).reshape(-1,1)
X_b = np.concatenate((onesX, X), axis=1)

# first way: manual

# y = theta * x
# y

theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('theta:', theta)

# second way: sklearn
from sklearn.linear_model import LinearRegression
reg_clf = LinearRegression()
reg_clf.fit(X, y)

print("sklearn Coef:",reg_clf.coef_)
print("sklearn Intercept:", reg_clf.intercept_)

# batch gradient descent
theta_batch = np.random.rand(2, 1)
iterations = 1000
eta = 0.1
m = len(X_b)
for i in range(iterations):

    # error = 1/m * sum [(y - y_pred)^2]
    #       = 1/m * sum [(y -  theta*x +b) ^ 2]

    # gradient is partial derivative for theta.
    # d error / d theta = 2/m * sum [(-x) * (y - (theta*x +b)]

    gradient = (2 / m) * X_b.T.dot(X_b.dot(theta_batch) - y)
    theta_batch = theta_batch - eta * gradient

print('theta_batch_gradient:', theta_batch)


# stochastic gradient descent
theta = np.random.rand(2, 1)
m = len(X_b)

eta = 0.1
epoch_num = 200
for epoch in range(epoch_num):
    for i in range(m):
        idx = np.random.randint(0, m)
        x_b_stochastic = X_b[idx:idx + 1]
        y_stochasitc = y[idx:idx + 1]
        gradient = (2 / m) * x_b_stochastic.T.dot(x_b_stochastic.dot(theta) - y_stochasitc)
        theta = theta - eta * gradient

print('theta_stochastic_gradient:', theta)