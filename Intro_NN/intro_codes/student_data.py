# Ref: http://www.ashukumar27.io/LogisticRegression-Backpropagation/
# REf: https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function


# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('/Users/pelin.balci/PycharmProjects/aws_machine_learning/inputs/student_data.csv')

# Printing out the first 10 rows of our data
print(data[:10])

# Importing matplotlib
from matplotlib import pyplot as plt


# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')


# Plotting the points
plot_points(data)
plt.show()
# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()


# TODO:  Make dummy variables for rank
one_hot_data = pd.get_dummies(data['rank'])
one_hot_data = pd.concat([data, one_hot_data], axis=1)

# TODO: Drop the previous rank column
one_hot_data = one_hot_data.drop(['rank'], axis=1)

# Print the first 10 rows of our data
one_hot_data[:10]


# Making a copy of our data
processed_data = one_hot_data[:]

# TODO: Scale the columns
processed_data.gpa = processed_data.gpa/4

processed_data.gre = processed_data.gre / 800

# Printing the first 10 rows of our procesed data
processed_data[:10]

# split the data into a Training and a Testing set. The size of the testing set will be 10% of the total data.
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])

# split the data into features (X) and targets (y).
features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(features[:10])
print(targets[:10])


# Training the 2-layer Neural Network
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)


# Write the error term. Remember that this is given by the equation (y- y')sigma'(x)
# It should be (y-y')*x. The product of x is in the function. Therefore y-y' is enough. Check the references.

# TODO: Write the error term formula
def error_term_formula(x, y, output):
    return (y - output)  # since we increase the parameters with this, we need to write negative of the function as y-y'
    #return (y-output)*sigmoid_prime(x) # --> solution. I think this is wrong.

## Alternative solution ##
# you could also *only* use y and the output
# and calculate sigmoid_prime directly from the activated output!

# below is an equally valid solution (it doesn't utilize x)
# def error_term_formula_other(x, y, output):
#     return (y-output) * output * (1 - output)


# for this case we need to use dE/dy' * dy'/d(Wx+b) = (-y/y' + (1-y)/(1-y')) * y' * (1-y')
# I couldn't find that dE/dy' = y-y'.


# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5


# Training function
def train_nn(features, targets, epochs, learnrate):
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here
            #   rather than storing h as a separate variable
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            #error = error_formula(y, output)
            #error = - y*np.log(output) - (1 - y) * np.log(1-output)

            # The error term
            #error_term = error_term_formula(x, y, output)
            error_term = (y-output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights


weights = train_nn(features, targets, epochs, learnrate)


# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))