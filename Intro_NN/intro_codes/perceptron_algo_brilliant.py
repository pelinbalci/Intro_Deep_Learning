# Ref: https://brilliant.org/wiki/perceptron/#:~:text=The%20perceptron%20is%20a%20machine,equal%20to%200%20or%201%3F

# Example of AND operator, as described above
lr = 0.5
data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
weights = [0, 0]
bias = 0

iter = 5

for i in range(iter):
    # Start with the weights from t-1
    weights = [i for i in weights]
    bias = bias

    # For each input data point
    for values in data:
        # Add bias (intercept) to line
        comparison = bias
        input_data = values[0]

        # For each variable, compute the value of the line
        for index in range(len(input_data)):
            comparison += weights[index] * input_data[index]

        # Obtain the correct classification and the classification of the algorithm
        observed_value = values[1]
        predicted_value = int(comparison > 0)

        # If the values are different, add an error to the weights and the bias
        if predicted_value != observed_value:
            for index in range(len(input_data)):
                weights[index] += lr * (observed_value - predicted_value) * input_data[index]
            bias += lr * (observed_value - predicted_value)