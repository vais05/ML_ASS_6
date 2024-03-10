import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights
W = np.array([10, 0.2, -0.75])

# Input data for AND gate
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Desired output for AND gate
y = np.array([0, 0, 0, 1])

# Varying learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Maximum number of epochs
max_epochs = 1000

# Convergence error threshold
convergence_error = 0.002

# Function to calculate step activation
def step_activation(x):
    return 1 if x >= 0 else 0

# Function to train perceptron
def train_perceptron(X, y, W, alpha, max_epochs, convergence_error):
    for epoch in range(max_epochs):
        error_sum = 0

        for i in range(len(X)):
            # Calculate the predicted output
            prediction = step_activation(np.dot(X[i], W))

            # Calculate the error
            error = y[i] - prediction

            # Update weights
            W = W + alpha * error * X[i]

            # Accumulate the squared error for this sample
            error_sum += error ** 2

        # Calculate the sum-squared error for all samples in this epoch
        total_error = 0.5 * error_sum

        # Check for convergence
        if total_error <= convergence_error:
            return epoch + 1  # Return the number of iterations to converge

    return max_epochs  # Return max_epochs if convergence is not reached

# List to store the number of iterations for each learning rate
iterations_list = []

# Train the perceptron for each learning rate
for alpha in learning_rates:
    iterations = train_perceptron(X, y, W, alpha, max_epochs, convergence_error)
    iterations_list.append(iterations)

# Plotting the number of iterations against learning rates
plt.plot(learning_rates, iterations_list, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations to Converge')
plt.title('Number of Iterations vs Learning Rate')
plt.show()
