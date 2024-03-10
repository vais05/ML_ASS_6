import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights (modify for XOR)
W = np.array([10, 0.2, -0.75])

# Input data for XOR gate
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Desired output for XOR gate
y = np.array([0, 1, 1, 0])

# Learning rate
alpha = 0.2  # Experiment with different values

# Maximum number of epochs
max_epochs = 1000

# Error threshold for convergence
convergence_error = 0.002

# Function to calculate step activation
def step_activation(x):
    return 1 if x >= 0 else 0

# Function to train perceptron
def train_perceptron(X, y, W, alpha, max_epochs, convergence_error):
    error_values = []

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

        # Append error to the list for plotting
        error_values.append(total_error)

        # Check for convergence
        if total_error <= convergence_error:
            print(f"Converged in {epoch + 1} epochs.")
            break

    return W, error_values

# Train the perceptron
final_weights, errors = train_perceptron(X, y, W, alpha, max_epochs, convergence_error)

# Plotting epochs against error values
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Error Convergence (XOR Gate)')  # Update title
plt.show()

# Display the final weights
print("Final weights for XOR gate:", final_weights)
