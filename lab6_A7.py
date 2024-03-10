import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights
weights_input_hidden = np.array([
    [0.5, 0.2, -0.1],
    [-0.3, 0.4, 0.2],
    [0.1, 0.3, -0.2]  # Added a third neuron
])

weights_hidden_output = np.array([-0.1, 0.3, 0.1])

# Input data for AND gate
X = np.array([
    [0, 0, 1],  # Adjusted the shape to match the number of input neurons
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Desired output for AND gate
y = np.array([0, 0, 0, 1])

# Learning rate
alpha = 0.05

# Maximum number of epochs
max_epochs = 1000

# Convergence error threshold
convergence_error = 0.002

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Function to train the neural network using backpropagation
def train_neural_network(X, y, weights_input_hidden, weights_hidden_output, alpha, max_epochs, convergence_error):
    errors = []

    for epoch in range(max_epochs):
        # Forward pass
        hidden_input = np.dot(X, weights_input_hidden.T)
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(hidden_output, weights_hidden_output.T)
        predicted_output = sigmoid(output_input)

        # Calculate the error
        error = y - predicted_output
        errors.append(np.mean(np.abs(error)))

        # Backpropagation
        # Backpropagation
        output_delta = alpha * error * sigmoid_derivative(predicted_output)
        hidden_error = np.dot(output_delta.reshape(-1, 1), weights_hidden_output.reshape(1, -1))
        hidden_delta = alpha * hidden_error * sigmoid_derivative(hidden_output)


        # Update weights
        weights_hidden_output += np.dot(output_delta.T, hidden_output)
        weights_input_hidden += np.dot(hidden_delta.T, X)

        # Check for convergence
        if np.mean(np.abs(error)) <= convergence_error:
            print(f"Converged in {epoch + 1} epochs.")
            break

    return errors

# Train the neural network
errors = train_neural_network(X, y, weights_input_hidden, weights_hidden_output, alpha, max_epochs, convergence_error)

# Plotting errors over epochs
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Error Convergence Over Epochs')
plt.show()

# Display the final weights
print("Final weights (Input to Hidden):")
print(weights_input_hidden)
print("Final weights (Hidden to Output):")
print(weights_hidden_output)
