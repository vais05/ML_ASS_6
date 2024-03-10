import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid activation function
    """
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    """
    Simple neural network with one hidden layer and two output neurons
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w1 = np.random.rand(n_features, hidden_size)  # Weights between input and hidden layer
        self.w2 = np.random.rand(hidden_size, 2)            # Weights between hidden and output layer
        self.b1 = np.zeros((1, hidden_size))                # Bias for hidden layer
        self.b2 = np.zeros((1, 2))                          # Bias for output layer

    def predict(self, X):
        """
        Forward pass of the network
        """
        net_hidden = np.dot(X, self.w1) + self.b1
        hidden_output = sigmoid(net_hidden)
        net_output = np.dot(hidden_output, self.w2) + self.b2
        output = sigmoid(net_output)
        return output

    def train(self, X, Y, epochs=1000):
        """
        Train the network using backpropagation
        """
        for epoch in range(epochs):
            # Forward pass
            net_hidden = np.dot(X, self.w1) + self.b1
            hidden_output = sigmoid(net_hidden)
            net_output = np.dot(hidden_output, self.w2) + self.b2
            output = sigmoid(net_output)

            # Error calculation
            error = Y - output

            # Backpropagation
            delta_output = error * sigmoid_derivative(output)
            delta_hidden = np.dot(delta_output, self.w2.T) * sigmoid_derivative(hidden_output)

            # Weight update
            self.w2 += self.learning_rate * np.dot(hidden_output.T, delta_output)
            self.w1 += self.learning_rate * np.dot(X.T, delta_hidden)

            # Convergence check (adjust convergence criteria as needed)
            total_mse = np.mean(np.square(error))
            if total_mse <= 0.01:
                print(f"Converged after {epoch+1} epochs")
                break

# Example usage
n_features = 2  # Number of input features (adjust based on your logic gate)
hidden_size = 4   # Number of hidden neurons (can be adjusted)
network = NeuralNetwork()

# Define training data (replace with your specific logic gate truth table)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_and = np.array([[0, 0], [0, 1], [0, 1], [1, 0]])  # Example: AND gate

# Train the network
network.train(X_and, Y_and)

# Test the network
print("Output for (0, 0):", network.predict(np.array([0, 0])))
print("Output for (0, 1):", network.predict(np.array([0, 1])))
print("Output for (1, 0):", network.predict(np.array([1, 0])))
print("Output for (1, 1):", network.predict(np.array([1, 1])))
