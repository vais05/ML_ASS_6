import numpy as np

# Step 1: Data Preparation
data = np.array([
    [20, 6, 2, 1],
    [16, 3, 6, 1],
    [27, 6, 2, 1],
    [19, 1, 2, 0],
    [24, 4, 2, 1],
    [22, 1, 5, 0],
    [15, 4, 2, 1],
    [18, 4, 2, 1],
    [21, 1, 4, 0],
    [16, 2, 4, 0]
])

# Separate features and target
X = data[:, :-1]
y = data[:, -1]

# Step 2: Initialize Weights and Learning Rate
weights = np.random.rand(X.shape[1])
learning_rate = 0.01

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Step 3: Training the Perceptron
for epoch in range(1000):
    for i in range(X.shape[0]):
        # Forward pass
        z = np.dot(X[i], weights)
        prediction = sigmoid(z)
        
        # Calculate the error
        error = y[i] - prediction
        
        # Backpropagation
        weights += learning_rate * error * sigmoid_derivative(prediction) * X[i]

# Step 4: Evaluation
for i in range(X.shape[0]):
    z = np.dot(X[i], weights)
    prediction = sigmoid(z)
    print(f"Transaction {i+1}: Predicted High Value = {prediction > 0.5}")
