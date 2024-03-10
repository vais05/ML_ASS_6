import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def step_activation(x):
    return 1 if x > 0 else 0

def bi_polar_step_activation(x):
    return -1 if x < 0 else 1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return max(0, x)

# Perceptron class
class Perceptron:
    def __init__(self, weights, learning_rate=0.05, max_iterations=1000, error_threshold=0.002):
        self.weights = weights
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold

    def predict(self, inputs, activation_func):
        weighted_sum = np.dot(self.weights, inputs)
        return activation_func(weighted_sum)

    def train(self, training_inputs, training_outputs, activation_func):
        epochs = 0
        error_values = []
        while epochs < self.max_iterations:
            predictions = [self.predict(inputs, activation_func) for inputs in training_inputs]
            error = sum([(output - prediction)**2 for output, prediction in zip(training_outputs, predictions)])
            error_values.append(error)
            
            if error <= self.error_threshold:
                break
            
            for i in range(len(training_inputs)):
                prediction = self.predict(training_inputs[i], activation_func)
                # Convert training_inputs[i] to a NumPy array for element-wise multiplication
                training_inputs_i_array = np.array(training_inputs[i])
                # Update weights using NumPy array operations
                self.weights += self.learning_rate * (training_outputs[i] - prediction) * training_inputs_i_array
            
            epochs += 1
        
        return epochs, error_values

# Training and plotting
training_inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
training_outputs = [0, 0, 0, 1]
weights = [10, 0.2, -0.75]

activation_functions = [step_activation, bi_polar_step_activation, sigmoid_activation, relu_activation]
for activation_func in activation_functions:
    perceptron = Perceptron(weights, learning_rate=0.05)
    epochs, error_values = perceptron.train(training_inputs, training_outputs, activation_func)
    
    # After training, adjust the epochs to match the length of error_values
    adjusted_epochs = len(error_values)

    # Plotting
    plt.plot(range(adjusted_epochs), error_values, label=activation_func.__name__)
    plt.xlabel('Epochs')
    plt.ylabel('Error Values')
    plt.legend()
    plt.show()
