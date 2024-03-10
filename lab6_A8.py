import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_neural_network(X, y, out, learning_rate, epochs, error_threshold):
    input_size = 2
    hidden_size = 2
    output_size = out

    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

    errors = []

    for epoch in range(epochs):
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        final_output = sigmoid(final_input)

        error = y - final_output

        mean_abs_error = np.mean(np.abs(error))
        errors.append(mean_abs_error)

        if mean_abs_error <= error_threshold:
            break

        output_error = error * sigmoid_derivative(final_output)
        hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

        weights_hidden_output += hidden_output.T.dot(output_error) * learning_rate
        weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate
        bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, errors

def test_neural_network(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    return final_output

def main():
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    out = 1
    learning_rate = 0.05
    epochs = 10000
    error_threshold = 0.002

    weights_input_hidden_xor, weights_hidden_output_xor, bias_hidden_xor, bias_output_xor, errors = train_neural_network(
        X_xor, y_xor, out, learning_rate, epochs, error_threshold
    )

    predicted_output = test_neural_network(X_xor, weights_input_hidden_xor, weights_hidden_output_xor, bias_hidden_xor, bias_output_xor)
    print("Final Predicted Output:")
    print(predicted_output)

    # Plotting errors over epochs
    plt.plot(range(len(errors)), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error vs. Epochs for XOR Gate Neural Network')
    plt.show()

if __name__ == "_main_":
    main()