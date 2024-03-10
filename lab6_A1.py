import numpy as np
import matplotlib.pyplot as plt

W = np.array([10, 0.2, -0.75])

X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
y = np.array([0, 0, 0, 1])

alpha = 0.05

max_epochs = 1000

convergence_error = 0.002

def step_activation(x):
    return 1 if x >= 0 else 0
def train_perceptron(X, y, W, alpha, max_epochs, convergence_error):
    error_values = []

    for epoch in range(max_epochs):
        error_sum = 0

        for i in range(len(X)):
            prediction = step_activation(np.dot(X[i], W))
            error = y[i] - prediction

            W = W + alpha * error * X[i]
            error_sum += error ** 2

        total_error = 0.5 * error_sum

        error_values.append(total_error)

        if total_error <= convergence_error:
            print(f"Converged in {epoch + 1} epochs.")
            break

    return W, error_values

final_weights, errors = train_perceptron(X, y, W, alpha, max_epochs, convergence_error)

plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Error Convergence Over Epochs')
plt.show()

print("Final weights:", final_weights)
