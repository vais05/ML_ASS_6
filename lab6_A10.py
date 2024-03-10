from sklearn.neural_network import MLPClassifier
import numpy as np

# AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# XOR gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create MLP classifiers with adjustments
mlp_and = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=5000, random_state=1, tol=1e-4)
mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=5000, random_state=1, tol=1e-4)

# Train the models
mlp_and.fit(X_and, y_and)
mlp_xor.fit(X_xor, y_xor)

# Test the models
print("AND gate predictions:")
print(mlp_and.predict(X_and))

print("XOR gate predictions:")
print(mlp_xor.predict(X_xor))
