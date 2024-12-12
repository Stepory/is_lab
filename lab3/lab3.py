import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def target_function(x):
    return  ((1 + 0.6 * np.sin(2 * np.pi * x / 0.7)) + 0.3 * np.sin(2 * np.pi * x)) / 2

# Radial basis function
def gaussian_rbf(x, c, r):
    return np.exp(-((x - c) ** 2) / (2 * r ** 2))

x_values = np.arange(0.1, 1 + 1/22, 1/22)

y_values = target_function(x_values)

# Parameters for the two RBFs C - centre, R - radius
# Chosen from y graph
peaks, _ = find_peaks(y_values)
c_values = x_values[peaks]

# Set spreads (r) based on distances between centers
# if len(c_values) > 1:
#     r_values = [0.5 * abs(c_values[i+1] - c_values[i]) for i in range(len(c_values) - 1)]
#     r_values.append(r_values[-1])
# else:
#     r_values = [0.1]
    
r_values = [0.15, 0.18]
    
print("Centres:", c_values)
print("Radiuses:", r_values)

plt.plot(y_values, label='y graph')
plt.savefig('lab3/graph.png')

# Calculate the RBFs for the input data
phi_matrix = np.array([gaussian_rbf(x_values, c, r) for c, r in zip(c_values, r_values)]).T

print("Phi matrix shape:", phi_matrix.shape)    

w = np.random.randn(len(c_values) + 1) 

learning_rate = 0.01
epochs = 100000

# TRAINING
for epoch in range(epochs):
    y_pred = w[0] + np.dot(phi_matrix, w[1:])
    error = y_values - y_pred
        
    # Update weights
    w[0] += learning_rate * np.sum(error)
    w[1:] += learning_rate * np.dot(error, phi_matrix)

print("error:", error)

# TESTING
x_test_values = np.arange(0.05, 1.05, 0.05)

y_test_values = target_function(x_test_values)

phi_test_matrix = np.array([gaussian_rbf(x_test_values, c, r) for c, r in zip(c_values, r_values)]).T

y_test_approx = w[0] + np.dot(phi_test_matrix, w[1:])

print("\nTesting on new inputs:")
print("Test inputs:", x_test_values)
print("Target outputs:", y_test_values)
print("Approximated outputs:", y_test_approx)

# Calculate mean squared error for test data
mse = np.mean((y_test_values - y_test_approx) ** 2)
print("Mean Squared Error on test data:", mse)

plt.figure(figsize=(10, 6))
plt.plot(y_test_values, label='Expected')
plt.savefig('lab3/expected.png')

plt.figure(figsize=(10, 6))
plt.plot(y_test_approx, label='Predicted')
plt.savefig('lab3/predicted.png')
