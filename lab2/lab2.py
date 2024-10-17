import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size) # Weights for the input layer
        self.b1 = np.zeros((1, self.hidden_size)) # Biases for the input layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size) # Weights for the hidden layer
        self.b2 = np.zeros((1, self.output_size)) # Biases for the hidden layer
            
    # Forward pass. This is where the input data is passed through the network and output is generated
    def forward(self, X):
        weighed_sum = np.dot(X, self.W1) + self.b1
        self.hidden_layer_output = np.tanh(weighed_sum)
        output_layer_output = np.dot(self.hidden_layer_output, self.W2) + self.b2 # Linear activation function
        
        return output_layer_output
    
    # Backward pass. This is where the network learns from the error and updates its weights and biases
    def backward(self, X, expected, output, learning_rate):
        num_samples = X.shape[0]

        error_output_layer = output - expected
        gradient_W2 = (1 / num_samples) * np.dot(self.hidden_layer_output.T, error_output_layer)
        gradient_b2 = (1 / num_samples) * np.sum(error_output_layer, axis=0, keepdims=True)

        error_hidden_layer = np.dot(error_output_layer, self.W2.T) * (1 - np.power(self.hidden_layer_output, 2))
        gradient_W1 = (1 / num_samples) * np.dot(X.T, error_hidden_layer)
        gradient_b1 = (1 / num_samples) * np.sum(error_hidden_layer, axis=0, keepdims=True)        
        
        self.W2 -= learning_rate * gradient_W2
        self.b2 -= learning_rate * gradient_b2
        self.W1 -= learning_rate * gradient_W1
        self.b1 -= learning_rate * gradient_b1

# Target function to get the expected output
def target_function(x):
    return (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2
    
# For 1 input
X = np.linspace(0, 1, 20).reshape(-1, 1)
expected_output = target_function(X)
mlp = MLP(input_size=1, hidden_size=6, output_size=1)

training_iterations = 100000 # AKA epochs
learning_rate = 0.01 

############
# Training #
############
for _ in range(training_iterations):
    output = mlp.forward(X)
    mlp.backward(X, expected_output, output, learning_rate)

###########
# Testing #
###########
TESTING_X = np.linspace(0, 1, 200).reshape(-1, 1)
predicted_output = mlp.forward(TESTING_X)
    
# Plot the expected and predicted output:
plt.figure(figsize=(10, 6))
plt.plot(expected_output, label='Expected')
plt.savefig('lab2/expected.png')

plt.figure(figsize=(10, 6))
plt.plot(predicted_output, label='Predicted')
plt.savefig('lab2/predicted.png')
