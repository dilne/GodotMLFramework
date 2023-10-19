import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the rely activation function
def relu(x):
    return np.maximum(0, x)

# Define the derivative of the relu function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define the leaky relu function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

# Define the derivative of the leaky relu function
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Initialize the weights for the input and hidden layers
def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

def he_initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    return weights_input_hidden, weights_hidden_output

# Feedforward through the network
def feedforward(X, weights_input_hidden, weights_hidden_output):
    hidden_output = relu(np.dot(X, weights_input_hidden))
    output = sigmoid(np.dot(hidden_output, weights_hidden_output))
    return hidden_output, output

# Gradient Descent Optimizer
def optim_sgd(X, y, learning_rate, weights_input_hidden, weights_hidden_output, hidden_output, output):
    output_error = y - np.squeeze(output)
    output_delta = output_error * sigmoid_derivative(np.squeeze(output))
    hidden_error = np.dot(output_delta.reshape(-1, 1), weights_hidden_output.T)
    hidden_delta = hidden_error * relu_derivative(hidden_output)
    weights_hidden_output += np.dot(hidden_output.T, output_delta.reshape(-1, 1)) * learning_rate
    weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output

def main():
    # Read the CSV file
    data = pd.read_csv('/Users/danielmilne/Documents/GitHub/GodotMLFramework/diabetes.csv')

    # Define the input and output data
    X = np.array(data.iloc[:, :8].values)
    y = np.array(data.iloc[:, 8].values)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    test_data = np.array([[6,148,72,35,0,33.6,0.627,50],
	[1,85,66,29,0,26.6,0.351,31]])

    input_size = 8
    hidden_size = 128
    output_size = 1

    # Initialize the weights
    weights_input_hidden, weights_hidden_output = he_initialize_weights(input_size, hidden_size, output_size)

    # Train the neural network
    epochs = 100
    learning_rate = 0.1
    
    for epoch in range(epochs):
        hidden_output, output = feedforward(X, weights_input_hidden, weights_hidden_output)
        weights_input_hidden, weights_hidden_output = optim_sgd(X, y, learning_rate, weights_input_hidden, weights_hidden_output, hidden_output, output)

        if epoch % 10 == 0:
            print(epoch)
            for data in test_data:
                _, prediction = feedforward(data, weights_input_hidden, weights_hidden_output)
                print(f"Input: {data}, Output: {prediction}")
            print()
        
    # Test the trained network
    for data in test_data:
        _, prediction = feedforward(data, weights_input_hidden, weights_hidden_output)
        print(f"Input: {data}, Output: {prediction}")

main()
