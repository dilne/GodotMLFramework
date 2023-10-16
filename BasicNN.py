import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the weights for the input and hidden layers
def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

# Feedforward through the network
def feedforward(X, weights_input_hidden, weights_hidden_output):
    hidden_output = sigmoid(np.dot(X, weights_input_hidden))
    output = sigmoid(np.dot(hidden_output, weights_hidden_output))
    return hidden_output, output

# Backpropagate and update the weights
def backpropagate(X, y, learning_rate, weights_input_hidden, weights_hidden_output, hidden_output, output):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weights_hidden_output += np.dot(hidden_output.T, output_delta) * learning_rate
    weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
    return weights_input_hidden, weights_hidden_output

# Example usage
def main():
    # Define the input and output data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 2
    output_size = 1

    # Initialize the weights
    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    # Train the neural network
    epochs = 100000
    learning_rate = 0.1

    for epoch in range(epochs):
        hidden_output, output = feedforward(X, weights_input_hidden, weights_hidden_output)
        weights_input_hidden, weights_hidden_output = backpropagate(X, y, learning_rate, weights_input_hidden, weights_hidden_output, hidden_output, output)

    # Test the trained network
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for data in test_data:
        _, prediction = feedforward(data, weights_input_hidden, weights_hidden_output)
        print(f"Input: {data}, Output: {prediction}")
        # Input: [0 0], Output: [0.03241971]
        # Input: [0 1], Output: [0.93139128]
        # Input: [1 0], Output: [0.93139131]
        # Input: [1 1], Output: [0.09149361]

main()
