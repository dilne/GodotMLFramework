import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

# Fully connected layer
def fully_connected(input_size, output_size, activation):
    return {'weights': 2 * np.random.random((input_size, output_size)) - 1,
            'activation': activation}

# Training the network
def train_network(network, X, y, lr, epochs, optimizer='gradient_descent'):
    start_time = time.time()    # Start the timer
    # Initialize parameters for Adam optimizer
    if optimizer == 'adam':
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m = [0 for _ in range(len(network))]
        v = [0 for _ in range(len(network))]

    for epoch in range(epochs):
        # Forward propagation
        layers = [X]
        for i in range(len(network)):
            if network[i]['activation'] == 'sigmoid':
                layers.append(sigmoid(np.dot(layers[i], network[i]['weights'])))
            elif network[i]['activation'] == 'relu':
                layers.append(relu(np.dot(layers[i], network[i]['weights'])))

        # Backpropagation
        deltas = [y - layers[-1]]
        for i in range(len(network)-2, -1, -1):
            error = deltas[-1].dot(network[i+1]['weights'].T)
            if network[i]['activation'] == 'sigmoid':
                delta = error * sigmoid_derivative(layers[i+1])
            elif network[i]['activation'] == 'relu':
                delta = error * relu_derivative(layers[i+1])
            deltas.append(delta)

        # Update weights
        for i in range(len(network)):
            if optimizer == 'gradient_descent':
                network[i]['weights'] += lr * layers[i].T.dot(deltas[-(i+1)])
            elif optimizer == 'adam':
                m[i] = beta1 * m[i] + (1 - beta1) * layers[i].T.dot(deltas[-(i+1)])
                v[i] = beta2 * v[i] + (1 - beta2) * np.square(layers[i].T.dot(deltas[-(i+1)]))
                m_hat = m[i] / (1 - np.power(beta1, epoch+1))
                v_hat = v[i] / (1 - np.power(beta2, epoch+1))
                network[i]['weights'] += lr * m_hat / (np.sqrt(v_hat) + epsilon)
    end_time = time.time()  # Stop the timer
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Train time taken: {elapsed_time} seconds")
    return network

def make_predictions(network, X):
    layers = [X]
    for i in range(len(network)):
        if network[i]['activation'] == 'sigmoid':
            layers.append(sigmoid(np.dot(layers[i], network[i]['weights'])))
        elif network[i]['activation'] == 'relu':
            layers.append(relu(np.dot(layers[i], network[i]['weights'])))
    return layers[-1]


# Creating the model
network = [fully_connected(8, 8, 'sigmoid'),
           fully_connected(8, 8, 'sigmoid'),
           fully_connected(8, 1, 'sigmoid')]

data = pd.read_csv('/Users/danielmilne/Documents/GitHub/GodotMLFramework/diabetes.csv')
# Split the dataset into features and labels
X = np.array(data.iloc[:, :8].values)
y = np.array(data.iloc[:, 8].values).reshape(-1, 1)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
lr = 0.001
epochs = 100
network = train_network(network, X_train, y_train, lr, epochs, optimizer='gradient_descent')

# Making predictions
start_time = time.time()    # Start the timer
predictions = make_predictions(network, X_test)
predicted_labels = (predictions > 0.5).astype(float)

# Calculate the number of correct predictions
correct_predictions = (predicted_labels == y_test.reshape(-1, 1)).sum()
print(f"Total correct predictions: {correct_predictions} out of {len(X_test)}")
end_time = time.time()  # Stop the timer
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Prediction time taken: {elapsed_time} seconds")
