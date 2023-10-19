import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Pima Diabetes dataset from a CSV file
data = pd.read_csv('/Users/danielmilne/Documents/GitHub/GodotMLFramework/diabetes.csv') # replace with your own

# Split the dataset into features and labels
X = np.array(data.iloc[:, :8].values)
y = np.array(data.iloc[:, 8].values)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize
def initialize(inputLayer_neurons, hiddenLayer_neurons, outputLayer_neurons):
    wh=np.random.uniform(size=(inputLayer_neurons,hiddenLayer_neurons))
    bh=np.random.uniform(size=(1,hiddenLayer_neurons))
    wout=np.random.uniform(size=(hiddenLayer_neurons,outputLayer_neurons))
    bout=np.random.uniform(size=(1,outputLayer_neurons))
    return wh, bh, wout, bout

def feedforward(X_train, wh, bh, bout):
    hidden_layer_input1=np.dot(X_train, wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)
    return output, hiddenlayer_activations

def backpropagation(X_train, y_train, output, hiddenlayer_activations, wh, bh, wout, bout):
    E = y_train.reshape(-1, 1) - output
    slope_output_layer = sigmoid_derivative(output)
    slope_hidden_layer = sigmoid_derivative(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

    wout += hiddenlayer_activations.T.dot(d_output) * 0.001
    bout += np.sum(d_output, axis=0, keepdims=True) * 0.001
    wh += X_train.T.dot(d_hiddenlayer) * 0.001
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * 0.001

    return wh, bh, wout, bout

# Define the structure of the neural network model
inputLayer_neurons = X_train.shape[1]  # number of features in data set
hiddenLayer_neurons = 8  # number of hidden layers neurons
outputLayer_neurons = 1  # number of neurons at output layer

# weight and bias initialization
wh, bh, wout, bout = initialize(inputLayer_neurons,
                                hiddenLayer_neurons,
                                outputLayer_neurons)

# Training the model for 500 epochs
for epoch in range(500):
    # Forward Propagation
    output, hiddenlayer_activations = feedforward(X_train, wh, bh, bout)

    # Backpropagation
    wh, bh, wout, bout = backpropagation(X_train,
                                         y_train,
                                         output,
                                         hiddenlayer_activations,
                                         wh,
                                         bh,
                                         wout,
                                         bout)

# Make predictions on the test data
test_hidden_layer_input1=np.dot(X_test, wh)
test_hidden_layer_input=test_hidden_layer_input1 + bh
test_hiddenlayer_activations = sigmoid(test_hidden_layer_input)
test_output_layer_input1=np.dot(test_hiddenlayer_activations,wout)
test_output_layer_input= test_output_layer_input1+ bout
predictions = sigmoid(test_output_layer_input)

predicted_labels = (predictions > 0.5).astype(float)

# Calculate the number of correct predictions
correct_predictions = (predicted_labels == y_test.reshape(-1, 1)).sum()

print(f"Total correct predictions: {correct_predictions} out of {len(X_test)}")
