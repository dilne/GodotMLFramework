import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Pima Diabetes dataset from a CSV file
data = pd.read_csv('/Users/danielmilne/Documents/GitHub/GodotMLFramework/diabetes.csv')

# Split the dataset into features and labels
X = np.array(data.iloc[:, :8].values)
y = np.array(data.iloc[:, 8].values)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)

# Define the neural network model
class DiabetesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)  # Input layer
        self.fc2 = nn.Linear(8, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x) 
        return x


model = DiabetesModel()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# Train the model for 100 epochs
for epoch in range(50):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    #print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# Convert y_test to a PyTorch tensor
y_test = torch.FloatTensor(y_test)

# Make predictions on the test data
with torch.no_grad():
    predictions = model(X_test)
    predicted_labels = (predictions > 0.5).float()

# Calculate the number of correct predictions
correct_predictions = (predicted_labels == y_test.view(-1, 1)).sum().item()

print(f"Total correct predictions: {correct_predictions} out of {len(X_test)}")
