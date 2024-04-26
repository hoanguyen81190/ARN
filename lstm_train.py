import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMClassifier

def preprocess_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def getDataLoader(X, y, batch_size=64, num_classes=10):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Create one-hot encoding
    y_onehot = np.eye(num_classes)[y]
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)  # Convert to tensor
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

listops_dir = 'listops-1000'

train_filepath = os.path.join(os. getcwd(), listops_dir, 'short_train.csv')  
val_filepath = os.path.join(os. getcwd(), listops_dir, 'short_val.csv')  
test_filepath = os.path.join(os. getcwd(), listops_dir, 'short_test.csv')  

train_df = pd.read_csv(train_filepath, header=None)
val_df = pd.read_csv(val_filepath, header=None)
test_df = pd.read_csv(test_filepath, header=None)

X_train, y_train = preprocess_data(train_df)
X_val, y_val = preprocess_data(val_df)
X_test, y_test = preprocess_data(test_df)

#reshape X
X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], 1])
X_val = X_val.reshape([X_val.shape[0], X_val.shape[1], 1])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], 1])

num_classes = 10
batch_size = 64
train_loader = getDataLoader(X_train, y_train, batch_size, num_classes)
val_loader = getDataLoader(X_val, y_val, batch_size, num_classes)
test_loader = getDataLoader(X_test, y_test, batch_size, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size =  X_train.shape[2]# Define your input size based on the data
num_layers = 2
output_size = num_classes  # For binary classification

model = LSTMClassifier(input_size, hidden_size1=256, hidden_size2=128, num_classes=output_size)
model.to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = OrdinalRegressionLoss()
#criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

patience = 5
best_val_accuracy = 0
counter = 0
num_epochs = 100
train_loss = float('inf')

# Train your model
for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    
    # Training loop
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #print(inputs.shape)
        outputs = model(inputs)
        train_loss = criterion(outputs, targets)  # Targets need to be reshaped to match output size

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        target_labels = torch.argmax(targets, dim=1)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == target_labels).sum().item()
        total_samples += targets.size(0)

    train_accuracy = total_correct / total_samples
        
    # Validation loop
    model.eval()
    val_correct = 0
    val_samples = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            target_labels = torch.argmax(targets, dim=1)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == target_labels).sum().item()
            val_samples += targets.size(0)
            
        val_accuracy = total_correct / total_samples
    
        # Check if validation loss has decreased
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}. New best val accuracy {best_val_accuracy} - save model')
            # Save the best model
            torch.save(model.state_dict(), 'best_classification_model.pth')
            counter = 0
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}. Val accuracy does not improve, waiting {counter}')
            counter += 1
    
    # Check if early stopping criteria met
    if counter >= patience:
        print("Early stopping!")
        break