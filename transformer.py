import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F

from transformers import LongformerForSequenceClassification

def preprocess_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def create_attention_mask(tokenized_inputs):
    attention_mask = torch.ones_like(tokenized_inputs)
    attention_mask[tokenized_inputs == 0] = 0  # Set the attention mask to 0 for padding tokens
    return attention_mask

def getDataLoader(X, y, batch_size=64, num_classes=10):
    X_tensor = torch.tensor(X, dtype=torch.int32)
    X_attMask = create_attention_mask(X_tensor)

    # Create one-hot encoding
    y_onehot = np.eye(num_classes)[y]
    y_tensor = torch.tensor(y_onehot, dtype=torch.int32)  # Convert to tensor
    
    dataset = TensorDataset(X_tensor, X_attMask, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

sequence_length = 800
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
""" X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], 1])
X_val = X_val.reshape([X_val.shape[0], X_val.shape[1], 1])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], 1]) """

num_classes = 10
batch_size = 64
train_loader = getDataLoader(X_train, y_train, batch_size, num_classes)
val_loader = getDataLoader(X_val, y_val, batch_size, num_classes)
test_loader = getDataLoader(X_test, y_test, batch_size, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "allenai/longformer-base-4096"
model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=10).to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

patience = 5
best_val_accuracy = 0
counter = 0
num_epochs = 100
train_loss = float('inf')

# Train your model
for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    
    # Training loop
    for batch in train_loader:
        inputs = {"input_ids": batch[0].to(device), "attention_mask": batch[1].to(device)}
        labels = batch[2].to(device)
        print({k: v.shape for k, v in inputs.items()})

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        logits = outputs.logits
        predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
        total_correct += (predictions == labels).sum().item()

    train_accuracy = total_correct / len(train_loader)
        
    # Validation loop
    model.eval()
    val_correct = 0
    val_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {"input_ids": batch[0].to(device), "attention_mask": batch[1].to(device)}
            labels = batch[2].to(device)

            logits = outputs.logits
            predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
            val_correct += (predictions == labels).sum().item()
            
        val_accuracy = total_correct / len(val_loader)
    
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