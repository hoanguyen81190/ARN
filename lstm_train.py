import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMClassifier
from dataset import load_and_preprocess_in_chunks

def getConfig():
    parser = argparse.ArgumentParser(description="Process parameters.")
    parser.add_argument("max_sequence_length", type=int, help="The maximum length of the sequence of tokens")

    args = parser.parse_args()
    return args

"""def getDataLoader(X, y, batch_size=64, num_classes=10):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Create one-hot encoding
    y_onehot = np.eye(num_classes)[y]
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)  # Convert to tensor
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
"""

listops_dir = 'listops-1000'

def evaluate_model(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            target_labels = torch.argmax(targets, dim=1)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == target_labels).sum().item()
            total_samples += targets.size(0)
            
    accuracy = total_correct / total_samples
    return accuracy

def train(args):
    print("Loading data...")
    train_filepath = os.path.join(os. getcwd(), listops_dir, f'train_{args.max_sequence_length}.csv')  
    val_filepath = os.path.join(os. getcwd(), listops_dir, f'val_{args.max_sequence_length}.csv')  
    test_filepath = os.path.join(os. getcwd(), listops_dir, f'test_{args.max_sequence_length}.csv')  

    train_df = pd.read_csv(train_filepath, header=None)
    val_df = pd.read_csv(val_filepath, header=None)
    test_df = pd.read_csv(test_filepath, header=None)

    embedding_size = 15
    num_classes = 10
    batch_size = 64
    chunk_size = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = embedding_size
    output_size = num_classes
    patience = 10
    learning_rates = [0.1, 0.01, 0.001]
    hidden_sizes = [64, 128, 256, 512]
    total_best_val_accuracy = 0
    best_hyperparameters = {}

    total_best_val_accuracy = 0
    best_hyperparameters = {}

    print("Starting hyperparameter tuning...")

    #prepare logging file
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H_%M_%S")
    logging_file = f"logging_{formatted_now}.txt"

    with open(logging_file, 'a') as file:
        # Train your model
        for lr in learning_rates:
            for hidden_size in hidden_sizes:
                best_val_accuracy = 0
                counter = 0
                num_epochs = 100
                train_loss = float('inf')

                model = LSTMClassifier(input_size, hidden_size=hidden_size, num_classes=output_size)
                model.to(device)

                # Define your loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                print(f"Training model with lr={lr} and hidden_size={hidden_size}")
                for epoch in range(num_epochs):
                    model.train()
                    total_correct = 0
                    total_samples = 0
                    
                    # Training loop
                    for train_loader in load_and_preprocess_in_chunks(train_filepath, chunk_size, embedding_size, batch_size, num_classes):
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
                    val_accuracy = 0
                    for val_loader in load_and_preprocess_in_chunks(val_filepath, chunk_size, embedding_size, batch_size, num_classes):
                        val_accuracy += evaluate_model(model, val_loader, device)
                    val_accuracy /= (chunk_size // batch_size)
                    
                    # Check if validation loss has decreased
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}. New best val accuracy {best_val_accuracy} - save model')
                        # Save the best model
                        torch.save(model.state_dict(), f'models/seq_{args.max_sequence_length}.pth')
                        counter = 0
                    else:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}. Val accuracy does not improve {val_accuracy}, best val accuracy {best_val_accuracy}, waiting {counter}')
                        counter += 1

                    if val_accuracy > total_best_val_accuracy:
                        total_best_val_accuracy = val_accuracy
                        best_hyperparameters = {'lr': lr, 'hidden_size': hidden_size}
                    
                    # Check if early stopping criteria met
                    if counter >= patience:
                        print("Early stopping!")
                        break
                
                print("Testing model...")
                test_accuracy = 0
                for test_loader in load_and_preprocess_in_chunks(test_filepath, chunk_size, embedding_size, batch_size, num_classes):
                    test_accuracy += evaluate_model(model, test_loader, device)
                test_accuracy /= (chunk_size // batch_size)

                logging_text = f"sequence_length: {args.max_sequence_length}, lr: {lr}, hidden_size: {hidden_size}, train_accuracy: {train_accuracy}, val_accuracy: {best_val_accuracy}, test_accuracy: {test_accuracy}\n"
                file.write(logging_text)

        file.write(f"Best validation accuracy: {total_best_val_accuracy}. Best hyperparameters: {best_hyperparameters} \n")

if __name__ == "__main__":
    args = getConfig()
    train(args)