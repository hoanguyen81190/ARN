import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn

from models import LSTMClassifier
from listOpsDataset import load_and_preprocess_in_chunks

def getConfig():
    parser = argparse.ArgumentParser(description="Process parameters.")
    parser.add_argument("max_sequence_length", type=int, help="The maximum length of the sequence of tokens")

    args = parser.parse_args()
    return args

listops_dir = 'listops-1000'

def evaluate_model(model, val_filepath, chunk_size, embedding_size, batch_size, num_classes, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for loader in load_and_preprocess_in_chunks(val_filepath, chunk_size, embedding_size, batch_size, num_classes):
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
    os.makedirs("models", exist_ok=True)

    print("Starting hyperparameter tuning...")

    #prepare logging file
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H_%M_%S")
    os.makedirs("loggings", exist_ok=True)
    logging_file = os.path.join("loggings", f"logging_{formatted_now}.txt")

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
                    val_accuracy = evaluate_model(model, val_filepath, chunk_size, embedding_size, batch_size, num_classes, device)
                    
                    # Check if validation loss has decreased
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}. New best val accuracy {best_val_accuracy} - save model')
                        # Save the best model
                        torch.save(model.state_dict(), os.path.join('models', f'seq_{args.max_sequence_length}.pth'))
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
                test_accuracy = evaluate_model(model, test_filepath, chunk_size, embedding_size, batch_size, num_classes, device)

                logging_text = f"sequence_length: {args.max_sequence_length}, lr: {lr}, hidden_size: {hidden_size}, train_accuracy: {train_accuracy}, val_accuracy: {best_val_accuracy}, test_accuracy: {test_accuracy}\n"
                file.write(logging_text)

        file.write(f"Best validation accuracy: {total_best_val_accuracy}. Best hyperparameters: {best_hyperparameters} \n")

if __name__ == "__main__":
    args = getConfig()
    train(args)