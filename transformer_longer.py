import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, BCELoss

from transformers import LongformerForSequenceClassification

def preprocess_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def create_attention_mask(tokenized_inputs):
    attention_mask = torch.ones_like(tokenized_inputs, dtype=torch.int32)
    attention_mask[tokenized_inputs == 0] = 0  # Set the attention mask to 0 for padding tokens
    return attention_mask

def getDataLoader(X, y, batch_size, num_classes=10):
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

short_sequence_length = 512

train_filepath = os.path.join(os. getcwd(), listops_dir, f'train_{short_sequence_length}.csv')  
val_filepath = os.path.join(os. getcwd(), listops_dir, f'train_{short_sequence_length}.csv')  
test_filepath = os.path.join(os. getcwd(), listops_dir, f'train_{short_sequence_length}.csv')  

train_df = pd.read_csv(train_filepath, header=None)
val_df = pd.read_csv(val_filepath, header=None)
test_df = pd.read_csv(test_filepath, header=None)

X_train, y_train = preprocess_data(train_df)
X_val, y_val = preprocess_data(val_df)
X_test, y_test = preprocess_data(test_df)

num_classes = 10
batch_size = 16
train_loader = getDataLoader(X_train, y_train, batch_size, num_classes)
val_loader = getDataLoader(X_val, y_val, batch_size, num_classes)
test_loader = getDataLoader(X_test, y_test, batch_size, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "allenai/longformer-base-4096"


#config = PerceiverConfig()

model = LongformerForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
model.cuda()


# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=2e-5)

patience = 5
best_val_accuracy = 0
counter = 0
num_epochs = 100
num_labels = 10

train_loss_set = []

# Train your model
for epoch in range(num_epochs):
    model.train()
    total_correct = 0

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    
    # Training loop
    for batch in train_loader:
        inputs = {"input_ids": batch[0].to(device), "token_type_ids": None, "attention_mask": batch[1].to(device)}
        labels = batch[2].to(device)
        #print({k: v.shape for k, v in inputs.items()})
        #print(batch[0].shape, batch[1].shape, batch[2].shape)

        optimizer.zero_grad()
        outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=batch[1].to(device))
        logits = outputs[0]
        loss_func = BCEWithLogitsLoss() 
        loss = loss_func(logits.view(-1,num_labels), labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
        # loss_func = BCELoss() 
        # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
        train_loss_set.append(loss.item())    

        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # scheduler.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += batch[0].size(0)
        nb_tr_steps += 1
        
    # Validation loop
    model.eval()
    val_correct = 0
    val_samples = 0
    for i, batch in enumerate(val_loader):
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch[0].to(device), batch[1].to(device), batch[2].to(device), None
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

            predicted_classes = np.argmax(pred_label, axis=1)
            true_classes = np.argmax(b_labels, axis=1)
            val_correct += np.sum(predicted_classes == true_classes)
            val_samples += b_labels.shape[0]

    val_accuracy = val_correct / val_samples
    # Check if validation loss has decreased
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {tr_loss/nb_tr_steps:.4f}. New best val accuracy {best_val_accuracy} - save model')
        # Save the best model
        torch.save(model.state_dict(), 'best_classification_model.pth')
        counter = 0
    else:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {tr_loss/nb_tr_steps:.4f}. Val accuracy does not improve {best_val_accuracy}, waiting {counter}')
        counter += 1
    
    # Check if early stopping criteria met
    if counter >= patience:
        print("Early stopping!")
        break