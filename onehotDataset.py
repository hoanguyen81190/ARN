import numpy as np
import torch
import re

max_sequence_length = 2000 #max length sequence from train dataset

tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ']', 'MIN', 'MAX', 'SM', 'MED']
token_to_index = {token: index for index, token in enumerate(tokens)}

def onehotEncode(row, max_sequence_length=2000):
    sequences = re.findall(r'(MIN|MAX|MED|SM|\d+|\])', row)
    sequence_indices = [token_to_index[token] for token in sequences]
    padded_sequence_indices = sequence_indices + [0] * (max_sequence_length - len(sequence_indices))
    padded_sequence_tensor = torch.tensor(padded_sequence_indices, dtype=torch.int32)
    
    # Convert integer indices to one-hot encodings
    one_hot_sequence = torch.eye(len(tokens))[padded_sequence_tensor]

    return one_hot_sequence.numpy()
    
def processData(data):
    X_array = data['Source'].apply(onehotEncode)  # Apply function row-wise

    # Convert the result to a numpy array
    X = np.array(X_array)
    X = np.stack(X)
    y = data['Target']
    return X, y