import pandas as pd
import numpy as np
import re
import torch

token_dict = {
    '0': 0, 
    '1': 1, 
    '2': 2, 
    '3': 3, 
    '4': 4, 
    '5': 5, 
    '6': 6, 
    '7': 7, 
    '8': 8, 
    '9': 9, 
    ']': 10, 
    'MIN': 11, 
    'MAX': 12, 
    'SM': 13, 
    'MED': 14
}

def load_file(file_name):
    data = pd.read_csv(file_name)
    
    def split_sequence(sequence):
        tokens = sequence.split('\t')  # Split the sequence into tokens
        source = tokens[0] # Join all tokens except the last one for Source
        target = tokens[1]  # Last token is the Target
        return source, target
    
    data['Source'], data['Target'] = zip(*data['Source\tTarget'].map(split_sequence))
    data['Source'] = data['Source'].apply(lambda x: re.sub(r'[\(\)\[]', '', x))
    data.drop(columns=['Source\tTarget'], inplace=True)
    return data

def tokenize(row):
    tokens = re.findall(r'(MIN|MAX|MED|SM|\d+|\])', row)
    return [token_dict[t] for t in tokens]

def pad_sequences(sequences, sequence_size):
    # Initialize a new numpy array to store padded sequences
    padded_sequences = np.zeros((len(sequences), sequence_size), dtype=np.int32)
    
    # Pad each sequence with zeros to match the maximum length
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    return np.asarray(padded_sequences)

def processData(data, sequence_size=2000):
    X_array = data['Source'].apply(tokenize)  # Apply function row-wise

    # Convert the result to a numpy array
    X = np.array(X_array)
    padded_X = pad_sequences(X, sequence_size)
    y = data['Target']
    return padded_X, y

def saveProcessedData(X, y, filepath):
    combined_data = np.column_stack((X, y))
    np.savetxt(filepath, combined_data, delimiter=',', fmt='%s')

def getShortSequences(df, max_sequence_length):
    df['Token_Count'] = df['Source'].apply(lambda x: len(re.findall(r'(MIN|MAX|MED|SM|\d+|\])', x)))
    short_df = df[df['Token_Count'] <= max_sequence_length]

    return short_df

tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ']', 'MIN', 'MAX', 'SM', 'MED']
token_to_index = {token: index for index, token in enumerate(tokens)}

def onehotEncode(row, max_sequence_length):
    sequences = re.findall(r'(MIN|MAX|MED|SM|\d+|\])', row)
    sequence_indices = [token_to_index[token] for token in sequences]
    padded_sequence_indices = sequence_indices + [0] * (max_sequence_length - len(sequence_indices))
    padded_sequence_tensor = torch.tensor(padded_sequence_indices, dtype=torch.int32)
    
    # Convert integer indices to one-hot encodings
    one_hot_sequence = torch.eye(len(tokens))[padded_sequence_tensor]

    return one_hot_sequence.numpy()
    
def processOneHotData(data, sequence_size=2000):
    df = getShortSequences(data, sequence_size)
    if len(df) <= 0:
        return None, None

    X_array = df['Source'].apply(lambda row: onehotEncode(row, sequence_size))  # Apply function row-wise

    # Convert the result to a numpy array
    X = np.array(X_array)
    X = np.stack(X)
    y = df['Target']
    return X, y

def saveOneHotData(X, y, filepath):
    X = X.reshape([X.shape[0], X.shape[1]*X.shape[2]])
    combined_data = np.column_stack((X, y))
    np.savetxt(filepath, combined_data, delimiter=',', fmt='%s')

def saveOneHotDataAppend(X, y, filepath):
    X = X.reshape([X.shape[0], X.shape[1]*X.shape[2]])
    combined_data = np.column_stack((X, y))
    with open(filepath, 'ab') as f:
        np.savetxt(f, combined_data, delimiter=',', fmt='%s')