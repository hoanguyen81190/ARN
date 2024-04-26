import pandas as pd
import numpy as np
import re

token_dict = {
    '0': 1, 
    '1': 2, 
    '2': 3, 
    '3': 4, 
    '4': 5, 
    '5': 6, 
    '6': 7, 
    '7': 8, 
    '8': 9, 
    '9': 10, 
    ']': 11, 
    'MIN': 12, 
    'MAX': 13, 
    'SM': 14, 
    'MED': 15
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
