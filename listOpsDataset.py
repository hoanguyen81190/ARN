import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

def preprocess_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def preprocess_chunk(chunk, embedding_size):
    X, y = preprocess_data(chunk)
    X = X.reshape([X.shape[0], int(X.shape[1] / embedding_size), embedding_size])
    return X, y

def getDataLoader(X, y, batch_size=64, num_classes=10):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Create one-hot encoding
    y_onehot = np.eye(num_classes)[y]
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)  # Convert to tensor
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

def load_and_preprocess_in_chunks(filepath, chunk_size, embedding_size, batch_size, num_classes):
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, header=None):
        X_chunk, y_chunk = preprocess_chunk(chunk, embedding_size)
        loader = getDataLoader(X_chunk, y_chunk, batch_size, num_classes)
        yield loader