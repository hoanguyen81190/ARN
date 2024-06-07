import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # Define the fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Forward pass through the LSTM layer
        out, _ = self.lstm(x)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Shape: (batch_size, num_classes)
        
        return out