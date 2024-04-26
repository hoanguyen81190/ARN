import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        
        # Define the first LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers=1, batch_first=True)
        
        # Define the second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers=num_layers-1, batch_first=True)
        
        # Define the fully connected layer for classification
        self.fc = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        # Forward pass through the first LSTM layer
        out, _ = self.lstm1(x)
        
        # Forward pass through the second LSTM layer
        out, _ = self.lstm2(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Shape: (batch_size, num_classes)
        
        return out