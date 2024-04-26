# Download dataset
Follow the instructions in LRA's github https://github.com/google-research/long-range-arena

# Prepare the dataset
Run prepare_data.py to generate clean, processed data. 
- X will be tokenized
- y will be converted to one-hot encoding with 10 classes: 0-9

There is possible to prepare a short version of the dataset that only contains max_sequence_length, follow the code example

# Train with LSTM
Run lstm_train.py, the code use pytorch 