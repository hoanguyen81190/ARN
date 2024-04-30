import os
from dataset import load_file, processData, saveProcessedData, getShortSequences

listops_dir = 'listops-1000'
def getPath(filename):
    return os.path.join(os. getcwd(), listops_dir, filename)  

train_df = load_file(getPath('basic_train.tsv') )
val_df = load_file(getPath('basic_val.tsv') )
test_df = load_file(getPath('basic_test.tsv') )

#X_train, y_train = processData(train_df)
#X_val, y_val = processData(val_df)
#X_test, y_test = processData(test_df)

#saveProcessedData(X_train, y_train, getPath('train.csv'))
#saveProcessedData(X_val, y_val, getPath('val.csv'))
#saveProcessedData(X_test, y_test, getPath('test.csv'))

short_sequence_length = 512
#prepare short sequences
short_train_df = getShortSequences(train_df, short_sequence_length)
short_val_df = getShortSequences(val_df, short_sequence_length)
short_test_df = getShortSequences(test_df, short_sequence_length)



short_X_train, short_y_train = processData(short_train_df, sequence_size=short_sequence_length)
short_X_val, short_y_val = processData(short_val_df, sequence_size=short_sequence_length)
short_X_test, short_y_test = processData(short_test_df, sequence_size=short_sequence_length)

print(short_X_train.shape)

saveProcessedData(short_X_train, short_y_train, getPath(f'train_{short_sequence_length}.csv'))
saveProcessedData(short_X_val, short_y_val, getPath(f'val_{short_sequence_length}.csv'))
saveProcessedData(short_X_test, short_y_test, getPath(f'test_{short_sequence_length}.csv'))