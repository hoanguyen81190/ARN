import os
import numpy as np
from dataset import load_file, processOneHotData, saveOneHotData, getShortSequences, saveOneHotDataAppend

listops_dir = 'listops-1000'
def getPath(filename):
    return os.path.join(os. getcwd(), listops_dir, filename)  

short_sequence_length = 2000
#prepare short sequences
""" print(f'Preparing training short sequences of length {short_sequence_length}...')
train_df = load_file(getPath('basic_train.tsv') )
short_train_df = getShortSequences(train_df, short_sequence_length)
short_X_train, short_y_train = processOneHotData(short_train_df, sequence_size=short_sequence_length)
del short_train_df
saveOneHotData(short_X_train, short_y_train, getPath(f'train_{short_sequence_length}.csv'))

print(f'Preparing validation short sequences of length {short_sequence_length}...')
val_df = load_file(getPath('basic_val.tsv') )
short_val_df = getShortSequences(val_df, short_sequence_length)
short_X_val, short_y_val = processOneHotData(short_val_df, sequence_size=short_sequence_length)
del short_val_df
saveOneHotData(short_X_val, short_y_val, getPath(f'val_{short_sequence_length}.csv'))

print(f'Preparing validation short sequences of length {short_sequence_length}...')
test_df = load_file(getPath('basic_test.tsv') )
short_test_df = getShortSequences(test_df, short_sequence_length)
short_X_test, short_y_test = processOneHotData(short_test_df, sequence_size=short_sequence_length)
del short_test_df
saveOneHotData(short_X_test, short_y_test, getPath(f'test_{short_sequence_length}.csv'))
print('Done!') """

def loadAndPrepare(filename, tag, short_sequence_length=2000):
    print(f'Preparing {tag} short sequences of length {short_sequence_length}...')
    df = load_file(getPath(filename))
    # Split train_df into 10 parts
    dfs = np.array_split(df, 10)

    for i, df in enumerate(dfs):
        print(f'Processing part {i+1}...')
        short_X_train, short_y_train = processOneHotData(df, sequence_size=short_sequence_length)
        # Append to the file instead of overwriting it
        if short_X_train is not None and short_y_train is not None:
            saveOneHotDataAppend(short_X_train, short_y_train, getPath(f'{tag}_{short_sequence_length}.csv'))
        del df

short_sequence_length = 700
loadAndPrepare('basic_train.tsv', 'train', short_sequence_length)
loadAndPrepare('basic_val.tsv', 'val', short_sequence_length)
loadAndPrepare('basic_test.tsv', 'test', short_sequence_length)
print('Done!')

short_sequence_length = 900
loadAndPrepare('basic_train.tsv', 'train', short_sequence_length)
loadAndPrepare('basic_val.tsv', 'val', short_sequence_length)
loadAndPrepare('basic_test.tsv', 'test', short_sequence_length)
print('Done!')