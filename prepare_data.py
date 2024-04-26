import os
from dataset import load_file, processData, saveProcessedData, getShortSequences

listops_dir = 'listops-1000'
def getPath(filename):
    return os.path.join(os. getcwd(), listops_dir, filename)  

train_df = load_file(getPath('basic_train.tsv') )
val_df = load_file(getPath('basic_val.tsv') )
test_df = load_file(getPath('basic_test.tsv') )

X_train, y_train = processData(train_df)
X_val, y_val = processData(val_df)
X_test, y_test = processData(test_df)

saveProcessedData(X_train, y_train, getPath('train.csv'))
saveProcessedData(X_val, y_val, getPath('val.csv'))
saveProcessedData(X_test, y_test, getPath('test.csv'))

#prepare short sequences
short_train_df = getShortSequences(train_df, 800)
short_val_df = getShortSequences(val_df, 800)
short_test_df = getShortSequences(test_df, 800)



short_X_train, short_y_train = processData(short_train_df, sequence_size=800)
short_X_val, short_y_val = processData(short_val_df, sequence_size=800)
short_X_test, short_y_test = processData(short_test_df, sequence_size=800)

print(short_X_train.shape)

saveProcessedData(short_X_train, short_y_train, getPath('short_train.csv'))
saveProcessedData(short_X_val, short_y_val, getPath('short_val.csv'))
saveProcessedData(short_X_test, short_y_test, getPath('short_test.csv'))