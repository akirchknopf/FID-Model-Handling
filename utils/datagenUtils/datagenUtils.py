import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence

def loadDataFrameFromPath(pathToCSVWithFileNamesAndLabels):
    df = pd.read_csv(pathToCSVWithFileNamesAndLabels, header=0, sep='\t')
    df.columns = ['id', 'fileName', 'label']
    df = df.astype(str)
    df['label'] = df['label'].apply(lambda x: np.array(x))
    return df

def loadDataFrameFromPathForNNTextAndImage(pathToCSVWithFileNamesAndLabels):
    df = pd.read_csv(pathToCSVWithFileNamesAndLabels, header=1, sep='\t', usecols=range(1,5))
    df.columns = ['id', 'fileName', 'text', 'label']
    df = df.astype(str)
    df['label'] = df['label'].apply(lambda x: np.array(x))
    return df

def convertRowToDictionary(row, columns, hasIndex = False):
    dict = {}
    for idx, col in enumerate(columns):
        if hasIndex:
            dict['index'] = row[idx]
        dict[col] = row[idx + 1]
    return dict

# Shorten it if its not even -> do it for fine tune and whole model!
def checkDataframe(df, batch_size):
    if (len(df) % batch_size) != 0:
        rest = len(df) % batch_size
        df = df[:(len(df)-rest)]
        print(f"Warning: Needed to remove {rest} elements to make it matching from dataframe")
        return df
    else:
        return df