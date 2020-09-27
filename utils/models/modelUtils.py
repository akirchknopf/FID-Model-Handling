import numpy as np



from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Model

from ..telegramUtils.telegram_bot import telegram_send_message


# Shorten it if its not even -> do it for fine tune and whole model!
def checkDataframe(df, batch_size):
    if (len(df) % batch_size) != 0:
        rest = len(df) % batch_size
        df = df[:(len(df)-rest)]
        print(f"Warning: Needed to remove {rest} elements to make it matching from dataframe")
        return df
    else:
        return df
    
# Loadings means and sts from config file
def loadMeanFromFile(pathToConfigFile, verbose=False):
    f = open(pathToConfigFile, "r")
    txtRaw = f.readline()
    if verbose:
        print(f'Raw string: {txtRaw}')
    txtRaw = txtRaw.replace("[", '')
    txtRaw = txtRaw.replace("]", '')
    txtRaw = txtRaw.split(' ')
    if verbose:
        print(f'Spitted string: {txtRaw}')
    R = float(txtRaw[0])
    G = float(txtRaw[1])
    B = float(txtRaw[2])
    meansOfDataset = np.array([R, G, B])
    return (meansOfDataset)


def calcAccTextModel(model, data_x, data_y, name, GLOBAL_BATCH_SIZE):
    test = model.predict(data_x, use_multiprocessing=False, batch_size = GLOBAL_BATCH_SIZE, verbose=1)
    test_max = np.argmax(test,axis=1)
    unique_elements, counts_elements = np.unique(test_max, return_counts=True)
    y_true = []
    for index, element in enumerate(data_y):
        y_true.append(element)
    y_true = [int(i) for i in y_true]

    acc = accuracy_score(np.array(y_true), np.array(test_max))

    print(f'{name} Accuracy is {acc}')
    telegram_send_message(f'{name} Acc: {acc}')
    return f'{name} Accuracy is {acc}'
    
def calAccImageModel(model, test_seq, name, GLOBAL_BATCH_SIZE):
    test = model.predict(test_seq, use_multiprocessing=False, batch_size = GLOBAL_BATCH_SIZE, verbose=1)
    test_max = np.argmax(test,axis=1)
    unique_elements, counts_elements = np.unique(test_max, return_counts=True)
    y_true = []
    for index, element in enumerate(test_seq):
        y_true.append(element[1])

    y_true = [item for sublist in y_true for item in sublist]
    y_true = [int(i) for i in y_true]

    acc = accuracy_score(np.array(y_true), np.array(test_max))
    print(f' {name} Acc: {acc}')
    telegram_send_message(f' {name} Acc: {acc}')
    return f'{name} Accuracy is {acc}'
    
def calcAccMetaModel(model, test_seq_whole, GLOBAL_BATCH_SIZE, name):
    test = model.predict(test_seq_whole, batch_size = GLOBAL_BATCH_SIZE, verbose=1)
    test_max = np.argmax(test,axis=1)
    unique_elements, counts_elements = np.unique(test_max, return_counts=True)

    y_true = []
    for index, element in enumerate(test_seq_whole):
        y_true.append(element[1])

    y_true = [item for sublist in y_true for item in sublist]
    y_true = [int(i) for i in y_true]

    acc = accuracy_score(np.array(y_true), np.array(test_max))
    print(f'{name} Accuracy is {acc}')
    telegram_send_message(f'{name} Acc: {acc}')
    return f'{name} Accuracy is {acc}'

    
def calcAccConcatModel(model_concat, test_seq, GLOBAL_BATCH_SIZE, name):
    test = model_concat.predict(test_seq, use_multiprocessing=False, batch_size = GLOBAL_BATCH_SIZE, verbose=1)
    test_max = np.argmax(test,axis=1)
    unique_elements, counts_elements = np.unique(test_max, return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))
    y_true = []
    for index, element in enumerate(test_seq):
        y_true.append(element[1])

    y_true = [item for sublist in y_true for item in sublist]
    y_true = [int(i) for i in y_true]

    acc = accuracy_score(np.array(y_true), np.array(test_max))
    print(f'{name} Accuracy is {acc}')
    telegram_send_message(f'{name} Acc: {acc}')
    return f'{name} Accuracy is {acc}'