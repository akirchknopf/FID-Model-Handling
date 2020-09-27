# umbauen


from tqdm import tqdm
import pandas as pd
import numpy as np


class FakeDetectionDataCommentsTrainVal:
    DATA_COLUMN = "comments"
    LABEL_COLUMN = "2_way_label"

    def __init__(self, train, val, tokenizer, classes, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len # vorher 0 
        self.classes = classes
        self.max_seq_length = 0

        ((self.train_x, self.train_y), (self.val_x, self.val_y)) = map(self._prepare, [train, val])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.val_x = map(self._pad, [self.train_x, self.val_x])

    def _prepare(self, df):
        x, y = [], []

        for _, row in tqdm(df.iterrows()):
#             print(row)
            text, label = row[FakeDetectionDataCommentsTrainVal.DATA_COLUMN], row[FakeDetectionDataCommentsTrainVal.LABEL_COLUMN]
            label = int(label)
            text = str(text).replace('[', '').replace(']', '').replace("'", '')
            tokens = self.tokenizer.tokenize(str(text)) # Achtung str hinzugefügt
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            
            if self.max_seq_length < self.max_seq_len:
                self.max_seq_length = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)
    
    def getMaxSeqLength(self):
        return self.max_seq_length


class FakeDetectionDataCommentsTest:
    DATA_COLUMN = "comments"
    LABEL_COLUMN = "2_way_label"

    def __init__(self, test, tokenizer, classes, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.classes = classes

        (self.test_x, self.test_y) = self._prepare(test)

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.test_x = self._pad(self.test_x)

    def _prepare(self, df):
        x, y = [], []

        for _, row in tqdm(df.iterrows()):
#             print(row)
            text, label = row[FakeDetectionDataCommentsTest.DATA_COLUMN], row[FakeDetectionDataCommentsTest.LABEL_COLUMN]
            label = int(label)
            text = str(text).replace('[', '').replace(']', '').replace("'", '')
            tokens = self.tokenizer.tokenize(str(text)) # Achtung str hinzugefügt
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)
