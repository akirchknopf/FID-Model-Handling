import os
import pandas as pd
import numpy as np
from numpy import asarray
from tensorflow.keras.utils import Sequence

from ..datagenUtils import convertRowToDictionary

class DataSequenceTitleMeta(Sequence):
    'Generates data for Keras'
    def __init__(self, df, text_list_title, batch_size):
        self.df = df
        self.text_list_title = text_list_title
        self.batch_size = batch_size
        self.on_epoch_end()
        
        len_of_df = np.array(len(self.df))
        len_of_text_list_title = len(text_list_title)

        assert len_of_df == len_of_text_list_title, f"Len of df and text list should be the same, df: {len_of_df}, text: {len_of_text_list}"


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Slice df
        df_batch = self.df.take(batch_indices)

        # Generate data
        title_x, meta_x, global_y = self.__data_generation(df_batch, batch_indices)

        return [[title_x, meta_x]], global_y

    def on_epoch_end(self):
        'Updates indexes after each epoch -> Resetting them' 
        self.indices = np.arange(len(self.df))
        
    def __data_generation(self, df_slice, batch_indices):
        batch_x = []
        batch_y = []

        for index, row in enumerate(df_slice.itertuples(), 1):
            rowFine = convertRowToDictionary(row, df_slice.columns, True)
            batch_x.append([ rowFine['upvote_ratio'], rowFine['num_comments']])
            batch_y += [rowFine['2_way_label']]
 
        global_y = np.array(batch_y).astype(np.float32)
        
        meta_x = np.array(batch_x)
        title_x = np.array([self.text_list_title[i] for i in batch_indices]) 

        return (title_x, meta_x, global_y)
