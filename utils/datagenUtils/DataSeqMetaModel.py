#backup original dataseq

import os
import pandas as pd
import numpy as np
from numpy import asarray
from tensorflow.keras.utils import Sequence

class DataSequenceMetaModel(Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size
        self.on_epoch_end()
        
        len_of_df = np.array(len(self.df))


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
        meta_x, global_y = self.__data_generation(df_batch, batch_indices)

        return [meta_x], global_y

    def on_epoch_end(self):
        'Updates indexes after each epoch -> Resetting them' 
        self.indices = np.arange(len(self.df))
        
    def get_image(self, path, img_size):
        return load_img(path, target_size=[img_size[0], img_size[1]])

    def __data_generation(self, df_slice, batch_indices):
        batch_x = []
        batch_y = []

        for index, rowRaw in enumerate(df_slice.itertuples(), 1):
            row = convertRowToDictionary(rowRaw, df_slice.columns, True)
            batch_x.append([ row['author_enc'], row['score'], row['hasNanScore'], row['upvote_ratio'], row['hasNanUpvote'] , row['num_comments']])
#             batch_x.append([row['author_enc'], row['score'], row['hasNanScore'], row['upvote_ratio'], row['hasNanUpvote'], row['num_comments']])
            batch_y.append([row['2_way_label']])
        meta_x = np.array(batch_x)
        global_y = np.array(batch_y).astype(np.float32)
        
        

        return (meta_x, global_y)
