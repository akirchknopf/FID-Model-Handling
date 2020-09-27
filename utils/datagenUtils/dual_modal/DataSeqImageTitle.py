import os
import pandas as pd
import numpy as np
from numpy import asarray
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from ..datagenUtils import convertRowToDictionary

class DataSequenceImageTitle(Sequence):
    'Generates data for Keras'
    def __init__(self, df, pathToImages, text_list, batch_size, image_size, meansOfDataset):
        self.df = df
        self.pathToImages = pathToImages
        self.text_list = text_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.meansOfDataset = meansOfDataset
        self.on_epoch_end()
        
        len_of_df = np.array(len(self.df))
        len_of_text_list = len(text_list)

        assert len_of_df == len_of_text_list , f"Len of df and text list should be the same, df: {len_of_df}, text: {len_of_text_list}"


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
        image_x, text_x, global_y = self.__data_generation(df_batch, batch_indices)

        return [[image_x, text_x]], global_y

    def on_epoch_end(self):
        'Updates indexes after each epoch -> Resetting them' 
        self.indices = np.arange(len(self.df))
        
    def get_image(self, path, img_size):
        return load_img(path, target_size=[img_size[0], img_size[1]])

    def __data_generation(self, df_slice, batch_indices):
        batch_x = []
        batch_y = []

        for index, row in enumerate(df_slice.itertuples(), 1):
            rowFine = convertRowToDictionary(row, df_slice.columns, True)
            path = os.path.join(self.pathToImages, row.imagePath)
            img = img_to_array(self.get_image(path, self.image_size))
            pixels = asarray(img)
            pixels = pixels.astype('float32')
            pixels = pixels - self.meansOfDataset
            pixels /= 255.0
            batch_x += [pixels]
            batch_y += [rowFine['2_way_label']]

        image_x = np.array(batch_x)
        global_y = np.array(batch_y).astype(np.float32)
        
        text_x = np.array([self.text_list[i] for i in batch_indices]) 

        return (image_x, text_x, global_y)
