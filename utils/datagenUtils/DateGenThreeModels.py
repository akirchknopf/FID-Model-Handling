import os
import pandas as pd
import numpy as np
from numpy import asarray
from tensorflow.keras.preprocessing.image import load_img, img_to_array
class DataGenThreeModels():
    
    def __init__(self, df, pathToImages, text_list, batch_size, image_size, meansOfDataset, stdsOfDataset):
        self.df = df
        self.pathToImages = pathToImages
        self.text_list = text_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.meansOfDataset = meansOfDataset
        self.stdsOfDataset = stdsOfDataset
        
    def createData(self):
        return self.dataGenThreeModels()

    def get_image(self, path):
        return load_img(path, target_size=[self.image_size[0], self.image_size[1]])

    def get_image_label_from_dataframe_slice(self, df_slice):

        batch_x = []
        batch_y = []

        for index, row in enumerate(df_slice.itertuples(), 1):
            path = os.path.join(self.pathToImages, row.fileName)
            img = img_to_array(self.get_image(path))
            pixels = asarray(img)
            pixels = pixels.astype('float32')            
            pixels = (pixels - self.meansOfDataset) / self.stdsOfDataset # Applying normalization
            pixels /= 255.0
            label = row.label
            batch_x += [pixels]
            batch_y += [label]

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y).astype(np.float32)

        return (batch_x, batch_y)


    def dataGenThreeModels(self):
        while True:

            len_of_df = np.array(len(self.df))
            len_of_text_list = len(self.text_list)

            assert len_of_df == len_of_text_list , f"Len of df and text list should be the same, df: {len_of_df}, text: {len_of_text_list}"

            batch_indices  = np.random.choice(a = len_of_df, size = self.batch_size)
            df_batch = self.df.take(batch_indices)
            batch_x, batch_y = self.get_image_label_from_dataframe_slice(df_batch)    

            text_x = np.array([self.text_list[i] for i in batch_indices])  

            yield [[batch_x], [batch_x], [text_x]], batch_y