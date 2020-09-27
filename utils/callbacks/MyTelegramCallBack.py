import tensorflow as tf
from ..telegramUtils.telegram_bot import telegram_send_message
from datetime import datetime
from time import strftime

class MyTelegramCallback(tf.keras.callbacks.Callback):
  
    
#   def on_train_batch_begin(self, batch, logs=None):
#     print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

#   def on_train_batch_end(self, batch, logs=None):
#     print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

#   def on_test_batch_begin(self, batch, logs=None):
#     print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

#   def on_test_batch_end(self, batch, logs=None):
#     print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    def on_epoch_begin(self, epoch, logs=None):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        message = 'Starting epoch: ' + str(epoch) + ' @ ' + str(current_time)
        telegram_send_message(message)   
    
    def on_epoch_end(self, epoch, logs={}):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        message = 'Finished  epoch: ' + str(epoch) + ' @ ' + str(current_time) + " with " + str(logs)  
        telegram_send_message(message) 
        
        
