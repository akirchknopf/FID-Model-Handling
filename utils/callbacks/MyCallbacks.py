import tensorflow.keras.callbacks as cb
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from .MyTelegramCallBack import MyTelegramCallback
from .MyTimeHistoryCallback import MyTimeHistoryCallback

class MyCallbacks:
    
    def __init__(self, tensorboardDir, checkPointDir, earlyStopping=True, earlyStoppingPatience=3, reduceLR=True, minLR=0.0001):    
        self.tensorboardDir = tensorboardDir
        self.checkPointDir = checkPointDir
        
        self.earlyStopping = earlyStopping
        self.earlyStoppingPatience = earlyStoppingPatience
        self.reduceLR = reduceLR
        self.minLR = minLR
        
    
    def myTensorboardCallback(self):
        return cb.TensorBoard(log_dir= self.tensorboardDir, write_graph=True, update_freq='batch', write_images=True)
    
    def myTelegramCallback(self):
        return MyTelegramCallback()
    
    def myModelCheckpoint(self):
        return ModelCheckpoint(filepath= self.checkPointDir, save_best_only=True, verbose=1)
    
    def myEarlyStoppingCallback(self):
        return EarlyStopping(monitor='val_loss', patience=self.earlyStoppingPatience, verbose=2, mode='min', restore_best_weights=True)
    
    def myReduceLROnPlateauCallback(self):
        return ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=self.minLR)   
        
    
    def createCheckpoints(self):        
        tbCB = self.myTensorboardCallback()
        tgCB = self.myTelegramCallback()
        mMCP = self.myModelCheckpoint()
        mESCB = self.myEarlyStoppingCallback()
        mLRPCB = self.myReduceLROnPlateauCallback()
        mTHCB  = MyTimeHistoryCallback()
        
        checkpoint_list = [tbCB, tgCB, mMCP, mTHCB]
        
        if self.earlyStopping:
            checkpoint_list.append(mESCB)
        
        if self.reduceLR:
            checkpoint_list.append(mLRPCB)

        return checkpoint_list
    
