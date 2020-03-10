import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

ClassNames = ['cats','dogs']

class ImageClassifier(ImageData,LabelData):
    '''
    ImageData : Pandas.DataFrame : Data for training the model 
    LabelData : Pandas.DataFrame : Data for checking the model's accuracy
    ''' 
    TrainImages,TrainLabels,TestImages,TestLabels = train_test_split(ImageData,LabelData)

    def __init__(self,ImageData,LabelData):
        self.ImageData = ImageData
        self.LabelData = LabelData

    def split(self,split_size = 0.7): 
        self.


