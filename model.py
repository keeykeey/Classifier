import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequencial
from keras.layers import Convolution,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import np_utils


ClassNames = ['cats','dogs']
ClassLabels = [0,1]
random_state = [i for i in range(50)]

class ImageClassifier(ImageData,LabelData):
    '''
    ImageData : Pandas.DataFrame : Data for training the model 
    LabelData : Pandas.DataFrame : Data for checking the model's accuracy
    ''' 
    TrainImages,TrainLabels,TestImages,TestLabels = train_test_split(ImageData,LabelData)

    def __init__(self,ImageData,LabelData,randome_state = 0):
        TrainImages,TrainLabels,TestImages,TestLabels = train_test_split(ImageData,LabelData,random_state = random_state))
        self.ImageData = ImageData
        self.LabelData = LabelData

    def split(self,split_size = 0.7): 
        self.



def main():
    _dogs = np.load("ResizedDogImg.npy")
    _cats = np.load("ResizedCatImg.npy")
    _dogsLabels = np.array([ 1 for i in range(len(_dogs))])
    _catsLabels = np.array([ 0 for i in range(len(_cats))])

    _dogscats = np.append(_dogs,_cats)
    _dogscatsLabels =np.append(_dogsLabels,_catsLabels)     

    Classifier = ImageClassifier(ImageData,LabelData)    
    Classifier.train()    #学習
    Classifier.eval()     #評価











    
   
