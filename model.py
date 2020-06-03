import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFOld
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequencial
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import np_utils


ClassNames = ['cats','dogs']
#ClassLabels = [0,1]
num_classes = 2
random_state = 42

class Classifier():
    '''
    trainTrain : numpy array for images
    trainTrainLabel : numpy array for each images' label
    trainValid : numpy array for images
    trainTrainLabel : numpy array for each images' label
    '''

    def __init__(self,trainTrain,trainTrainLabel,trainValid,trainValidLabel):
        self.trainTrain = trainTrain
        self.trainTrainLabel = trainTrainLabel
        self.trainValid = trainValid
        self.trainValidLabel = trainValidLabel 

    def train():
        model = Sequential()



        model.add(Conv2D(32,(3,3),padding='same',input_shape=self.TrainImages.shape[1:]))





def main():
    _dogs = np.load("ResizedDogImg.npy")
    _cats = np.load("ResizedCatImg.npy")
    _dogsLabels = np.array([ 1 for i in range(len(_dogs))])
    _catsLabels = np.array([ 0 for i in range(len(_cats))])

    _dogsCats = np.append(_dogs,_cats)
    _dogsCatsLabels =np.append(_dogsLabels,_catsLabels)

    Classifier = Classifier(ImageData,LabelData)
    Classifier.train()    #学習
    Classifier.eval()     #評価










