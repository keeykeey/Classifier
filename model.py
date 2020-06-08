import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFOld
#import tensorflow as tf
#from tensorflow import keras
#from tensortlow.keras import layers
from keras.models import Sequencial
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import np_utils


ClassNames = ['cats','dogs']
#ClassLabels = [0,1]
num_classes = 2

class Classifier():
    '''
    trainTrain : numpy array for images
    trainTrainLabel : numpy array for each images' label
    trainValid : numpy array for images
    trainTrainLabel : numpy array for each images' label
    '''

    def __init__(self,trainTrain,trainTrainLabel,trainValid,trainValidLabel):
        self.trainTrain = trainTrain
        self.trainTrainLabel = keras.utils.to_categorical(trainTrainLabel,num_classes)
        self.trainValid = trainValid
        self.trainValidLabel = keras.utils.to_categorical(trainValidLabel,num_classes) 

    def train():
        N_HIDDEN = 2
        RESHAPED = allDatas.shape[1] * allDatas.shape[2]
        DROPOUT = 0.1
        NB_CLASSES = 2
        BATCH_SIZE = 32
        EPOCHS=3
        OPTIMIZER = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)

        model = Sequential()
        model.add(Conv2D(32,(3,3),padding='same',input_shape=self.trainTrain.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32,(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(DROPOUT))

        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64,(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(DROPOUT))

        model.add(Flatten())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                 optimizer=OPTIMIZER,
                 metrics = ['accuracy'])

        model.fit(self.trainTrain,self.trainTrainLabel)

        return model    

    def eval():
        pred = model.predict(self.trainValid)     
        pred = np.argmax(pred,axis = 1)
        trueAnswer = np.argmax(self.trainValidLabel,num_classes)
        numCorrect = (pred == trueAnswer).sum()
        precision = np.round(numCorrect / len(pred),2)        
        print('precision:',precision)

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










