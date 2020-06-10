import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import np_utils
import time

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

    def train(self):
        N_HIDDEN = 2
        DROPOUT = 0.2
        NB_CLASSES = 2
        BATCH_SIZE = 32
        EPOCHS=10
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

        model.fit(self.trainTrain,self.trainTrainLabel,validation_split = 0.2,batch_size=30,epochs=EPOCHS)

  # def eval(model):
        pred1 = model.predict(self.trainValid)     
        pred2 = np.argmax(pred1,axis = 1)
        trueAnswer = np.argmax(self.trainValidLabel,axis = 1)
        numCorrect = (pred2 == trueAnswer).sum()
        precision = numCorrect / len(pred2)        
        print('Precision of the Model is ... :',precision)
        print(numCorrect)
        print(len(pred2))
        print(pred1)
        print(pred2)
        print(trueAnswer)
        
        model.save('saved_models/{}.h5'.format(time.ctime()))

def main():
    rangeIndex = np.random.randint(0,4000,400)
    _dogs = np.load("ResizedDogImg.npy")[rangeIndex]
    _cats = np.load("ResizedCatImg.npy")[rangeIndex]
    _dogsLabel = np.array([ 1 for i in range(len(_dogs))])
    _catsLabel = np.array([ 0 for i in range(len(_cats))])

    _dogsCats = np.concatenate([_dogs,_cats])
    _dogsCatsLabel =np.concatenate([_dogsLabel,_catsLabel])

    shuffleIndex = np.random.randint(0,_dogsCats.shape[0],_dogsCats.shape[0])
    _dogsCats = _dogsCats[shuffleIndex]
    _dogsCatsLabel = _dogsCatsLabel[shuffleIndex]

    trainTrain = _dogsCats[:600]
    trainValid = _dogsCats[600:]
    trainTrainLabel = _dogsCatsLabel[:600]
    trainValidLabel = _dogsCatsLabel[600:]
    
    print('traintrainlabel \n',trainTrainLabel)
    
    print('trainvalidlabel \n',trainValidLabel)

    classifyingModel = Classifier(trainTrain,trainTrainLabel,trainValid,trainValidLabel)
    model = classifyingModel.train()    #学習、評価

if __name__ == '__main__':
    main()








