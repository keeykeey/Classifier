import os
import numpy as np

CurrentDirectory = os.getcwd()
CatPath = CurrentDirectory + '/training_set/cats'
DogPath = CurrentDirectory + '/training_set/dogs'

CatFileList = os.listdir(PathOfCatImgs)
DogfileList = os.listdir(PathOfDogimgs)
CatFileList.remove('_DS_Store')
DogFileList.remove('_DS_Store')

def OriginalData():
    CatImgList = []
    DogImgList = []

    for i in range(len(CatFileList)):
        ArrayOfImg = cv2.imread(CatPath + '/' + CatFileList[i])
        CatImgList.append(ArrayOfImg)

    for i in range(len(DogFileList)):
        ArrayOfImg = cv2.imread(DogPath
        DogImgList.append(ArrayOfImg)

    OriginalSizedCatsImages = np.array(CatImgList)
    OriginalSisedDogsImages = np.array(DogImgList)

    np.save(CurrentDirectory,OriginalSizedCatsImages)
    np.save(CurrentDirectory,OriginalSizedDogsImages)

def resize(x):
    SumOfCatsImagesShape = np.array([0,0,0])
    SumOfDogsImagesShape = np.array([0,0,0])

    for i in range(len(OriginalSizedCatsImages)):
        SumOfCatsImagesShape += OriginalSizedCatsImages[i].shape
    for i in range(len(OriginalSizedCatsImages)):
        SumOfDogsImagesShape += OriginalSizedDogsImages[i].shape
    MeanOfCatsAndDogsImageShape = (SumOfCatsImagesShape+SumOfDogsImagesShape)/(OriginalSizedCatsImages.size + OriginalSizedDogsImages.size)
    MeanOfCatsAndDogsImagesShape = np.round(MeanOfCatsAndDogsImages.shape

    Length = MeanOfCatsAndDogsImagesShape[0]
    Height = MeanOfCatsAndDogsImagesShape[1]

    x_out = []
    for i in range(len(x)):
        img = cv2.resize(x[i],dsize = (Length,Height))
        x_out.append(img)
    
    return img









