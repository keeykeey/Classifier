import os
import numpy as np
import cv2

CurrentDirectory = os.getcwd()

def preparation():

    PathOfCatsImgs = CurrentDirectory + '/training_set/cats'
    PathOfDogsImgs = CurrentDirectory + '/training_set/dogs'

    CatsFileList = os.listdir(PathOfCatsImgs)
    DogsFileList = os.listdir(PathOfDogsImgs)
    CatsFileList.remove('_DS_Store')
    DogsFileList.remove('_DS_Store')

    return PathOfCatsImgs,PathOfDogsImgs,CatsFileList,DogsFileList

def GenerateAndSaveOriginalData(PathOfCatImgs,PathOfDogImgs,CatFileList,DogFileList):
    CatImgList = []
    DogImgList = []

    for i in range(len(CatFileList)):
        ArrayOfImg = cv2.imread(PathOfCatImgs + '/' + CatFileList[i])
        CatImgList.append(ArrayOfImg)

    for i in range(len(DogFileList)):
        ArrayOfImg = cv2.imread(PathOfDogImgs + '/' + DogFileList[i])
        DogImgList.append(ArrayOfImg)

    OriginalSizedCatsImages = np.array(CatImgList)
    OriginalSizedDogsImages = np.array(DogImgList)

    np.save(CurrentDirectory,OriginalSizedCatsImages)
    np.save(CurrentDirectory,OriginalSizedDogsImages)

    return OriginalSizedCatsImages,OriginalSizedDogsImages

def resize(OriginalSizedCatsImages,OriginalSizedDogsImages):
    SumOfCatsImagesShape = np.array([0,0,0])
    SumOfDogsImagesShape = np.array([0,0,0])

    for i in range(len(OriginalSizedCatsImages)):
        SumOfCatsImagesShape += OriginalSizedCatsImages[i].shape
    for i in range(len(OriginalSizedDogsImages)):
        SumOfDogsImagesShape += OriginalSizedDogsImages[i].shape
    MeanOfCatsAndDogsImagesShape = (SumOfCatsImagesShape+SumOfDogsImagesShape)/(OriginalSizedCatsImages.size + OriginalSizedDogsImages.size)
    MeanOfCatsAndDogsImagesShape = np.round(MeanOfCatsAndDogsImagesShape)

    Length = MeanOfCatsAndDogsImagesShape[0]
    Height = MeanOfCatsAndDogsImagesShape[1]

    CatResizedImgs = []
    for i in range(len(OriginalSizedCatsImages)):
        img = cv2.resize(OriginalSizedCatsImages[i],dsize = (Length,Height))
        CatResizedImgs.append(img)
    
    DogResizedImgs = []
    for i in range(len(OriginalSizedDogsImages)):
        img = cv2.resize(OriginalSizedDogsImages[i],dsize = (Length,Height))
        DogResizedImgs.append(img)

    np.save(CatResizedImgs)
    np.save(DogResizedImgs)

#    return ResizedCatsImgs,ResizedDogsImgs 

def regularize(ResizedNumpyNdarray):
    RegularizedNumpyNdarray = np.round(ResizedNumpyNdarray.astype('float32')/255,decimals = 4)
    return RegularizedNumpyNdarray

def main():
    PathOfCatsImgs,PathOfDogsImgs,CatsFileList,DogsFileList =  preparation()
    OriginalSizedCatsImages,OriginalSizedDogsImages = GenerateAndSaveOriginalData(PathOfCatsImgs,PathOfDogsImgs,CatsFileList,DogsFileList)
    ResizedCatsImgs,ResizedDogsImgs = resize(OriginalSizedCatsImages,OriginalSizedDogsImages)
    regularize(ResizedCatsImages)
    regularize(ResizedDogsImages)

#    RegularSizedCatsImages = regularize(ResizedCatsImages)
#    RegularSizedDogsImages = regularize(ResizedDogsImages)

if __name__ == '__main__':
    main() 




