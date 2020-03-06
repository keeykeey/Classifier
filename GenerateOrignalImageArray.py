import os
import numpy as np

CurrentDirectory = os.getcwd()
CatPath = CurrentDirectory + '/training_set/cats'
DogPath = CurrentDirectory + '/training_Set/dogs'

CatFileList = os.listdir(PathOfCatImgs)
DogfileList = os.listdir(PathOfDogimgs)
CatFileList.remove('_DS_Store')
DogFileList.remove('_DS_Store')

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

