
def preparation():

    import os
    import numpy as np

    CurrentDirectory = os.getcwd()
    PathOfCatImgs = CurrentDirectory + '/training_set/cats'
    PathOfDogImgs = CurrentDirectory + '/training_set/dogs'

    CatFileList = os.listdir(PathOfCatImgs)
    DogFileList = os.listdir(PathOfDogImgs)
    CatFileList.remove('_DS_Store')
    DogFileList.remove('_DS_Store')

    return PathOfCatImgs,PathOfDogImgs,CatFileList,DogFileList

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
    for i in range(len(OriginalSizedSogsImages)):
        SumOfDogsImagesShape += OriginalSizedDogsImages[i].shape
    MeanOfCatsAndDogsImagesShape = (SumOfCatsImagesShape+SumOfDogsImagesShape)/(OriginalSizedCatsImages.size + OriginalSizedDogsImages.size)
    MeanOfCatsAndDogsImagesShape = np.round(MeanOfCatsAndDogsImages.shape)

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

    returb CatResizedImgs,DogResizedImgs 

def regularize(ResizedNumpyNdarray):
    RegularizedNumpyNdarray = np.round(ResizedNumpyNdarray.astype('float32')/255,decimals = 4)
    return RegularizedNumpyNdarray

def main():
    preparation()
    GenerateAndSaveOriginalData()
    ModifiedSizeCatsImages = resize(OriginalSizedCatsImages)
    ModigiedSizeDogsImages = resize(ModifiedSizedDogsImages)
    RegularSizedCatsImages = regularize(ModifiedSizedCatsImages)
    RegularSizedDogsImages = regularize(ModifiedSizedDogsImages)

if __name__ == '__main__':
    main() 




