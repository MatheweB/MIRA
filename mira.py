from PIL import Image
import os
import sys
import tensorflow as tf
import math


#images = [[phototype1[image[imageFile,imageName]]

def buildNetwork(training, testing, labelNum, attributes, labels):
    #Do networthy stuff
    pass

def subdivide(images,divisionNum): #squares go DOWN, not up
    imageSideSize = images[0][0][0].size[0] #gets the length of one side of an image
    stepNum = 2 #Decides how far over we move per subdivision (keep this as 2 or some multiple of 2)

    segNum = (imageSideSize/divisionNum)/stepNum #describes how much we should move over each time

    divisionList = [] #List of steps for our algorithm

    squareSize = (imageSideSize/divisionNum)
    totalSize = (imageSideSize) - squareSize
    currentSize = 0
    divisionList.append(currentSize)
    while currentSize < totalSize:
        currentSize = currentSize + segNum
        divisionList.append(currentSize)
    
    for imageType in images:
        for image in imageType:
            subImageList = []
            for xStep in divisionList:
                for yStep in divisionList:
                    subImage = image[0].crop((xStep,yStep,squareSize,squareSize)) #FIGURE THIS OUT
                    subImage.show() #Remove when done
                    subImageList.append(list(subImage.getdata()))
            image[0] = subImageList #changes all images into an array of sub-images    
    

def preProcessImages(images):
    imageSize = getLargestImage(images) #[x,y]
    transformImages(images, imageSize)

def getLargestImage(images): #gets the largest, most square image
    largestImage = None
    bestDim = 1
    
    for imageType in images:
        for image in imageType:
            size = image[0].size
            x = size[0]
            y = size[1]
            diff = math.fabs(x-y)
            if diff == 0: #accounts for a perfect square
                diff = 1
            if diff/(x+y) < bestDim:
                bestDim = diff/(x+y)
                largestImage = image[0]

    largestSize = largestImage.size
    x = largestSize[0]
    y = largestSize[1]
    while x != y: #ensures the size is a square
        if x < y:
            x += 1
        if y < x:
            y += 1
            
    while x%4 != 0: #ensures the dimensions are easy to deal with in subdividing
        x += 1
        y += 1

    return [x,y]


def transformImages (images, size): 
    for imageType in images:
        for image in imageType:
            image[0] = image[0].resize(size, Image.ANTIALIAS)
    
    
def main(images,num_neurons,learning_rate,training_runs,percentage,divisionNum,seed):
                       
    preProcessImages(images)
    subdivide(images,divisionNum)

    print(len(images[0][0][0][0])) #imagetype,image,imagefile,1st sub-image, 1st pixel value
    print(len(images[0][0][0][1]))

    exit(5)
    buildNetwork(trainingSet, testingSet, len(labels), attributes, labels)
    



if __name__ == "__main__":
    num_neurons = 1#int(sys.argv[1])
    learning_rate = 1#float(sys.argv[2]) # K-Value (an integer)
    training_runs = 1#int(sys.argv[3])
    percentage = 1#float(sys.argv[4])  # Percentage of data to be used for TRAINING
    divisionNum = 2#int(sys.argv[5]) #Ideally some multiple of 2 for convenience
    seed = 1 #float(sys.argv[6])

    
    #Opens Images Here (Image files must be of type .jpg)
    images = [] #where an image is [image,label]
    photoTypes = []
    CWD = os.getcwd() + "/photos"
    
    for photoType in os.listdir(CWD): #Adds all photo type folders
        if "." in photoType:
            continue
        else:
            photoTypes.append(photoType)
    
    for photoType in photoTypes:
        photoTypeList = [] #separates all images of a specific phototype for easy train/test distribution
        photoDir = CWD + "/" + photoType
        for imageName in os.listdir(photoDir):
            if ".jpg" in imageName:
                image = Image.open(photoDir +"/"+ imageName).convert('L') #Greyscales
                photoTypeList.append([image,photoType])
            else:
                continue
        images.append(photoTypeList)
                
    main(images,num_neurons,learning_rate,training_runs,percentage,divisionNum,seed)
     
    
    

    
    
