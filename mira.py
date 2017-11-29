from PIL import Image
import os
import sys
import tensorflow as tf
import math

def buildNetwork(training, testing, labelNum, attributes, labels):
    pass

def subdivide(images,divisionNum,stepNum): #squares go DOWN, not up
    imageSideSize = images[0][0][0].size[0] #gets the length of one side of an image

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
                    subImage = image[0].crop((xStep,yStep,xStep+squareSize,yStep+squareSize))
                    subImageList.append(list(subImage.getdata()))
            image[0] = subImageList #changes all images into an array of sub-images    
    

def preProcessImages(images,modNum):
    imageSize = getLargestImage(images,modNum) #[x,y]
    transformImages(images, imageSize)

def getLargestImage(images,modNum): #gets the largest, most square image
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
    if x != y:
        if x > y:
            y = x
        if y > x:
            x = y
    
            
    while x%modNum != 0: #ensures the dimensions are easy to deal with in subdividing (have factors that work) 
        x += 1

    return [x,x]


def transformImages (images, size): 
    for imageType in images:
        for image in imageType:
            image[0] = image[0].resize(size, Image.ANTIALIAS)
    
    
def main(images,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum,seed):
    
    preProcessImages(images,(stepNum*divisionNum))
    subdivide(images,divisionNum,stepNum)

    print(len(images[0][0][0][0])) #imagetype,image,imagefile,1st sub-image, 1st pixel value
    print(len(images[0][0][0][1]))

    buildNetwork(trainingSet, testingSet, len(labels), attributes, labels)
    



if __name__ == "__main__":
    num_neurons = 1#int(sys.argv[1])
    learning_rate = 1#float(sys.argv[2])
    training_runs = 1#int(sys.argv[3])
    percentage = 1#float(sys.argv[4])  #Percentage of data to be used for TRAINING
    divisionNum = 3#int(sys.argv[5]) #How many times we want to divide up the image into smaller squares
    stepNum = 2#int(sys.argv[6]) #Decides how much overlap we have per subdivision 
    seed = 1 #float(sys.argv[7])

    
    #Opens Images Here (Image files must be of type .jpg)
    images = [] #where an image is [image,label]
    photoTypes = []
    CWD = os.getcwd() + "/photos"
    
    for photoType in os.listdir(CWD): #Adds all photo type folders
        if "." not in photoType:
            photoTypes.append(photoType)
            
    for photoType in photoTypes:
        photoTypeList = [] #separates all images of a specific phototype for easy train/test distribution
        photoDir = CWD + "/" + photoType
        for imageName in os.listdir(photoDir):
            if ".jpg" in imageName:
                image = Image.open(photoDir +"/"+ imageName).convert('L') #Greyscales
                photoTypeList.append([image,photoType])
                
        images.append(photoTypeList)
                
    main(images,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum,seed)
     
    
    

    
    
