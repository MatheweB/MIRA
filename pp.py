from PIL import Image
import math
import random

class Instance:
    def __init__(self):
        self.sLabel = ""
        self.vLabel = []
        self.image = None
        self.subImages = []

def vectorizeLabels(instances):
    numLabels = len(instances)
    for n in range(0, numLabels):
        for i in instances[n]:
            vl = [0]*numLabels
            vl[n] = 1
            i.vLabel = vl

def splitSets(instances, tperc):
    trainSplit = []
    testSplit = []
    for n in instances:
        random.shuffle(instances[n])
        splitIdx = int(len(instances[n]) * tperc)
        trainSplit.extend(instances[n][:splitIdx])
        testSplit.extend(instances[n][splitIdx:])
    random.shuffle(trainSplit)
    random.shuffle(testSplit)
    return trainSplit, testSplit
        

def subdivide(instances,divisionNum,stepNum,imageSideSize): #squares go DOWN, not up
    
    segNum = (imageSideSize/divisionNum)/stepNum #describes how much we should move over each time

    divisionList = [] #List of steps for our algorithm

    squareSize = (imageSideSize/divisionNum)
    totalSize = (imageSideSize) - squareSize
    currentSize = 0
    divisionList.append(currentSize)
    while currentSize < totalSize:
        currentSize = currentSize + segNum
        divisionList.append(currentSize)

    for instanceType in instances:
        for instance in instanceType:
            subImageList = []
            for xStep in divisionList:
                for yStep in divisionList:
                    subImage = instance.image.crop((xStep,yStep,xStep+squareSize,yStep+squareSize))
                    subImageList.append(list(subImage.getdata()))
            instance.subImages = subImageList #changes all images into an array of sub-images    
    

def preProcessImages(instances,modNum, divisionNum, stepNum, tperc):
    imageSize = getLargestImage(instances,modNum) #[x,y]
    imageSideSize = imageSize[0]
    
    transformImages(instances, imageSize)
    subdivide(instances,divisionNum,stepNum,imageSideSize)

    exit(5)
    
    vectorizeLabels(instances) #QUINN TO WRITE THESE
    return splitSets(instances, tperc)

def getLargestImage(instances,modNum): #gets the largest, most square image
    largestImage = None
    bestDim = 1
    
    for instanceType in instances:
        for instance in instanceType:
            size = instance.image.size
            x = size[0]
            y = size[1]
            diff = math.fabs(x-y)
            if diff == 0: #accounts for a perfect square
                diff = 1
            if diff/(x+y) < bestDim:
                bestDim = diff/(x+y)
                largestImage = instance.image

    largestSize = largestImage.size
    x = largestSize[0]
    y = largestSize[1]
    if x != y and y > x:
        x = y
            
    while x%modNum != 0: #ensures the dimensions are easy to deal with in subdividing (have factors that work) 
        x += 1

    return [x,x]


def transformImages (instances, size): 
    for instanceType in instances:
        for instance in instanceType:
            instance.image = instance.image.resize(size, Image.ANTIALIAS)

    
    
