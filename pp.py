from PIL import Image
import numpy as np
import math
import random

class Instance:
    def __init__(self):
        self.sLabel = ""
        self.vLabel = []
        self.image = None

def numPyIfy(instances):
    for instanceType in instances:
        for instance in instanceType:
            instance.image = np.asarray(instance.image, dtype=np.float32)

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
        random.shuffle(n)
        splitIdx = int(len(n) * tperc)
        trainSplit.extend(n[:splitIdx])
        testSplit.extend(n[splitIdx:])
    random.shuffle(trainSplit)
    random.shuffle(testSplit)
    return trainSplit, testSplit
        
def preProcessImages(instances,modNum,tperc):
    imageSize = getLargestImage(instances,modNum) #[x,y]
    imageSideSize = imageSize[0]
    
    transformImages(instances, imageSize)

    vectorizeLabels(instances)
    numPyIfy(instances)

    trainingSet, testingSet = splitSets(instances, tperc)
    return trainingSet, testingSet, int(imageSideSize)

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

    
    
