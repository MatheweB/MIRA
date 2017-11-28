from PIL import Image
import os
import sys
import tensorflow as tf
import math


def buildNetwork(training, testing, labelNum, attributes, labels):
    #Do networthy stuff
    pass

def subdivide(image,subNum):
    pass

    #Decides how many subdivisions there are

    #subNum is the size of the subdivision that we want (as an nxn square)

    #rounds subNum if it doesn't work
    
    

def preProcessImages(images):
    largestImage = getLargestImage(images)
    imageSize = largestImage.size #[x,y]
    transformImages(images, imageSize)
    

def getLargestImage(images):
    largestImage = None
    bestDim = 1
    
    for image in images:
        size = image.size
        x = size[0]
        y = size[1]
        diff = math.fabs(x-y)
        if diff/(x+y) < bestDim:
            bestDim = diff/(x+y)
            largestImage = image

    print(largestImage.size)
    exit(5)


def transformImages (images, size): #MAKE SURE TO HAVE DIMENSIONS BE A MULTIPLE OF 4
    pass
    
    
def main(images,num_neurons,learning_rate,training_runs,percentage,seed):
                        
    preProcessImages(images)
    
    for image in images:
        subdivide(image)

    #Now we have a list of image tensors??

    buildNetwork(trainingSet, testingSet, len(labels), attributes, labels)
    



if __name__ == "__main__":
    num_neurons = 1#int(sys.argv[1])
    learning_rate = 1#float(sys.argv[2]) # K-Value (an integer)
    training_runs = 1#int(sys.argv[3])
    percentage = 1#float(sys.argv[4])  # Percentage of data to be used for TRAINING
    seed = 1#float(sys.argv[5])

    #OPEN FILES HERE
    images = [] #where an image is [image,label]
    photoTypes = []
    CWD = os.getcwd() + "/photos"
    
    for photoType in os.listdir(CWD): #Adds all photo type folders
        photoTypes.append(photoType)
        
    
    for photoType in photoTypes:
        photoDir = CWD + "/" + photoType
        for imageName in os.listdir(photoDir):
            image = Image.open(imageName).convert('LA') #Greyscales and keeps alpha
            images.append([image,photoType])

    #Possibly make cases for seed not being a string, etc...
    main(images,num_neurons,learning_rate,training_runs,percentage,seed)
     
    
    

    
    
