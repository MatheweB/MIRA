from PIL import Image
import os
import sys
import tensorflow as tf
import math


def buildNetwork(training, testing, labelNum, attributes, labels):
    #Do networthy stuff
    pass

def subdivide(image,divisionNum):
    #Use method where we subdivide both sides into (divisionNum) pieces, and then
    #iterate over half and make a square, then go over half and make a square, etc...

    halfNum = (imageSize/divisionNum)/2 #describes how much we should move over each time
    #possibly a for loop for going over the top row, then moving down half... repeat...
    
    #img2 = img.crop((0, 0, 100, 100)) #crops a 100x100 square starting at top left (0,0)
    
    #Decides how many subdivisions there are

    #subNum is the size of the subdivision that we want (as an nxn square)

    #rounds subNum if it doesn't work
    
    

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

    #Now we have a list of image tensors??

    buildNetwork(trainingSet, testingSet, len(labels), attributes, labels)
    



if __name__ == "__main__":
    num_neurons = 1#int(sys.argv[1])
    learning_rate = 1#float(sys.argv[2]) # K-Value (an integer)
    training_runs = 1#int(sys.argv[3])
    percentage = 1#float(sys.argv[4])  # Percentage of data to be used for TRAINING
    divisonNum = #int(sys.argv[5])
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
                image = Image.open(photoDir +"/"+ imageName).convert('LA') #Greyscales and keeps alpha
                photoTypeList.append([image,photoType])
            else:
                continue
        images.append(photoTypeList)
                
    main(images,num_neurons,learning_rate,training_runs,percentage,divisionNum,seed)
     
    
    

    
    
