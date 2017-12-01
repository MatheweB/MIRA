from PIL import Image
from pp import preProcessImages, Instance
import os
import sys
import tensorflow as tf
import math
import random

def buildNetwork(training, testing, numLabels, numSub, numPixels):

    #None = number of images
    #numSub = number of sub-images
    #numPixel = number of pixels in sub-image
    
    X = tf.placeholder(tf.float32, shape = [None, numSub, numPixel])
    
    Y = tf.placeholder(tf.float32, shape = [None, numLabels])

    #Layer 1
    #w_hidden_1 = tf.Variable(tf.truncated_normal[])


    #Layer2
    #w_hidden_2


    #Output Layer
    #w_hidden_3


    #None = number of images
    #numlabels = number of labels (10 for our original set)
    #y = tf.placeholder(tf.float32, shape = [None, numSub, numLabels])

    pass

    
def main(instances,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum):
    
    trainingSet, testingSet = preProcessImages(instances,(stepNum*divisionNum), divisionNum, stepNum) #MAKE SURE THESE ARE RIGHT

    buildNetwork(trainingSet, testingSet, len(instances), len(instances[0][0].subImages), len(instances[0][0].subImages[0]))
    

if __name__ == "__main__":
    num_neurons = 1#int(sys.argv[1])
    learning_rate = 1#float(sys.argv[2])
    training_runs = 1#int(sys.argv[3])
    percentage = 1#float(sys.argv[4])  #Percentage of data to be used for TRAINING
    divisionNum = 3#int(sys.argv[5]) #How many times we want to divide up the image into smaller squares
    stepNum = 2#int(sys.argv[6]) #Decides how much overlap we have per subdivision 
    seed = 1 #float(sys.argv[7])
    
    #Opens Images Here (Image files must be of type .jpg)
    instances = []
    photoTypes = [] #list of "labels"
    
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
                instance = Instance()
                instance.sLabel = photoType
                instance.image = image
                photoTypeList.append(instance)
                
        instances.append(photoTypeList)

    random.seed(seed)
                
    main(instances,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum)

    
    
