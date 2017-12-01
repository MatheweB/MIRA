from PIL import Image
import os
import sys
import tensorflow as tf
import math

def buildNetwork(training, testing, labelNum, attributes, labels):

    #numAttributes = size of a sub-picture list (each picture)

    #None = numOfImages*numOfSubImage(of 1 pic)
    #x = x-dimension of sub-image
    #y = y-dimension of sub-image
    x = tensorflow.placeholder(tensorflow.float32, shape = [None, x, y])

    #HIDDEN LAYER
    w_hidden = tensorflow.Variable(tensorflow.truncated_normal([numAttributes, NUM_NEURONS],
                                               stddev = 0.01, seed = seedy))
    
    b_hidden = tensorflow.Variable(tensorflow.constant(0.1, shape = [NUM_NEURONS]))

    net_hidden = tensorflow.matmul(x, w_hidden) + b_hidden

    out_hidden = tensorflow.sigmoid(net_hidden)

    #OUTPUT LAYER
    w_output = tensorflow.Variable(tensorflow.truncated_normal([NUM_NEURONS, numLabels], stddev=0.01,
                                                               seed = seedy))

    b_output = tensorflow.Variable(tensorflow.constant(0.1, shape = [numLabels]))

    net_output = tensorflow.matmul(out_hidden, w_output) + b_output

    if numLabels != 1:
        predict = tensorflow.nn.softmax(net_output)
    else:
        predict = tensorflow.sigmoid(net_output)

    #True labels
    y = tensorflow.placeholder(tensorflow.float32, shape = [None, numLabels])

    if numLabels != 1:
        cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=y,logits=net_output))
    else:
        cost = tensorflow.reduce_sum(0.5*(y-predict)*(y-predict))
        
    trainer = tensorflow.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    sess = tensorflow.Session()
    init = tensorflow.global_variables_initializer().run(session = sess)

    #Training:
    runs = 0
    
    while(runs < training_runs):
        runs += 1
        _,p = sess.run([trainer,predict], feed_dict = {x:training[0],y:training[1]})
        
    predictionList = sess.run(predict, feed_dict = {x:test[0]})
    return predictionList


def subdivide(images,divisionNum,stepNum,imageSideSize): #squares go DOWN, not up
    
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
    return imageSize[0]

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
    
    imageSideSize = preProcessImages(images,(stepNum*divisionNum))
    subdivide(images,divisionNum,stepNum,imageSideSize)

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
     
    
    

    
    
