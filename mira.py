from PIL import Image


def buildNetwork(training, testing, labelNum, attributes, labels):
    #Do networthy shit
    pass

def subdivide(image,subNum):

    #Decides how many subdivisions there are

    #subNum is the size of the subdivision that we want (as an nxn square)

    #rounds subNum if it doesn't work
    
    

def preProcessImages(images):
    largestImage = getLargestImage(images)
    imageSize = largestImage.size #[x,y]
    transformImages(images, imageSize)
    

def getLargestImage(images):
    pass


def transformImages (images, size): #MAKE SURE TO HAVE DIMENSIONS BE A MULTIPLE OF 4

    img = Image.open('image.png').convert('LA') #(A maintains Alpha Channel/Intensity)
    pass
    
    
def main():

    
    preProcessImages(images)
    
    for image in images:
        subdivide(image)

    #Now we have a list of image tensors??

    buildNetwork(trainingSet, testingSet, len(labels), attributes, labels)
    


if __name__ == "__main__":
    path = sys.argv[1]  # Usually just the filename in the same folder
    num_neurons = int(sys.argv[2])
    learning_rate = float(sys.argv[3]) # K-Value (an integer)
    training_runs = int(sys.argv[4])
    percentage = float(sys.argv[5])  # Percentage of data to be used for TRAINING
    seed = float(sys.argv[6])

    #OPEN FILES HERE
    images = [] #where an image is [image,label]
    
    for image in thingy:
        pass

    #Possibly make cases for seed not being a string, etc...
    main(images,num_neurons,learning_rate,training_runs,percentage,seed)
     
    
    

    
    
