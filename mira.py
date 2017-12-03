from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
from pp import preProcessImages, Instance
from convo import buildNetwork
import os
import sys
import tensorflow as tf
import numpy as np
import math
import random

tf.logging.set_verbosity(tf.logging.INFO)

    
def main(instances,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum):
   # tf.app.run()
    
    trainingSet, testingSet = preProcessImages(instances,(stepNum*divisionNum),percentage)

    #BUILDS CLASSIFIER
    mnist_classifier = tf.estimator.Estimator(
        model_fn=buildNetwork, model_dir=str(os.getcwd())+'/model')

    tensors_to_log = {"probabilities": "softmax_tensor"} #Uses softmax
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50) #probabilities logged every 50 iterations

    #instantiating images
    train_data = []
    eval_data = []
    for inst in trainingSet:
        train_data.append(inst.image)
    for inst in testingSet:
        eval_data.append(inst.image)

    #instrantiating labels
    train_labels = []
    eval_labels = []
    for inst in trainingSet:
        train_labels.append(inst.vLabel)
    for inst in testingSet:
        eval_labels.append(inst.vLabel)
    
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_labels = np.asarray(train_labels, dtype=np.int32)

    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    photo_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
        
    

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
                image = Image.open(photoDir +"/"+ imageName)
                instance = Instance()
                instance.sLabel = photoType
                instance.image = image
                photoTypeList.append(instance)
                
        instances.append(photoTypeList)

    random.seed(seed)
    
    main(instances,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum)
    

    
    
