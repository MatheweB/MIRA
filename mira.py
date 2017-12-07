from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from PIL import Image
from pp import preProcessImages, Instance
import convo

import os
import sys
import numpy as np
import math
import random

tf.logging.set_verbosity(tf.logging.INFO)

def main(instances,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum,specificDimension,labelNum,seed):

    convo.modNum = int(stepNum*divisionNum)
    convo.divisionNum = int(divisionNum)
    convo.strideNum = int(stepNum)
    convo.learning_rate = float(learning_rate)
    convo.labelNum = labelNum
    convo.num_neurons = num_neurons #Doesn't work
    convo.random_seed = seed
    
    trainingSet, testingSet, convo.dimension = preProcessImages(instances,(stepNum*divisionNum),percentage,specificDimension)

    #BUILDS CLASSIFIER
    photo_classifier = tf.estimator.Estimator(
        model_fn=convo.buildNetwork, model_dir=str(os.getcwd())+'/model')

    tensors_to_log = {"probabilities": "softmax_tensor"} #Uses softmax
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50) #probabilities logged every 50 iterations

    #instantiating images
    train_data = []
    eval_data = []
    for inst in trainingSet:
        train_data.append(inst.image)
    for inst in testingSet:
        eval_data.append(inst.image)

    train_data = np.asarray(train_data, dtype=np.float32)
    eval_data = np.asarray(eval_data, dtype=np.float32)

    #instrantiating labels
    train_labels = []
    eval_labels = []

    
    for inst in trainingSet:
        train_labels.append(inst.vLabel)
    for inst in testingSet:
        eval_labels.append(inst.vLabel)
    
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_labels = np.asarray(eval_labels, dtype=np.int32)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    photo_classifier.train(
        input_fn=train_input_fn,
        steps=training_runs,
        hooks=[logging_hook])
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = photo_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
        
    

if __name__ == "__main__":
    num_neurons = 1000#int(sys.argv[1])
    learning_rate = 0.01#float(sys.argv[2])
    training_runs = 1000#int(sys.argv[3])
    percentage = 0.70#float(sys.argv[4])  #Percentage of data to be used for TRAINING
    divisionNum = 2#int(sys.argv[5]) #How many times we want to divide up the image into smaller squares
    stepNum = 2#int(sys.argv[6]) #Decides how much overlap we have per subdivision 
    seed = 123 #float(sys.argv[7])
    specificDimension = 32#float(sys.argv[8])
    
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
                image = Image.open(photoDir +"/"+ imageName).convert('RGB')
                storedImg = image.copy()
                image.close()
                instance = Instance()
                instance.sLabel = photoType
                instance.image = storedImg
                photoTypeList.append(instance)
                
        instances.append(photoTypeList)

    random.seed(seed)
    tf.set_random_seed(seed) #Sets seed for tensorflow
    
    main(instances,num_neurons,learning_rate,training_runs,percentage,divisionNum,stepNum,specificDimension, len(photoTypes),seed)
    

    
    
