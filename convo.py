from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random #Possibly implement seed here?

dimension = 0
modNum = 0
divisionNum = 0
strideNum = 0
learning_rate = 0
labelNum = 0
num_neurons = 0
random_seed = 0

def anti_one_hot(labels):
  unencoded = tf.argmax(labels, axis=1)
  return unencoded

def buildNetwork(features, labels, mode):
  tf.set_random_seed(random_seed)

  #Variables to change:
  drop_num = 0.4 #Drps % of items in training
  
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, dimension, dimension, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[divisionNum, divisionNum], strides=strideNum)
  
  reshapeNum1 = dimension/divisionNum #CHANGE??
  print(reshapeNum1)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[divisionNum, divisionNum], strides=strideNum)
  
  reshapeNum2 = reshapeNum1/divisionNum #CHANGE??
  print(reshapeNum2)

  # Dense Layer
  
  pool2_flat = tf.reshape(pool2, [-1, int(reshapeNum2 * reshapeNum2 * 64)]) #64 = num of filters in pool2
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu) #Num neurons here?
  dropout = tf.layers.dropout(
      inputs=dense, rate=drop_num, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10) #depends on label num

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = labels
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=anti_one_hot(labels), predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
