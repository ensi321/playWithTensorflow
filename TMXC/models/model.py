from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import tensorflow as tf

def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  first_layer = tf.contrib.layers.linear(features, 10)

  # This layer is for training only, only use part of the first_layer
  output_layer = tf.nn.dropout(first_layer, params['keep_prob'])

  # When predicting, use all units of first layer
  predictions = first_layer


  # Calculate loss 
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output_layer))


  train_op = tf.contrib.layers.optimize_loss(
      loss=cross_entropy,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.5,
      optimizer="SGD")

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions,
      loss=cross_entropy,
      train_op=train_op)