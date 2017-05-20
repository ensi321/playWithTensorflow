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

  # We have 2 sets of weights/bias. One for encoder, one for decoder
  encoder = tf.contrib.rnn.BasicLSTMCell(params['hidden_size'])
  decoder = tf.contrib.rnn.BasicLSTMCell(params['hidden_size'])
  state = (tf.zeros([1,params['hidden_size']]),)*2
  print('Features:')
  print(features)
  print('Targets:')
  print(targets)

  print('Encoding:')
  # Encode
  for word in features:
      output, state = encoder(word, state)
      print(output)
      print(state)

  print('Decoding:')
  # Decode
  predictions = []
  for i in range(25):
      output, state = decoder(output, state)
      print(output)
      print(state)
      predictions += output
      if output == 2:
          break

  # Pad the prediction til 25
  predictions = predictions + [0] * (25 - len(predictions))
  targets = targets + [0] * (25 - len(targets))



  # Calculate loss 
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=predictions))


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