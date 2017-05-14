from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from six.moves import urllib

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.examples.tutorials.mnist import input_data
from models.model import model_fn

tf.logging.set_verbosity(tf.logging.INFO)

data_dir = './data'

def main(_):
  # Load datasets
  mnist = input_data.read_data_sets(data_dir, one_hot=True)
  training_set = mnist.train
  test_set = mnist.test

  # Instantiate Estimator
  nn = tf.contrib.learn.Estimator(model_fn=model_fn, params={})

  def get_train_inputs():
    x = tf.constant(training_set.images)
    y = tf.constant(training_set.labels)
    return x, y

  def get_test_inputs():
    x = tf.constant(test_set.images)
    y = tf.constant(test_set.labels)
    return x, y

  # This is experiment
  experiment = tf.contrib.learn.Experiment(nn, get_train_inputs, get_test_inputs, train_steps=1000, eval_steps=1, eval_delay_secs=0)
  experiment.train()
  # experiment.evaluate()

  # Score accuracy
  predictions = experiment.estimator.predict(x=test_set.images, as_iterable=False)
  correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_set.labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    print(sess.run(accuracy))


if __name__ == "__main__":

  # Dont worry about the FLAG and arguments
  tf.app.run(main=main, argv=[])