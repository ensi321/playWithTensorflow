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
  nn = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir='./log/training', params={}, config=tf.contrib.learn.RunConfig(save_checkpoints_secs=5))

  # train input_fn
  def get_train_inputs():
    x = tf.constant(training_set.images)
    y = tf.constant(training_set.labels)
    return x, y

  # test input_fn
  def get_test_inputs():
    x = tf.constant(test_set.images)
    y = tf.constant(test_set.labels)
    return x, y

  # Accuracy function for validation_monitor
  def custom_accuracy(predictions, labels):
    predictions = tf.argmax(predictions, 1)
    labels = tf.argmax(labels, 1)
    return tf.contrib.metrics.streaming_accuracy(predictions, labels)


  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      test_set.images,
      test_set.labels,
      every_n_steps=100,
      metrics={"accuracy":tf.contrib.learn.MetricSpec(
        metric_fn=custom_accuracy)})

  # This is experiment
  experiment = tf.contrib.learn.Experiment(nn, get_train_inputs, get_test_inputs, train_steps=1000, eval_steps=1, 
    eval_delay_secs=0, train_monitors=[validation_monitor])
  experiment.train()
  experiment.evaluate()


if __name__ == "__main__":

  # Dont worry about the FLAG and arguments
  tf.app.run(main=main, argv=[])