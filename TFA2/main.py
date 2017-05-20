from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from models.model import model_fn
import data_utils


data_dir = "./data"
from_vocab_size = 40000
to_vocab_size = 40000
max_train_data_size = 200000
# This is size for LSTM
hidden_size = 1024



# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  # data_set = [[] for _ in _buckets]
  source_set = []
  target_set = []

  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1

        source_ids = [int(x) for x in source.split()]
        source_ids.append(data_utils.EOS_ID)
        source_ids = source_ids + [0] * (25 - len(source_ids))
        # strip down to 25 words
        source_ids = source_ids[:25]

        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        target_ids = target_ids + [0] * (25 - len(target_ids))
        target_ids = target_ids[:25]

        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()


    	source_set.append(source_ids)
    	target_set.append(target_ids)

        source, target = source_file.readline(), target_file.readline()
  return source_set, target_set


def main(_):
	# This assume you prefectly executed the script, downloaded the dataset, and created vocabularies in data dir
	from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
	          data_dir, from_vocab_size, to_vocab_size)

	dev_source, dev_target = read_data(from_dev, to_dev)
	train_source, train_target = read_data(from_train, to_train, max_train_data_size)

	# Instantiate Estimator
	nn = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir='./log/training', params={'hidden_size': hidden_size}, config=tf.contrib.learn.RunConfig(save_checkpoints_secs=5))

	# train input_fn
	def get_train_inputs():
		x = np.array(train_source)
		print(x)
		print(x.shape)
		x = tf.convert_to_tensor(x)
		y = np.array(train_target)
		y = tf.convert_to_tensor(y)
		return x, y

	# test input_fn
	def get_test_inputs():
		x = np.array(dev_source)
		x = tf.convert_to_tensor(x)
		y = np.array(dev_target)
		y = tf.convert_to_tensor(y)
		return x, y

	# This is experiment
	experiment = tf.contrib.learn.Experiment(nn, get_train_inputs, get_test_inputs, train_steps=1000, eval_steps=1, 
	  eval_delay_secs=0, train_monitors=[])
	experiment.train()
	experiment.evaluate()


if __name__ == "__main__":
	tf.app.run(main=main, argv=[])
