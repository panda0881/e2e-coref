#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import original_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  with tf.Session() as session:
    model.restore(session)
    model.evaluate(session, official_stdout=True)
