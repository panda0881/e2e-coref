#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import original_model as cm
import util
import ujson as json

with open('freq_spans.json', 'r') as f:
  frequency_entity_list = json.load(f)
  new_list = list()
  for tmp_s in frequency_entity_list:
    new_list.append(' '.join(tmp_s))
  new_list = set(new_list)

test_data = list()
with open('conll_data/test.pronoun.jsonlines', 'r') as f:
  for line in f:
    test_data.append(json.loads(line))

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  with tf.Session() as session:
    model.restore(session)
    # model.evaluate(session, official_stdout=True)
    model.evaluate_pronoun_by_frequency(session, test_data, new_list)
