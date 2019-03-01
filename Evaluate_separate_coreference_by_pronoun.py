#!/usr/bin/env python
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import ujson as json

import os
#
import tensorflow as tf
import original_model as cm
import util
from util import *
from tqdm import tqdm

# if __name__ == "__main__":
# config = util.initialize_from_env()
# model = cm.CorefModel(config)
# with tf.Session() as session:
#     model.restore(session)
#     model.evaluate(session, official_stdout=True)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_data = list()
    print('Start to process data...')
    with open('medical_data/test.pronoun.jsonlines', 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    # with open('test.english.middle.pronoun.jsonlines', 'r') as f:
    #     for line in f:
    #         test_data.append(json.loads(line))


    config = util.initialize_from_env()
    model = cm.CorefModel(config)
    all_coreference = 0
    predict_coreference = 0
    correct_predict_coreference = 0
    result_by_pronoun_type = dict()
    for tmp_pronoun_type in interested_pronouns:
        result_by_pronoun_type[tmp_pronoun_type] = {'all_coreference': 0, 'predict_coreference': 0,
                                                    'correct_predict_coreference': 0}

    with tf.Session() as session:
        model.restore(session)
        model.evaluate_pronoun_coreference(session, test_data)
        # print('we are working on NP-NP')




