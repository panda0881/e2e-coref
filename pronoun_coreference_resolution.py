#!/usr/bin/env python
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import ujson as json

import os
#
import tensorflow as tf
import coref_model as cm
import util
from util import *

# if __name__ == "__main__":
# config = util.initialize_from_env()
# model = cm.CorefModel(config)
# with tf.Session() as session:
#     model.restore(session)
#     model.evaluate(session, official_stdout=True)
all_count = dict()
all_count['NP'] = 0
for pronoun_type in all_pronouns_by_type:
    all_count[pronoun_type] = 0
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    test_data = list()
    print('Start to process data...')
    with open('test.english.jsonlines', 'r') as f:
        for line in f:
            tmp_example = json.loads(line)
            all_sentence = list()
            for s in tmp_example['sentences']:
                all_sentence += s
            all_clusters = list()
            for c in tmp_example['clusters']:
                tmp_c = list()
                for w in c:
                    tmp_w = list()
                    for token in all_sentence[w[0]:w[1] + 1]:
                        tmp_w.append(token)
                    tmp_c.append((w, tmp_w))
                all_clusters.append(tmp_c)
            for c in all_clusters:
                for i in range(len(c)):
                    if len(c[i][1]) == 1 and c[i][1][0] in all_pronouns:
                        for pronoun_type in all_pronouns_by_type:
                            if c[i][1][0] in all_pronouns_by_type[pronoun_type]:
                                all_count[pronoun_type] += 1
                    else:
                        all_count['NP'] += 1
    print('finish processing data')


    # config = util.initialize_from_env()
    # model = cm.CorefModel(config)
    #
    # with tf.Session() as session:
    #     model.restore(session)
    #
    #     # print('we are working on NP-NP')
    #     model.evaluate_external_data(session, test_data, official_stdout=True)

print(all_count)

print('end')
