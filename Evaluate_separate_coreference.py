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
            NP_NP_clusters = list()
            NP_P_clusters = list()
            P_P_clusters = list()
            for c in all_clusters:
                for i in range(len(c)):
                    for j in range(len(c)):
                        if i < j:
                            if len(c[i][1]) == 1 and c[i][1][0] in all_pronouns:
                                if len(c[j][1]) == 1 and c[j][1][0] in all_pronouns:
                                    P_P_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
                                else:
                                    NP_P_clusters.append((tuple(c[j][0]), tuple(c[i][0])))
                            else:
                                if len(c[j][1]) == 1 and c[j][1][0] in all_pronouns:
                                    NP_P_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
                                else:
                                    NP_NP_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
            tmp_example['NP_NP_clusters'] = NP_NP_clusters
            tmp_example['NP_P_clusters'] = NP_P_clusters
            tmp_example['P_P_clusters'] = P_P_clusters
            test_data.append(tmp_example)
    print('finish processing data')


    config = util.initialize_from_env()
    model = cm.CorefModel(config)

    with tf.Session() as session:
        model.restore(session)

        # print('we are working on NP-NP')
        model.evaluate_external_data(session, test_data, official_stdout=True)



print('end')
