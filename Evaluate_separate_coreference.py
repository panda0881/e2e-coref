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

# if __name__ == "__main__":
# config = util.initialize_from_env()
# model = cm.CorefModel(config)
# with tf.Session() as session:
#     model.restore(session)
#     model.evaluate(session, official_stdout=True)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    personal_pronouns = ['I', 'me', 'we', 'us', 'you', 'she', 'her', 'he', 'him', 'it', 'them', 'they']
    relative_pronouns = ['that', 'which', 'who', 'whom', 'whose', 'whichever', 'whoever', 'whomever']
    demonstrative_pronouns = ['this', 'these', 'that', 'those']
    indefinite_pronouns = ['anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything',
                           'neither', 'nobody', 'noone', 'nothing', 'one', 'somebody', 'someone', 'something', 'both',
                           'few', 'many', 'several', 'all', 'any', 'most', ' none', 'some']
    reflexive_pronouns = ['myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves']
    interrogative_pronouns = ['what', 'who', 'which', 'whom', 'whose']
    possessive_pronoun = ['my', 'your', 'his', 'her', 'its', 'our', 'your', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'yours', 'theirs']

    all_pronouns_by_type = dict()
    all_pronouns_by_type['personal'] = personal_pronouns
    all_pronouns_by_type['relative'] = relative_pronouns
    all_pronouns_by_type['demonstrative'] = demonstrative_pronouns
    all_pronouns_by_type['indefinite'] = indefinite_pronouns
    all_pronouns_by_type['reflexive'] = reflexive_pronouns
    all_pronouns_by_type['interrogative'] = interrogative_pronouns
    all_pronouns_by_type['possessive'] = possessive_pronoun

    all_pronouns = list()
    for pronoun_type in all_pronouns_by_type:
        all_pronouns += all_pronouns_by_type[pronoun_type]

    all_pronouns = set(all_pronouns)

    # NP_NP_test_data = list()
    # NP_P_test_data = list()
    # P_P_test_data = list()
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
                                    NP_P_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
                            else:
                                if len(c[j][1]) == 1 and c[j][1][0] in all_pronouns:
                                    NP_P_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
                                else:
                                    NP_NP_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
            tmp_example['NP_NP_clusters'] = NP_NP_clusters
            # NP_NP_test_data.append(tmp_example)
            tmp_example['NP_P_clusters'] = NP_P_clusters
            # NP_P_test_data.append(tmp_example)
            tmp_example['P_P_clusters'] = P_P_clusters
            # P_P_test_data.append(tmp_example)
            test_data.append(tmp_example)
    print('finish processing data')


    config = util.initialize_from_env()
    model = cm.CorefModel(config)

    with tf.Session() as session:
        model.restore(session)

        # print('we are working on NP-NP')
        model.evaluate_external_data(session, test_data, official_stdout=True)



print('end')
