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
from tqdm import tqdm

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
    # with open('test.english.jsonlines', 'r') as f:
    #     for line in f:
    #         tmp_example = json.loads(line)
    #         all_sentence = list()
    #         for s in tmp_example['sentences']:
    #             all_sentence += s
    #         all_clusters = list()
    #         for c in tmp_example['clusters']:
    #             tmp_c = list()
    #             for w in c:
    #                 tmp_w = list()
    #                 for token in all_sentence[w[0]:w[1] + 1]:
    #                     tmp_w.append(token)
    #                 tmp_c.append((w, tmp_w))
    #             all_clusters.append(tmp_c)
    #         NP_NP_clusters = list()
    #         NP_P_clusters = list()
    #         P_P_clusters = list()
    #         for c in all_clusters:
    #             for i in range(len(c)):
    #                 for j in range(len(c)):
    #                     if i < j:
    #                         if len(c[i][1]) == 1 and c[i][1][0] in all_pronouns:
    #                             if len(c[j][1]) == 1 and c[j][1][0] in all_pronouns:
    #                                 P_P_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
    #                             else:
    #                                 NP_P_clusters.append((tuple(c[j][0]), tuple(c[i][0])))
    #                         else:
    #                             if len(c[j][1]) == 1 and c[j][1][0] in all_pronouns:
    #                                 NP_P_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
    #                             else:
    #                                 NP_NP_clusters.append((tuple(c[i][0]), tuple(c[j][0])))
    #         tmp_example['NP_NP_clusters'] = NP_NP_clusters
    #         tmp_example['NP_P_clusters'] = NP_P_clusters
    #         tmp_example['P_P_clusters'] = P_P_clusters
    #         test_data.append(tmp_example)
    # print('finish processing data')
    with open('test.english.middle.pronoun.jsonlines', 'r') as f:
        for line in f:
            test_data.append(json.loads(line))


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

        # print('we are working on NP-NP')
        counter = 0
        for tmp_example in tqdm(test_data):
            counter += 1
            # predicted_cluster = model.predict_cluster_for_one_example(session, tmp_example)

            all_sentence = list()
            for s in tmp_example['sentences']:
                all_sentence += s
            for pronoun_example in tmp_example['pronoun_info']:
                tmp_pronoun = all_sentence[pronoun_example['current_pronoun'][0]]
                current_pronoun_type = get_pronoun_type(tmp_pronoun)
                tmp_candidates = list()
                tmp_candidates.append(tuple(pronoun_example['current_pronoun']))
                for np in pronoun_example['candidate_NPs']:
                    tmp_candidates.append(tuple(np))
                tmp_example['all_candidates'] = tmp_candidates
                predicted_cluster = model.predict_cluster_for_one_example(session, tmp_example)
                for coref_cluster in predicted_cluster:
                    find_pronoun = False
                    for mention in coref_cluster:
                        # print((
                        #     mention['startIndex'] - 1 + find_all_sentence_position(tmp_result, mention['position']),
                        #     mention['endIndex'] - 2 + find_all_sentence_position(tmp_result, mention['position'])))
                        if mention[0] == pronoun_example['current_pronoun'][0]:
                            find_pronoun = True
                            # print('lalala')
                    if find_pronoun:
                        for mention in coref_cluster:
                            if verify_correct_NP_match(mention, pronoun_example['candidate_NPs'], 'exact'):
                                predict_coreference += 1
                                result_by_pronoun_type[current_pronoun_type]['predict_coreference'] += 1
                                if verify_correct_NP_match(mention, pronoun_example['correct_NPs'], 'exact'):
                                    correct_predict_coreference += 1
                                    result_by_pronoun_type[current_pronoun_type]['correct_predict_coreference'] += 1
                all_coreference += len(pronoun_example['correct_NPs'])
                result_by_pronoun_type[current_pronoun_type]['all_coreference'] += len(pronoun_example['correct_NPs'])
            if counter % 10 == 0:
                p = correct_predict_coreference / predict_coreference
                r = correct_predict_coreference / all_coreference
                f1 = 2 * p * r / (p + r)
                print("Average F1 (py): {:.2f}%".format(f1 * 100))
                print("Average precision (py): {:.2f}%".format(p * 100))
                print("Average recall (py): {:.2f}%".format(r * 100))
                print('end')
        # model.evaluate_external_data(session, test_data, official_stdout=True)


    for tmp_pronoun_type in interested_pronouns:
        print('Pronoun type:', tmp_pronoun_type)
        tmp_p = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                result_by_pronoun_type[tmp_pronoun_type]['predict_coreference']
        tmp_r = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                result_by_pronoun_type[tmp_pronoun_type]['all_coreference']
        tmp_f1 = 2 * tmp_p * tmp_r / (tmp_p + tmp_r)
        print('p:', tmp_p)
        print('r:', tmp_r)
        print('f1:', tmp_f1)
    p = correct_predict_coreference / predict_coreference
    r = correct_predict_coreference / all_coreference
    f1 = 2 * p * r / (p + r)
    print("Average F1 (py): {:.2f}%".format(f1 * 100))
    print("Average precision (py): {:.2f}%".format(p * 100))
    print("Average recall (py): {:.2f}%".format(r * 100))
    print('end')
