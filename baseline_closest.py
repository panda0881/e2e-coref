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
# all_count = dict()
# all_count['NP'] = 0
# for pronoun_type in all_pronouns_by_type:
#     all_count[pronoun_type] = 0
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
            tmp_all_NPs = list()
            Pronoun_dict = dict()
            for pronoun_type in interested_pronouns:
                Pronoun_dict[pronoun_type] = list()
            for c in all_clusters:
                for i in range(len(c)):
                    if len(c[i][1]) == 1 and c[i][1][0] in all_pronouns:
                        for pronoun_type in interested_pronouns:
                            if c[i][1][0] in all_pronouns_by_type[pronoun_type]:
                                potential_NPs = list()
                                for j in range(len(c)):
                                    if len(c[j][1]) != 1 or c[j][1][0] not in all_pronouns:
                                        potential_NPs.append(c[j][0])
                                if len(potential_NPs) > 0:
                                    Pronoun_dict[pronoun_type].append({'pronoun': c[i][0], 'NPs': potential_NPs})
                    else:
                        tmp_all_NPs.append(c[i][0])
            tmp_example['pronoun_coreference_info'] = {'all_NP': tmp_all_NPs, 'pronoun_dict': Pronoun_dict}
            test_data.append(tmp_example)
    print('finish processing data')

    coreference_result = dict()
    for pronoun_type in interested_pronouns:
        coreference_result[pronoun_type] = {'correct_coref': 0, 'all_coref': 0, 'accuracy': 0.0}
    for tmp_example in tqdm(test_data):
        all_NPs = tmp_example['all_NP']
        for conll_NP in tmp_example['pronoun_coreference_info']['all_NP']:
            if conll_NP not in all_NPs:
                all_NPs.append(conll_NP)
        for pronoun_type in interested_pronouns:
            for pronoun_example in tmp_example['pronoun_coreference_info']['pronoun_dict'][pronoun_type]:
                pronoun_span = pronoun_example['pronoun']
                correct_NPs = pronoun_example['NPs']
                current_NP_span = [-1, -1]
                current_distance = 1000
                for NP in all_NPs:
                    if NP[1] < pronoun_span[0]:
                        distance = pronoun_span[0]*2-NP[0]-NP[1]
                        if distance < current_distance:
                            current_distance = distance
                            current_NP_span = NP
                coreference_result[pronoun_type]['all_coref'] += 1
                if verify_correct_NP_match(current_NP_span, correct_NPs, 'cover'):
                    coreference_result[pronoun_type]['correct_coref'] += 1
                coreference_result[pronoun_type]['accuracy'] = coreference_result[pronoun_type][
                                                                   'correct_coref'] / \
                                                               coreference_result[pronoun_type][
                                                                   'all_coref']
    print(coreference_result)



# print(all_count)

print('end')
