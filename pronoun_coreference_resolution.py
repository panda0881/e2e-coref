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

def find_sentence_index(example, span):
    separate_sentence_range = list()
    all_sentence = list()
    for s in example['sentences']:
        separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s) - 1))
        all_sentence += s
    for j, sentence_s_e in enumerate(separate_sentence_range):
        if sentence_s_e[0] <= span[0] <= sentence_s_e[1]:
            return j

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

    with open('selected_example.json', 'r') as f:
        selected_example = json.load(f)
    pronoun_count = dict()
    for pronoun_type in interested_pronouns:
        pronoun_count[pronoun_type] = 0
    with open('test.english.jsonlines', 'r') as f:
        counter = 0
        for line in f:
            # if counter not in selected_example:
            #     counter += 1
            #     continue
            counter += 1
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
                    if len(c[i][1]) != 1 or c[i][1][0] not in all_pronouns:
                        tmp_all_NPs.append(c[i][0])
            for c in all_clusters:
                for i in range(len(c)):
                    if len(c[i][1]) == 1 and c[i][1][0] in all_pronouns:
                        for pronoun_type in interested_pronouns:
                            if c[i][1][0] in all_pronouns_by_type[pronoun_type]:
                                pronoun_index = find_sentence_index(tmp_example, c[i][0])
                                potential_NPs = list()
                                for j in range(len(c)):
                                    if len(c[j][1]) != 1 or c[j][1][0] not in all_pronouns:
                                        if -2 <= find_sentence_index(tmp_example, c[j][0]) - pronoun_index <= 0:
                                            potential_NPs.append(c[j][0])
                                if len(potential_NPs) > 0:
                                    pronoun_count[pronoun_type] += 1
                                    candidate_NPs = list()
                                    for NP in tmp_all_NPs:
                                        if -2 <= find_sentence_index(tmp_example, NP) - pronoun_index <= 0:
                                            candidate_NPs.append(NP)
                                    Pronoun_dict[pronoun_type].append(
                                        {'pronoun': c[i][0], 'NPs': potential_NPs, 'candidate_NPs': candidate_NPs})
            tmp_example['pronoun_coreference_info'] = {'all_NP': tmp_all_NPs, 'pronoun_dict': Pronoun_dict}
            test_data.append(tmp_example)
    print('finish processing data')
    print(pronoun_count)


    config = util.initialize_from_env()
    model = cm.CorefModel(config)

    with tf.Session() as session:
        model.restore(session)

        # print('we are working on NP-NP')
        # data_for_analysis = model.evaluate_pronoun_coreference_with_filter(session, test_data, filter_span=5, rank=False)

        predicated_data = model.evaluate_pronoun_coreference(session, test_data)
        # with open('failed_cases.jsonlines', 'w') as f:
        #     for e in data_for_analysis:
        #         f.write(json.dumps(e))
        #         f.write('\n')
        with open('predicated_data.jsonlines', 'w') as f:
            for e in predicated_data:
                f.write(json.dumps(e))
                f.write('\n')

# print(all_count)

print('end')
