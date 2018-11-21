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

    with open('interested_examples.json', 'r') as f:
        selected_cases = json.load(f)

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
    stored_case = list()
    with tf.Session() as session:
        model.restore(session)

        # print('we are working on NP-NP')
        counter = 0
        for example_num, tmp_example in tqdm(enumerate(test_data)):
            if example_num not in [220, 262, 298, 308]:
                continue
            counter += 1
            predicted_cluster = model.predict_cluster_for_one_example(session, tmp_example)

            all_sentence = list()
            for s in tmp_example['sentences']:
                all_sentence += s
            for i, pronoun_example in enumerate(tmp_example['pronoun_info']):
                if [example_num, i] not in [[220, 2], [262, 19], [298, 14], [308, 20]]:
                    continue
                tmp_pronoun = all_sentence[pronoun_example['current_pronoun'][0]]
                current_pronoun_type = get_pronoun_type(tmp_pronoun)
                good_example = False
                tmp_predict = 0
                tmp_correct_predict = 0
                tmp_all = 0
                # print(predicted_cluster)
                found_predict = 0
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
                                tmp_predict += 1
                                # print(mention)
                                print([example_num, i])
                                print(mention)
                                if verify_correct_NP_match(mention, pronoun_example['correct_NPs'], 'exact'):
                                    tmp_correct_predict += 1
                tmp_all += len(pronoun_example['correct_NPs'])
                if tmp_correct_predict > 0 and tmp_correct_predict == tmp_all and tmp_correct_predict == tmp_predict:
                    good_example = True
                if not good_example:
                    stored_case.append((example_num, i))
    print(len(stored_case))
    # with open('interested_examples.json', 'w') as f:
    #     json.dump(stored_case, f)


