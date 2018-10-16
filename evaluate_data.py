import os
# import tensorflow as tf
# import coref_model as cm
# import util
from util import *
import ujson as json
from pycorenlp import StanfordCoreNLP

nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(10)]
tmp_nlp_list = [StanfordCoreNLP('http://localhost:90%d' % (i + 10)) for i in range(5)]
nlp_list += tmp_nlp_list

personal_pronouns = ['I', 'me', 'we', 'us', 'you', 'she', 'her', 'he', 'him', 'it', 'them', 'they']
relative_pronouns = ['that', 'which', 'who', 'whom', 'whose', 'whichever', 'whoever', 'whomever']
demonstrative_pronouns = ['this', 'these', 'that', 'those']
indefinite_pronouns = ['anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything',
                       'neither', 'nobody', 'noone', 'nothing', 'one', 'somebody', 'someone', 'something', 'both',
                       'few', 'many', 'several', 'all', 'any', 'most', ' none', 'some']
reflexive_pronouns = ['myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves']
interrogative_pronouns = ['what', 'who', 'which', 'whom', 'whose']
possessive_pronoun = ['my', 'your', 'his', 'her', 'its', 'our', 'your', 'their', 'mine', 'yours', 'his', 'hers', 'ours',
                      'yours', 'theirs']

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

all_test_data = list()

with open('test.english.jsonlines', 'r') as f:
    for line in f:
        tmp_example = json.loads(line)
        all_sentence = list()
        separate_sentence_range = list()
        for s in tmp_example['sentences']:
            separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s)))
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

        tmp_data_to_analyze = list()
        for pair in NP_P_clusters:
            NP_position = pair[0]
            Pronoun_position = pair[1]
            NP = all_sentence[NP_position[0]:NP_position[1] + 1]
            Pronoun = all_sentence[Pronoun_position[0]:Pronoun_position[1] + 1]
            target_sentence = ''
            sentence_position = 0
            for i, sentence_s_e in enumerate(separate_sentence_range):
                if sentence_s_e[0] < Pronoun_position[0] < sentence_s_e[1]:
                    for w in tmp_example['sentences'][i]:
                        target_sentence += ' '
                        target_sentence += w
                    sentence_position = Pronoun_position[0] - sentence_s_e[0]
                    break
            if len(target_sentence) > 0:
                target_sentence = target_sentence[1:]
            cleaned_sentence = clean_sentence_for_parsing(target_sentence)
            tmp_output = nlp_list[0].annotate(cleaned_sentence,
                                      properties={'annotators': 'tokenize,depparse,lemma', 'outputFormat': 'json'})

            Before_length = 0
            stored_dependency_list = list()
            for s in tmp_output['sentences']:
                enhanced_dependency_list = s['enhancedPlusPlusDependencies']
                for relation in enhanced_dependency_list:
                    if relation['dep'] == 'ROOT':
                        continue
                    governor_position = relation['governor']
                    dependent_position = relation['dependent']
                    if governor_position+Before_length == Pronoun_position or dependent_position+Before_length == Pronoun_position:
                        stored_dependency_list.append(((governor_position, s['tokens'][governor_position - 1]['lemma'],
                                                    s['tokens'][governor_position - 1]['pos']), relation['dep'], (
                                                       dependent_position, s['tokens'][dependent_position - 1]['lemma'],
                                                       s['tokens'][dependent_position - 1]['pos'])))
            tmp_data_to_analyze.append({'NP': (NP_position, NP), 'pronoun_related_edge': stored_dependency_list})
        all_test_data.append(tmp_data_to_analyze)

with open('test_data_for_analyzing.json', 'w') as f:
    json.dump(all_test_data, f)

print('end')
