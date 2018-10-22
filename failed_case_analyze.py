# import coref_model as cm
# import util
# from util import *
import ujson as json
from pycorenlp import StanfordCoreNLP

interested_entity_types = ['NATIONALITY', 'ORGANIZATION', 'PERSON', 'DATE', 'CAUSE_OF_DEATH', 'CITY', 'LOCATION',
                           'NUMBER', 'TITLE', 'TIME', 'ORDINAL', 'DURATION', 'MISC', 'COUNTRY', 'SET', 'PERCENT',
                           'STATE_OR_PROVINCE', 'MONEY', 'CRIMINAL_CHARGE', 'IDEOLOGY', 'RELIGION', 'URL', 'EMAIL']
interested_pronouns = ['third_personal', 'neutral', 'demonstrative', 'possessive']

third_personal_pronouns = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them',
                           'They']

neutral_pronoun = ['it', 'It']

first_and_second_personal_pronouns = ['I', 'me', 'we', 'us', 'you', 'Me', 'We', 'Us', 'You']
relative_pronouns = ['that', 'which', 'who', 'whom', 'whose', 'whichever', 'whoever', 'whomever',
                     'That', 'Which', 'Who', 'Whom', 'Whose', 'Whichever', 'Whoever', 'Whomever']
demonstrative_pronouns = ['this', 'these', 'that', 'those', 'This', 'These', 'That', 'Those']
indefinite_pronouns = ['anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything',
                       'neither', 'nobody', 'none', 'nothing', 'one', 'somebody', 'someone', 'something', 'both',
                       'few', 'many', 'several', 'all', 'any', 'most', 'some',
                       'Anybody', 'Anyone', 'Anything', 'Each', 'Either', 'Everybody', 'Everyone', 'Everything',
                       'Neither', 'Nobody', 'None', 'Nothing', 'One', 'Somebody', 'Someone', 'Something', 'Both',
                       'Few', 'Many', 'Several', 'All', 'Any', 'Most', 'Some']
reflexive_pronouns = ['myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves',
                      'Myself', 'Ourselves', 'Yourself', 'Yourselves', 'Himself', 'Herself', 'Itself', 'Themselves']
interrogative_pronouns = ['what', 'who', 'which', 'whom', 'whose', 'What', 'Who', 'Which', 'Whom', 'Whose']
possessive_pronoun = ['my', 'your', 'his', 'her', 'its', 'our', 'your', 'their', 'mine', 'yours', 'his', 'hers', 'ours',
                      'yours', 'theirs', 'My', 'Your', 'His', 'Her', 'Its', 'Our', 'Your', 'Their', 'Mine', 'Yours',
                      'His', 'Hers', 'Ours', 'Yours', 'Theirs']

all_pronouns_by_type = dict()
all_pronouns_by_type['first_and_second_personal'] = first_and_second_personal_pronouns
all_pronouns_by_type['third_personal'] = third_personal_pronouns
all_pronouns_by_type['neutral'] = neutral_pronoun
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

no_nlp_server = 15
nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(no_nlp_server)]

tmp_nlp = nlp_list[0]
#
# with open('test.english.jsonlines', 'r') as f:
#     for line in f:
#         tmp_example = json.loads(line)
#         for w_list in tmp_example['sentences']:
#             tmp_s = ''
#             for w in w_list:
#                 tmp_s += ' '
#                 tmp_s += w
#             if len(tmp_s) > 0:
#                 tmp_s = tmp_s[1:]
#                 tmp_output = tmp_nlp.annotate(tmp_s,
#                                               properties={'annotators': 'tokenize,pos,lemma,ner', 'outputFormat': 'json'})
#                 print(tmp_output)

coreference_result_by_entity_type = dict()
for entity_type in interested_entity_types:
    coreference_result_by_entity_type[entity_type] = {'correct_coref': 0, 'all_coref': 0, 'accuracy': 0.0}
coreference_result_by_entity_type['Others'] = {'correct_coref': 0, 'all_coref': 0, 'accuracy': 0.0}
all_entity_type = dict()

all_test_examples = list()
with open('parsed_test_example.jsonlines', 'r') as f:
    for line in f:
        tmp_example = json.loads(line)
        all_test_examples.append(tmp_example)

all_failed_cases = list()
with open('failed_cases.jsonlines', 'r') as f:
    for line in f:
        tmp_failed_cases = json.loads(line)
        all_failed_cases.append(tmp_failed_cases)

for i in range(len(all_test_examples)):
    print('We are working on example:', i)
    if len(all_failed_cases[i]) == 0:
        continue
    tmp_example = all_test_examples[i]
    all_sentence = list()
    separate_sentence_range = list()
    for s in tmp_example['sentences']:
        separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s)))
        all_sentence += s
    tmp_failed_cases = all_failed_cases[i]
    NP_P_pairs = list()
    for tmp_failed_case in tmp_failed_cases:
        for NP in tmp_failed_case[1]['NPs']:
            NP_P_pairs.append((NP, tmp_failed_case[1]['pronoun'], tmp_failed_case[0]))
    tmp_data_to_analyze = list()
    for pair in NP_P_pairs:
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
        # cleaned_sentence = clean_sentence_for_parsing(target_sentence)
        tmp_output = nlp_list[0].annotate(target_sentence,
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
                if governor_position + Before_length == sentence_position + 1 or dependent_position + Before_length == sentence_position + 1:
                    stored_dependency_list.append(((governor_position, s['tokens'][governor_position - 1]['lemma'],
                                                    s['tokens'][governor_position - 1]['pos']), relation['dep'], (
                                                       dependent_position, s['tokens'][dependent_position - 1]['lemma'],
                                                       s['tokens'][dependent_position - 1]['pos'])))
            Before_length += len(s['tokens'])
        # print(len(stored_dependency_list))
        tmp_data_to_analyze.append({'NP': (NP_position, NP), 'pronoun_related_edge': stored_dependency_list})
    print('lalala')

print(all_entity_type)
print('end')
