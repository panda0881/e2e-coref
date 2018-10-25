# import coref_model as cm
# import util
# from util import *
import ujson as json
from pycorenlp import StanfordCoreNLP
import pickle

interested_entity_types = ['NATIONALITY', 'ORGANIZATION', 'PERSON', 'DATE', 'CAUSE_OF_DEATH', 'CITY', 'LOCATION',
                           'NUMBER', 'TITLE', 'TIME', 'ORDINAL', 'DURATION', 'MISC', 'COUNTRY', 'SET', 'PERCENT',
                           'STATE_OR_PROVINCE', 'MONEY', 'CRIMINAL_CHARGE', 'IDEOLOGY', 'RELIGION', 'URL', 'EMAIL']
interested_pronouns = ['third_personal', 'neutral', 'demonstrative']

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


def get_pronoun_related_words(example, pronoun_position):
    related_words = list()
    separate_sentence_range = list()
    all_sentence = list()
    for s in example['sentences']:
        separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s) - 1))
        all_sentence += s
    target_sentence = ''
    sentence_position = 0
    for j, sentence_s_e in enumerate(separate_sentence_range):
        if sentence_s_e[0] <= pronoun_position[0] <= sentence_s_e[1]:
            for w in example['sentences'][j]:
                target_sentence += ' '
                target_sentence += w
            sentence_position = pronoun_position[0] - sentence_s_e[0]
            break
    if len(target_sentence) > 0:
        target_sentence = target_sentence[1:]
    tmp_output = nlp_list[0].annotate(target_sentence,
                                      properties={'annotators': 'tokenize,depparse,lemma', 'outputFormat': 'json'})
    for s in tmp_output['sentences']:
        enhanced_dependency_list = s['enhancedPlusPlusDependencies']
        for relation in enhanced_dependency_list:
            if relation['dep'] == 'ROOT':
                continue
            governor_position = relation['governor']
            dependent_position = relation['dependent']
            if relation[
                'governorGloss'] in all_pronouns and sentence_position <= governor_position <= sentence_position + 2:
                if relation['dep'] in ['dobj', 'nsubj']:
                    related_words.append((relation['dep'], s['tokens'][relation['dependent']-1]['lemma']))

            if relation[
                'dependentGloss'] in all_pronouns and sentence_position <= dependent_position <= sentence_position + 2:
                if relation['dep'] in ['dobj', 'nsubj']:
                    related_words.append((relation['dep'], s['tokens'][relation['governor']-1]['lemma']))

        # Before_length += len(s['tokens'])
    return related_words


def detect_key_words(NP_words):
    tmp_s = ''
    for w in NP_words:
        tmp_s += ' '
        tmp_s += w
    if len(tmp_s) > 0:
        tmp_s = tmp_s[1:]
    parsed_result = tmp_nlp.annotate(tmp_s, properties={'annotators': 'tokenize, parse, depparse, lemma', 'outputFormat': 'json'})
    for relation in parsed_result['sentences'][0]['enhancedPlusPlusDependencies']:
        if relation['dep'] == 'ROOT':
            return parsed_result['sentences'][0]['tokens'][relation['dependent']-1]['lemma']


def find_sentence_index(example, span):
    separate_sentence_range = list()
    all_sentence = list()
    for s in example['sentences']:
        separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s) - 1))
        all_sentence += s
    for j, sentence_s_e in enumerate(separate_sentence_range):
        if sentence_s_e[0] <= span[0] <= sentence_s_e[1]:
            return j

all_examples = list()
with open('test.english.jsonlines', 'r') as f:
    for line in f:
        tmp_example = json.loads(line)
        all_examples.append(tmp_example)

# interested_pronouns = ['third_personal', 'neutral', 'demonstrative', 'possessive']
# interested_pronouns = ['third_personal']

gold_NP_sentence_distance_dict = dict()
parsed_test_data = list()
# with open('predicated_data.jsonlines', 'r') as f:
#     counter = 0
#     for line in f:
#         print('we are working on example:', counter)
#         tmp_example = all_examples[counter]
#         counter += 1
#         tmp_predicate_result = json.loads(line)
#         all_sentence = list()
#         for s in tmp_example['sentences']:
#             all_sentence += s
#         tmp_parsed_date = dict()
#         for pronoun_type in interested_pronouns:
#             tmp_parsed_date[pronoun_type] = list()
#             for pronoun_example in tmp_predicate_result[pronoun_type]:
#                 parsed_pronoun_example = pronoun_example
#                 pronoun_span = pronoun_example['pronoun']
#                 related_words = get_pronoun_related_words(tmp_example, pronoun_span)
#                 pronoun_sentence_index = find_sentence_index(tmp_example, pronoun_span)
#                 current_sentence = tmp_example['sentences'][pronoun_sentence_index]
#                 gold_NPs = pronoun_example['NPs']
#                 gold_NP_words = list()
#                 gold_NP_sentence_index = list()
#                 gold_NP_keywords = list()
#                 predicated_NPs = pronoun_example['predicated_NPs']
#                 predicated_NP_words = list()
#                 predicated_NP_index = list()
#                 predicated_NP_keywords = list()
#                 for NP in gold_NPs:
#                     gold_NP_words.append(all_sentence[NP[0]:NP[1]+1])
#                     gold_NP_sentence_index.append(find_sentence_index(tmp_example, NP[:1]))
#                     if str(find_sentence_index(tmp_example, NP)-find_sentence_index(tmp_example, pronoun_span)) not in gold_NP_sentence_distance_dict:
#                         gold_NP_sentence_distance_dict[str(find_sentence_index(tmp_example, NP[:1])-find_sentence_index(tmp_example, pronoun_span))] = 0
#                     gold_NP_sentence_distance_dict[str(find_sentence_index(tmp_example, NP[:1])-find_sentence_index(tmp_example, pronoun_span))] += 1
#                     gold_NP_keywords.append(detect_key_words(all_sentence[NP[0]:NP[1]+1]))
#                 for NP in predicated_NPs:
#                     predicated_NP_words.append(all_sentence[NP[0]:NP[1]+1])
#                     predicated_NP_index.append(find_sentence_index(tmp_example, NP[:1]))
#                     predicated_NP_keywords.append(detect_key_words(all_sentence[NP[0]:NP[1] + 1]))
#                 parsed_pronoun_example['related_words'] = related_words
#                 parsed_pronoun_example['pronoun_sentence_index'] = pronoun_sentence_index
#                 parsed_pronoun_example['current_sentence'] = current_sentence
#                 parsed_pronoun_example['gold_NP_words'] = gold_NP_words
#                 parsed_pronoun_example['gold_NP_sentence_index'] = gold_NP_sentence_index
#                 parsed_pronoun_example['gold_NP_keywords'] = gold_NP_keywords
#                 parsed_pronoun_example['predicated_NP_words'] = predicated_NP_words
#                 parsed_pronoun_example['predicated_NP_index'] = predicated_NP_index
#                 parsed_pronoun_example['predicated_NP_keywords'] = predicated_NP_keywords
#                 tmp_parsed_date[pronoun_type].append(parsed_pronoun_example)
#         parsed_test_data.append(tmp_parsed_date)
#         # print('lalala')
#
# print(gold_NP_sentence_distance_dict)
#
# with open('parsed_test_pronoun_example.jsonlines', 'w') as f:
#     for e in parsed_test_data:
#         f.write(json.dumps(e))
#         f.write('\n')








with open('SP/corpus_stats.pkl', 'rb') as f:
    corpus_stats = pickle.load(f)
# #
word2id = corpus_stats['word2id']
with open('SP/pairs_count.pkl', 'rb') as f:
    wiki_count = pickle.load(f)
nsubj_count = wiki_count['nsubj']
dobj_count = wiki_count['dobj']
print(wiki_count.keys())
print(list(nsubj_count.keys())[:10])

with open('parsed_test_pronoun_example.jsonlines', 'r') as f:
    counter = 0
    for line in f:
        print('we are working on example:', counter)
        tmp_example = all_examples[counter]
        counter += 1
        tmp_predicate_result = json.loads(line)
        all_sentence = list()
        for s in tmp_example['sentences']:
            all_sentence += s
        tmp_parsed_date = dict()
        for pronoun_type in interested_pronouns:
            tmp_parsed_date[pronoun_type] = list()
            for parsed_pronoun_example in tmp_predicate_result[pronoun_type]:
                counted_pronoun_example = parsed_pronoun_example
                pronoun_span = parsed_pronoun_example['pronoun']
                related_words = parsed_pronoun_example['related_words']
                pronoun_sentence_index = parsed_pronoun_example['pronoun_sentence_index']
                current_sentence = parsed_pronoun_example['current_sentence']
                gold_NPs = parsed_pronoun_example['NPs']
                gold_NP_words = parsed_pronoun_example['gold_NP_words']
                gold_NP_sentence_index = parsed_pronoun_example['gold_NP_sentence_index']
                gold_NP_keywords = parsed_pronoun_example['gold_NP_keywords']
                predicated_NPs = parsed_pronoun_example['predicated_NPs']
                predicated_NP_words = parsed_pronoun_example['predicated_NP_words']
                predicated_NP_index = parsed_pronoun_example['predicated_NP_index']
                predicated_NP_keywords = parsed_pronoun_example['predicated_NP_keywords']
                gold_NP_scores = list()
                predicated_NP_scores = list()
                for i, NP_keyword in enumerate(gold_NP_keywords):
                    for e in tmp_example['entities']:
                        if gold_NPs[i][0] == e[0][0] and gold_NPs[i][1] == e[0][1]:
                            NP_keyword = e[1].lower()
                    tmp_occurance = 0
                    if NP_keyword in word2id:
                        for related_word in related_words:
                            if related_word[0] == 'nsubj' and related_word[1] in word2id:
                                if (word2id[related_word[1]], word2id[NP_keyword]) in nsubj_count:
                                    tmp_occurance += nsubj_count[(word2id[related_word[1]], word2id[NP_keyword])]
                            if related_word[0] == 'dobj' and related_word[1] in word2id:
                                if (word2id[related_word[1]], word2id[NP_keyword]) in dobj_count:
                                    tmp_occurance += dobj_count[(word2id[related_word[1]], word2id[NP_keyword])]
                    gold_NP_scores.append(tmp_occurance)
                for i, NP_keyword in enumerate(predicated_NP_keywords):
                    for e in tmp_example['entities']:
                        if predicated_NPs[i][0] == e[0][0] and predicated_NPs[i][1] == e[0][1]:
                            NP_keyword = e[1].lower()
                    tmp_occurance = 0
                    if NP_keyword in word2id:
                        for related_word in related_words:
                            if related_word[0] == 'nsubj' and related_word[1] in word2id:
                                if (word2id[related_word[1]], word2id[NP_keyword]) in nsubj_count:
                                    tmp_occurance += nsubj_count[(word2id[related_word[1]], word2id[NP_keyword])]
                            if related_word[0] == 'dobj' and related_word[1] in word2id:
                                if (word2id[related_word[1]], word2id[NP_keyword]) in dobj_count:
                                    tmp_occurance += dobj_count[(word2id[related_word[1]], word2id[NP_keyword])]
                    predicated_NP_scores.append(tmp_occurance)
                counted_pronoun_example['gold_NP_scores'] = gold_NP_scores
                counted_pronoun_example['predicated_NP_scores'] = predicated_NP_scores
                # print(related_words)
                # print(predicated_NP_scores)
                # print(predicated_NP_keywords)
                # print(predicated_NP_words)

                tmp_parsed_date[pronoun_type].append(counted_pronoun_example)
        parsed_test_data.append(tmp_parsed_date)

with open('parsed_test_pronoun_example.jsonlines', 'w') as f:
    for e in parsed_test_data:
        f.write(json.dumps(e))
        f.write('\n')
correct_scores = list()
wrong_scores = list()
#
correct_count = 0
wrong_count = 0

first_correct = 0
second_correct = 0

tmp_records = list()

first_thresholds = range(10)
second_thresholds = range(1000)


with open('parsed_test_pronoun_example.jsonlines', 'r') as f:
    counter = 0
    for line in f:
        # print('we are working on example:', counter)
        tmp_example = all_examples[counter]
        counter += 1
        tmp_predicate_result = json.loads(line)
        tmp_parsed_date = dict()

        for pronoun_type in interested_pronouns:
            tmp_parsed_date[pronoun_type] = list()
            for parsed_pronoun_example in tmp_predicate_result[pronoun_type]:
                counted_pronoun_example = parsed_pronoun_example
                pronoun_span = parsed_pronoun_example['pronoun']
                related_words = parsed_pronoun_example['related_words']
                pronoun_sentence_index = parsed_pronoun_example['pronoun_sentence_index']
                current_sentence = parsed_pronoun_example['current_sentence']
                gold_NPs = parsed_pronoun_example['NPs']
                gold_NP_words = parsed_pronoun_example['gold_NP_words']
                # gold_NP_sentence_index = parsed_pronoun_example['gold_NP_sentence_index']
                gold_NP_keywords = parsed_pronoun_example['gold_NP_keywords']
                predicated_NPs = parsed_pronoun_example['predicated_NPs']
                predicated_NP_words = parsed_pronoun_example['predicated_NP_words']
                # predicated_NP_index = parsed_pronoun_example['predicated_NP_index']
                predicated_NP_keywords = parsed_pronoun_example['predicated_NP_keywords']
                gold_NP_scores = counted_pronoun_example['gold_NP_scores']
                predicated_NP_scores = counted_pronoun_example['predicated_NP_scores']
                if len(predicated_NP_scores) > 1:
                    # if predicated_NPs[0] in gold_NPs and predicated_NPs[1] not in gold_NPs:
                        # if predicated_NP_scores[1] > predicated_NP_scores[0]:
                        #     print('lalal')
                        # print('lala')
                        # if predicated_NP_scores[1] > 1000 and predicated_NP_scores[0] < 5:
                        #     if predicated_NP_scores[1] > predicated_NP_scores[0]:
                        #         second_large += 1
                        #     else:
                        #         first_large += 1
                        # tmp_records.append(predicated_NP_scores[:2])
                    if predicated_NP_scores[1] > 500 and predicated_NP_scores[0] < 1:
                        if predicated_NPs[0] in gold_NPs and predicated_NPs[1] not in gold_NPs:
                            first_correct += 1
                        if predicated_NPs[1] in gold_NPs and predicated_NPs[0] not in gold_NPs:
                            second_correct += 1

                for i, predicated_NP in enumerate(predicated_NPs[:5]):
                    if predicated_NP in gold_NPs:
                        correct_scores.append(predicated_NP_scores[i])
                    else:
                        wrong_scores.append(predicated_NP_scores[i])
                if len(predicated_NPs) > 0:
                    final_NP = predicated_NPs[0]
                    if len(predicated_NP_scores) > 1:
                        if predicated_NP_scores[0] > 0:
                            if predicated_NP_scores[1]/predicated_NP_scores[0] > 10 and predicated_NP_scores[1] > 1000:
                                final_NP = predicated_NPs[1]
                        else:
                            if predicated_NP_scores[1] > 1000:
                                final_NP = predicated_NPs[1]
                    # if predicated_NPs[0] in gold_NPs and final_NP not in gold_NPs:
                    #     print('lalal')
                    if final_NP in gold_NPs:
                        correct_count += 1
                    else:
                        wrong_count += 1
                else:
                    wrong_count += 1


print(sum(correct_scores)/len(correct_scores))
print(sum(wrong_scores)/len(wrong_scores))

print(correct_count, wrong_count, correct_count+wrong_count, correct_count/(correct_count+wrong_count))

for record in tmp_records:
    print(record)

print(first_correct)
print(second_correct)

print('end')


