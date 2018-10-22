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

# no_nlp_server = 15
# nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(no_nlp_server)]
#
# tmp_nlp = nlp_list[0]
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

with open('failed_cases.jsonlines', 'r') as f:
    for line in f:
        tmp_failed_cases = json.loads(line)
        print('lalal')

print(all_entity_type)
print('end')
