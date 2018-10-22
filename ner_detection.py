# import coref_model as cm
# import util
# from util import *
import ujson as json
from pycorenlp import StanfordCoreNLP

no_nlp_server = 15
nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(no_nlp_server)]

tmp_nlp = nlp_list[0]

all_examples = list()
with open('test.english.jsonlines', 'r') as f:
    counter = 0
    for line in f:
        counter += 1
        print(counter)
        tmp_example = json.loads(line)
        # tmp_example['parsed_result'] = list()
        previous_words = 0
        detected_entities = list()
        for w_list in tmp_example['sentences']:
            tmp_s = ''
            for w in w_list:
                tmp_s += ' '
                tmp_s += w
            if len(tmp_s) > 0:
                tmp_s = tmp_s[1:]
                tmp_output = tmp_nlp.annotate(tmp_s,
                                              properties={'annotators': 'tokenize,pos,lemma,ner', 'outputFormat': 'json'})
                for s in tmp_output['sentences']:
                    for e in s['entitymentions']:
                        detected_entities.append(((e['docTokenBegin']+previous_words, e['docTokenEnd']+previous_words-1), e['ner']))
                    previous_words += len(s['tokens'])
                # tmp_example['parsed_result'].append(tmp_output)
        tmp_example['entities'] = detected_entities
        all_examples.append(tmp_example)

with open('parsed_test_example.jsonlines', 'w') as f:
    for example in all_examples:
        f.write(json.dumps(example))
        f.write('\n')

print('end')
