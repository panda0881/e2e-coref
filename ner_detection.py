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
        tmp_example['parsed_result'] = list()
        for w_list in tmp_example['sentences']:
            tmp_s = ''
            for w in w_list:
                tmp_s += ' '
                tmp_s += w
            if len(tmp_s) > 0:
                tmp_s = tmp_s[1:]
                tmp_output = tmp_nlp.annotate(tmp_s,
                                              properties={'annotators': 'tokenize,pos,lemma,ner', 'outputFormat': 'json'})
                tmp_example['parsed_result'].append(tmp_output)
        all_examples.append(tmp_example)

with open('parsed_test_example.json', 'w') as f:
    json.dump(all_examples, f)

print('end')
