# import coref_model as cm
# import util
# from util import *
import ujson as json
from pycorenlp import StanfordCoreNLP

no_nlp_server = 15
nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(no_nlp_server)]

tmp_nlp = nlp_list[0]

with open('test.english.jsonlines', 'r') as f:
    for line in f:
        tmp_example = json.loads(line)
        for w_list in tmp_example['sentences']:
            tmp_s = ''
            for w in w_list:
                tmp_s += ' '
                tmp_s += w
            if len(tmp_s) > 0:
                tmp_s = tmp_s[1:]
                tmp_output = tmp_nlp.annotate(tmp_s,
                                              properties={'annotators': 'tokenize,pos,lemma,ner', 'outputFormat': 'json'})
                print(tmp_output)

print('end')
