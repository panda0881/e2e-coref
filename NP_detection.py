import ujson as json
from pycorenlp import StanfordCoreNLP
from pyparsing import OneOrMore, nestedExpr
from util import *

no_nlp_server = 15
nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(no_nlp_server)]

tmp_nlp = nlp_list[0]


def detect_sub_structure(input_parsed_result, starting_posiiton=0):
    all_sub_structures = list()
    tmp_data = input_parsed_result[1:-1]
    tag = tmp_data.split(' ')[0]
    current_NP = []
    if tag == 'NP':
        word_counter = 0
        last_triger = ''
        for c in tmp_data:
            if c == '(':
                last_triger = '('
            if c == ')':
                if last_triger == '(':
                    word_counter += 1
        current_NP.append([starting_posiiton, starting_posiiton + word_counter - 1])

    front_counter = 0
    end_counter = 0
    current_sub_substructure = ''
    last_triger = ''
    word_counter = 0
    previous_word_counter = 0
    for c in tmp_data:
        if c == '(':
            front_counter += 1
            last_triger = '('
        if c == ')':
            end_counter += 1
            if last_triger == '(':
                word_counter += 1
            last_triger = ')'
        if front_counter > end_counter:
            current_sub_substructure += c
        if front_counter == end_counter and end_counter > 0:
            front_counter = 0
            end_counter = 0
            all_sub_structures.append((current_sub_substructure + ')', starting_posiiton + previous_word_counter))
            previous_word_counter += word_counter
            word_counter = 0

            current_sub_substructure = ''
    final_result = current_NP
    for tmp_sub_structure in all_sub_structures:
        tmp_result = detect_sub_structure(tmp_sub_structure[0], tmp_sub_structure[1])
        final_result += tmp_result
    return final_result

all_result = list()
with open('test.english.jsonlines', 'r') as f:
    counter = 0
    for line in f:
        print(counter)
        counter +=1
        tmp_example = json.loads(line)
        all_sentence = list()
        separate_sentence_range = list()
        for s in tmp_example['sentences']:
            separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s)))
            all_sentence += s
        sentence_for_parsing = list()
        all_NPs = []
        previous_words = 0
        for w_list in tmp_example['sentences']:
            tmp_s = ''
            for w in w_list:
                tmp_s += ' '
                tmp_s += w
            if len(tmp_s) > 0:
                tmp_s = tmp_s[1:]
                tmp_output = tmp_nlp.annotate(tmp_s,
                                              properties={'annotators': 'tokenize, parse', 'outputFormat': 'json'})
                for sub_sentence in tmp_output['sentences']:
                    parsed_result = sub_sentence['parse']
                    NPs = detect_sub_structure(' '.join(parsed_result.replace('\n', '').split()), previous_words)
                    previous_words += len(sub_sentence['tokens'])
                    for NP in NPs:
                        try:
                            if NP[0] == NP[1] and all_sentence[NP[0]] in all_pronouns:
                                print('find a pronoun', all_sentence[NP[0]], NP)
                                continue
                            else:
                                all_NPs.append(NP)
                        except:
                            print(NP)
                            print(len(all_sentence))
                            if NP[0] == NP[1] and all_sentence[NP[0]] in all_pronouns:
                                print('find a pronoun', all_sentence[NP[0]], NP)
                                continue
                            else:
                                all_NPs.append(NP)
                    # all_NPs += NPs
        tmp_example['all_NP'] = all_NPs
        all_result.append(tmp_example)

with open('test.english.jsonlines', 'w') as f:
    for example in all_result:
        f.write(json.dumps(example))
        f.write('\n')

print('end')
