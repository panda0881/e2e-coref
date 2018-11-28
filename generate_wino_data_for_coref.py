import ujson as json
from pycorenlp import StanfordCoreNLP
import xml.etree.ElementTree as etree


def find_span(tokens, target_words):
    match_spans = list()
    tmp_words = list()
    parsed_result = nlp.annotate(target_words, properties=props)
    for t in parsed_result['tokens']:
        tmp_words.append(t['originalText'])
    target_length = len(tmp_words)
    for ind in (i for i,e in enumerate(tokens) if e==tmp_words[0]):
        if tokens[ind:ind+target_length]==tmp_words:
            match_spans.append((ind, ind+target_length-1))
    if target_words in ['it', 'Adam', 'Bob', 'the mouse', 'he', 'Pete', 'Fred', 'my father', 'Yakutsk', 'she', 'her', 'Alice']:
        match_spans = [match_spans[0]]
    try:
        assert len(match_spans) == 1
    except AssertionError:
        print(tokens, target_words)
        print(len(match_spans))
        assert len(match_spans) == 1
    return match_spans


def generate_one_conll_example_from_wino(sentence, pronoun, candidate_A, candidate_B, result):
    tmp_example = dict()
    tmp_example['doc_key'] = 'Wino'
    tmp_example['sentences'] = list()
    tmp_example['speakers'] = list()
    tmp_example['consituents'] = list()
    tmp_example['ner'] = list()
    tmp_example['entities'] = list()

    tmp_sentence = list()
    speakers = list()
    parsed_result = nlp.annotate(sentence, properties=props)
    for t in parsed_result['tokens']:
        tmp_sentence.append(t['originalText'])
        speakers.append('Wino')
    tmp_example['sentences'].append(tmp_sentence)
    tmp_example['speakers'].append(speakers)
    pronoun_span = find_span(tmp_sentence, pronoun)
    candidate_A_span = find_span(tmp_sentence, candidate_A)
    candidate_B_span = find_span(tmp_sentence, candidate_B)
    if result in ['A', 'A.']:
        correct_candidate_span = candidate_A_span
    elif result in ['B', 'B.']:
        correct_candidate_span = candidate_B_span
    else:
        print(result)
    tmp_example['pronoun_info'] = [{'current_pronoun': pronoun_span[0], 'candidate_NPs': candidate_A_span+candidate_B_span, 'correct_NPs': correct_candidate_span}]
    tmp_example['clusters'] = [pronoun_span+candidate_A_span+candidate_B_span]

    return tmp_example

nlp = StanfordCoreNLP('http://localhost:9000')
props = {'annotators': 'tokenize', 'outputFormat': 'json'}

tree = etree.parse('WSCollection.xml')
root = tree.getroot()
original_problems = root.getchildren()
problems = list()

for original_problem in original_problems:
    problem = dict()
    for information in original_problem.getchildren():
        if information.tag == 'answers':
            answers = information.getchildren()
            answer_list = list()
            for answer in answers:
                answer_list.append(answer.text.strip())
            problem['answers'] = answer_list
        elif information.tag == 'text':
            texts = information.getchildren()
            text_dict = dict()
            for text1 in texts:
                text_dict[text1.tag] = text1.text.replace('\n', ' ').strip()
            problem['text'] = text_dict
        elif information.tag == 'quote':
            pass
        else:
            problem[information.tag] = information.text.replace(' ', '')
    problems.append(problem)

Wino_data = list()
for i, tmp_example in enumerate(problems):
    print('question id:', i+1)
    tmp_sentence = tmp_example['text']['txt1'] + ' ' + tmp_example['text']['pron'] + ' ' + tmp_example['text']['txt2']
    tmp_pronoun = tmp_example['text']['pron']
    candidate_A = tmp_example['answers'][0]
    candidate_B = tmp_example['answers'][1]
    answer = tmp_example['correctAnswer']
    tmp_question_in_conll_format = generate_one_conll_example_from_wino(tmp_sentence, tmp_pronoun, candidate_A, candidate_B, answer)
    Wino_data.append(tmp_question_in_conll_format)


print('number of questions:', len(Wino_data))
with open('wino.jsonlines', 'w') as f:
    for tmp_question in Wino_data:
        f.write(json.dumps(tmp_question))
        f.write('\n')


print('end')
