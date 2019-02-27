import ujson as json


def compress_medical_data(file_name):
    old_data = list()
    with open(file_name, 'r') as f:
        for line in f:
            old_data.append(json.loads(line))

    new_data = list()
    for tmp_example in old_data:
        max_sentence_length = max(len(s) for s in tmp_example['sentences'])
        print('number of sentences:', len(tmp_example['sentences']))
        print('max sentence length:', max(len(s) for s in tmp_example['sentences']))
        print('number of words:', sum(len(s) for s in tmp_example['sentences']))
        new_sentences = list()
        tmp_merge_s = list()
        for s in tmp_example['sentences']:
            if len(tmp_merge_s) + len(s) > max_sentence_length and len(tmp_merge_s) > 0:
                new_sentences.append(tmp_merge_s)
                tmp_merge_s = list()
            tmp_merge_s += s
        if len(tmp_merge_s) > 0:
            new_sentences.append(tmp_merge_s)
        print('%')
        print('number of sentences:', len(new_sentences))
        print('max sentence length:', max(len(s) for s in new_sentences))
        print('number of words:', sum(len(s) for s in new_sentences))
        print()
        print()
        new_example = tmp_example
        new_example['sentences'] = new_sentences
        new_data.append(new_example)
    with open(file_name, 'w') as f:
        for tmp_example in new_data:
            f.write(json.dumps(tmp_example))
            f.write('\n')


def add_speaker_for_medical_data(file_name):
    old_data = list()
    with open(file_name, 'r') as f:
        for line in f:
            old_data.append(json.loads(line))

    new_data = list()
    for i, tmp_example in enumerate(old_data):
        tmp_speakers = list()
        for s in tmp_example['sentences']:
            tmp_speakers_by_sentence = list()
            for w in s:
                tmp_speakers_by_sentence.append(str(i))
            tmp_speakers.append(tmp_speakers_by_sentence)
        new_example = tmp_example
        new_example['speakers'] = tmp_speakers
        new_data.append(new_example)
    with open(file_name, 'w') as f:
        for tmp_example in new_data:
            f.write(json.dumps(tmp_example))
            f.write('\n')


add_speaker_for_medical_data('medical_data/train.pronoun.jsonlines')
add_speaker_for_medical_data('medical_data/test.pronoun.jsonlines')
print('end')
