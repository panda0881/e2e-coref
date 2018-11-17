import ujson as json
from pycorenlp import StanfordCoreNLP
import pickle
import math

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
                    related_words.append((relation['dep'], s['tokens'][relation['dependent'] - 1]['lemma']))

            if relation[
                'dependentGloss'] in all_pronouns and sentence_position <= dependent_position <= sentence_position + 2:
                if relation['dep'] in ['dobj', 'nsubj']:
                    related_words.append((relation['dep'], s['tokens'][relation['governor'] - 1]['lemma']))

        # Before_length += len(s['tokens'])
    return related_words


def detect_key_words(NP_words):
    tmp_s = ''
    for w in NP_words:
        tmp_s += ' '
        tmp_s += w
    if len(tmp_s) > 0:
        tmp_s = tmp_s[1:]
    parsed_result = tmp_nlp.annotate(tmp_s, properties={'annotators': 'tokenize, parse, depparse, lemma',
                                                        'outputFormat': 'json'})
    for relation in parsed_result['sentences'][0]['enhancedPlusPlusDependencies']:
        if relation['dep'] == 'ROOT':
            return parsed_result['sentences'][0]['tokens'][relation['dependent'] - 1]['lemma']


def find_sentence_index(example, span):
    separate_sentence_range = list()
    all_sentence = list()
    for s in example['sentences']:
        separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s) - 1))
        all_sentence += s
    for j, sentence_s_e in enumerate(separate_sentence_range):
        if sentence_s_e[0] <= span[0] <= sentence_s_e[1]:
            return j


def get_feature_for_NP(example, NP_span):
    features = dict()
    tmp_all_sentence = list()
    for tmp_s in example['sentences']:
        tmp_all_sentence += tmp_s
    # print(NP_span)
    # print(len(tmp_all_sentence))
    NP_words = tmp_all_sentence[NP_span[0]:NP_span[1] + 1]
    tmp_s = ''
    for w in NP_words:
        tmp_s += ' '
        tmp_s += w
    if len(tmp_s) > 0:
        tmp_s = tmp_s[1:]
    parsed_result = tmp_nlp.annotate(tmp_s, properties={'annotators': 'tokenize, parse, depparse, lemma',
                                                        'outputFormat': 'json'})
    pos_of_key_word = ''
    key_word = ''
    for relation in parsed_result['sentences'][0]['enhancedPlusPlusDependencies']:
        if relation['dep'] == 'ROOT':
            pos_of_key_word = parsed_result['sentences'][0]['tokens'][relation['dependent'] - 1]['pos']
            key_word = parsed_result['sentences'][0]['tokens'][relation['dependent'] - 1]['lemma']
            break
    if pos_of_key_word == 'NNS' or 'and' in NP_words:
        features['plural'] = True
    else:
        features['plural'] = False

    features['identity'] = 'NA'
    for detected_ner in example['ner']:
        if detected_ner[0] == NP_span[0] and detected_ner[1] == NP_span[1]:
            if detected_ner[2] == 'PERSON':
                features['identity'] = 'PERSON'
            else:
                features['identity'] = 'NON-PERSON'
            break
    if key_word in person_keywords:
        features['identity'] = 'PERSON'
    return features


person_keywords = ['girl', 'boy', 'teacher', 'president', 'person', 'father', 'sister', 'friend', 'people', 'worker',
                   'citizen', 'listener', 'they', 'colleague', 'reporter', 'lawyer', 'correspondant', 'official',
                   'President', 'civilian', 'spokesman', 'student', 'commander', 'navy', 'parent', 'child', 'anyone',
                   'women', 'lover', 'employee', 'committee', 'everyone', 'you', 'God', 'Christ', 'rider', 'follower',
                   'army', 'neighbor']


def update_features(tmp_features, tmp_words):
    new_features = list()
    for i, feature in enumerate(tmp_features):
        tmp_string = ''
        for w in tmp_words[i]:
            tmp_string += ' '
            tmp_string += w
        if len(tmp_string) > 0:
            tmp_string = tmp_string[1:]
        if tmp_string in plural_words:
            feature['plural'] = True
        if tmp_string in single_words:
            feature['plural'] = False
        if tmp_string in both_words:
            feature['plural'] = 'BOTH'

        if tmp_string in person_words:
            feature['identity'] = 'PERSON'
        if tmp_string in object_words:
            feature['identity'] = 'NON-PERSON'
        if tmp_string in both_identity_words:
            feature['identity'] = 'NA'
        new_features.append(feature)
    return new_features


person_words = ['Tanshui', 'that guy']

object_words = []

both_identity_words = ['Peter Pan']

plural_words = ['seven main types of pipes buried underground', 'many of them',
                'somebody quite close to this investigation', 'the British Royal Marines', 'the North Koreans',
                'North Koreans', 'The North Koreans', 'these missile tests the other day', 'the regime in North Korea',
                'The Iranians', 'arms control agreements', 'the Iranians', 'nearly 20 million left - behind children',
                'at least half of them', 'many left - behind children', '51 percent of the children we surveyed',
                'the majority of the parents',
                'a couple of them -LRB- BTW , some of the most respectable people I know', 'all of his children',
                'the " 95 % " laboring people', 'this generation of poor people',
                'the rest of the channels', "a urine sample or a blood sample", "all of Lorraine 's friends",
                'an architectural group', "only those who did not have God 's mark on their foreheads",
                'those who were no longer pure enough to enter the place of worship', 'any of those from Macedonia',
                'All who compete in the games', 'those who are weak', 'Some Jews', 'Most of the people', 'they all',
                'all of them', 'some of them', 'the group of followers', 'Some of the Pharisees',
                'anyone who said Jesus was the Christ', 'The whole group of followers',
                'those who do not use what they have', 'all of them', 'They all',
                'his um , um , grandparents who had custody of him', '58 percent of the left - behind children',
                'at least three Yemenis', 'those that took the view at that time',
                "terrorists prevented from attacking their targets using their more conventional means ` truck bombs '",
                'a crowd of some 300,000 people', 'armed Palestinians trying to cut through a border fence',
                "Eighty percent of Dongguan 's 1,800 computer - related companies",
                'them : women draped with beauty pageant - like sashes with the Chinese characters for " Global Travel " emblazoned across them',
                'the small number of biochip companies so far established in Taiwan', 'High - rises',
                'many middle - aged who have worked diligently for years to attain management positions',
                "today 's middle - aged Chinese women", 'Middle - aged people', 'the middle - aged',
                '64 % of people aged 65 to 74',
                "one of today 's fastest - growing income groups , the upper - middle class",
                'the upper - middle class',
                'many popular , though revenue - losing , provisions , a number of which are included in the House - passed bill',
                'The explosion of junk bonds and takeovers',
                'Everyone who has left houses , brothers , sisters , father , mother , children , or farms to follow me',
                'the elderly', 'these left - behind children', 'this group of left - behind children',
                'such a big harvest of people to bring in']

single_words = ['traffic management and emergency repair', 'this kind of early warning and forecast mechanism',
                'this policy of patient diplomacy and of crafting multilateral coalitions to tighten the pressure around them',
                'Security assurances we will not topple their regime',
                "Fifty Years of Peter Pan , Roger Lancelyn Green , pub. by Peter Davies , London , 1954",
                "the socialism underpinned by planned economy , public ownership and the people 's democratic dictatorship",
                'everything that has happened in the last several years yeah . and things like that',
                'Security assurances we will not topple their regime .', 'flows',
                'nationally syndicated cartoonist and author John Callahan , who is a quadriplegic',
                'Milosevic , whose parents and uncle committed suicide',
                'three days of good vibes and who knows what else',
                "Sharon Osbourne , Ozzy 's long - time manager , wife and best friend",
                "the fact that she did meet with Bashar Al - Assad and it was n't Saudi Arabia which is traditionally being more dovish and has supported US policy",
                'supports', 'supplies', 'a biochip bearing a record and analysis of their own DNA',
                'the head of an old and disused water pipe',
                'the difference in approach betwen Mr. Mosbacher and Mrs. Hills', 'higher oil prices',
                'Conrad Leslie , a futures analyst and head of Leslie Analytical in Chicago',
                'The real estate and thrift concern', 'Ms. Cunningham , a novelist and playwright ,',
                "the court 's cutbacks in civil rights",
                'Maddie -LRB- Lynn Redgrave -RRB- , an Irish widow and mother of three',
                'a 1972 CBS sitcom called `` Bridget Loves Bernie , '' whose sole distinction was that it led to the real - life marriage of Meredith Baxter and David Birney',
                'Mr. Granville , once a widely followed market guru and still a well - known newsletter writer',
                'Mr. Rogers , a professor of finance at Columbia University and former co-manager of one of the most successful hedge funds in history , Quantum Fund',
                "the Tumen River region 's international cooperation and development program",
                'the total amount of imports and exports',
                'a 1972 CBS sitcom called `` Bridget Loves Bernie , '' whose sole distinction was that it led to the real - life marriage of Meredith Baxter and David Birney',
                'the essentially healthy and scientific " father " system', 'that false and demonstrably false charge',
                'the juren Li Yingzhen , who was wounded resisting the Japanese occupation and fled to the mainland , where he settled down',
                'registered and recently started privately owned enterprises of the whole city', 'The Martyr of the Nation , God is his protector and we do not recommend anyone above God']

both_words = ['the Times', 'Sixty Minutes', "Clinton 's office", 'the Lebanese', 'the Bush administration',
              'a Muslim Arab civilization', 'the Iranian government', 'North Korea', 'China', 'the North',
              'the Lost Boys', 'The JW', 'the Mahdi Army', '10 Sunnis', 'The Rafida', 'my own country', 'the Rafida',
              'the family', 'the Sunnis', 'the military', 'the Dutch', 'White lillies and black dresses .', 'Moses',
              'her family', 'the black dress', 'the Jews', 'Barnabas', 'some Jews', 'the Palestinians', 'the wounded',
              'the stoning and shooting', 'The FBI', 'Palestinians', 'Nickelodeon', 'Banks', 'the Scott family',
              'The Scotts', 'the Iraqis', 'the pair', 'the Gore campaign', 'the Republican leadership', 'the Gore team',
              'the Gore campaign', 'the Arabs', 'the satirical theater called `` Index ''',
              'the duo Positive Black Soul or PBS', 'the Indians', 'Russia', 'Tahsin Printing', 'The company',
              "the satirical theater called `` Index ''", 'the company',
              "the ITRI 's Molecular Biomedical Technology Division",
              'the New Party , which had already pulled out of the Advisory Group', 'Han Chinese',
              'many an old Tanshui native', 'makes', 'The Cincinnati consumer - products giant', '443 million bushels',
              'the State Department', 'CNN', 'cable - TV - system operators', "China 's Communist leadership",
              "A group of Arby 's franchisees", 'the Europeans', 'Thomson', 'fewer than half', 'one - fourth',
              'Twenty percent', 'About 40 %', "People 's Insurance Co. in Gansu Province",
              "The United Nations Industrial Development Organization", 'Customs', 'The Pharisees', 'the North Koreans',
              'the Taliban', 'High - rises', 'The Bush administration', 'the country', 'The Palestinians',
              'Positive Black Soul', 'the Israelis', 'the court',
              'beauty pageant - like sashes with the Chinese characters for " Global Travel " emblazoned across them',
              'a group']

all_examples = list()
with open('test.english.jsonlines', 'r') as f:
    for line in f:
        tmp_example = json.loads(line)
        all_examples.append(tmp_example)

parsed_test_data = list()
with open('parsed_test_pronoun_example.jsonlines', 'r') as f:
    counter = 0
    for line in f:
        # print('we are working on example:', counter)
        tmp_example = all_examples[counter]
        counter += 1
        tmp_predicate_result = json.loads(line)
        tmp_parsed_date = dict()

        all_sentence = list()
        for s in tmp_example['sentences']:
            all_sentence += s
        tmp_parsed_date = dict()
        for pronoun_type in interested_pronouns:
            tmp_parsed_date[pronoun_type] = list()
            for parsed_pronoun_example in tmp_predicate_result[pronoun_type]:
                modified_pronoun_example = parsed_pronoun_example
                pronoun_span = parsed_pronoun_example['pronoun']
                pronoun = all_sentence[pronoun_span[0]:pronoun_span[1] + 1]
                related_words = parsed_pronoun_example['related_words']
                pronoun_sentence_index = parsed_pronoun_example['pronoun_sentence_index']
                current_sentence = parsed_pronoun_example['current_sentence']
                gold_NPs = parsed_pronoun_example['NPs']
                gold_NP_words = parsed_pronoun_example['gold_NP_words']
                gold_NP_features = parsed_pronoun_example['gold_NP_features']
                # gold_NP_sentence_index = parsed_pronoun_example['gold_NP_sentence_index']
                gold_NP_keywords = parsed_pronoun_example['gold_NP_keywords']
                predicated_NPs = parsed_pronoun_example['predicated_NPs']
                predicated_NP_words = parsed_pronoun_example['predicated_NP_words']
                predicated_NP_features = parsed_pronoun_example['predicated_NP_features']
                # predicated_NP_index = parsed_pronoun_example['predicated_NP_index']
                predicated_NP_keywords = parsed_pronoun_example['predicated_NP_keywords']
                modified_pronoun_example['gold_NP_features'] = update_features(gold_NP_features, gold_NP_words)
                modified_pronoun_example['predicated_NP_features'] = update_features(predicated_NP_features,
                                                                                     predicated_NP_words)
                tmp_parsed_date[pronoun_type].append(modified_pronoun_example)
        parsed_test_data.append(tmp_parsed_date)

print('number of examples:', len(parsed_test_data))
with open('parsed_test_pronoun_example.jsonlines', 'w') as f:
    for e in parsed_test_data:
        f.write(json.dumps(e))
        f.write('\n')

print('end')
