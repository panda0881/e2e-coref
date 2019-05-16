import sys
import json

if __name__ == "__main__":
    file_names = ['train.english.jsonlines', 'dev.english.jsonlines', 'test.english.jsonlines']

    words_to_keep = set()
    for json_filename in file_names:
        with open(json_filename) as json_file:
            for line in json_file.readlines():
                for sentence in json.loads(line)["sentences"]:
                    words_to_keep.update(sentence)

    print("Found {} words in {} dataset(s).".format(len(words_to_keep), 3))

    total_lines = 0
    kept_lines = 0
    out_filename = "MRNE.filtered"
    with open('/home/data/corpora/MRWE/all_embedding_add_context.txt', 'r') as in_file:
        with open(out_filename, "w") as out_file:
            for line in in_file.readlines():
                total_lines += 1
                word = line.split()[0]
                if word in words_to_keep:
                    kept_lines += 1
                    out_file.write(line)

    print("Kept {} out of {} lines.".format(kept_lines, total_lines))
    print("Wrote result to {}.".format(out_filename))
