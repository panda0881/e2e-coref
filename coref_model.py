from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util
import coref_ops
import conll
import metrics
from util import *


class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        if config['use_MRNE']:
            self.MRNE_embedding = util.EmbeddingDictionary(config['MRNE_embeddings'])
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.eval_data = None  # Load eval data lazily.

        input_props = []
        input_props.append((tf.string, [None, None]))  # Tokens.
        input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # Context embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.size]))  # Head embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
        input_props.append((tf.int32, [None, None, None]))  # Character indices.
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None]))  # Speaker IDs.
        input_props.append((tf.int32, []))  # Genre.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts.
        input_props.append((tf.int32, [None]))  # Gold ends.
        input_props.append((tf.int32, [None]))  # Cluster ids.

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                                   self.config["decay_frequency"], self.config["decay_rate"],
                                                   staircase=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_example(example, is_training=True)
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
        saver = tf.train.Saver(vars_to_restore)
        checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        file_key = doc_key.replace("/", ":")
        group = self.lm_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_pronoun_example(self, example, is_training):
        clusters = example["clusters"]

        gold_mentions = sorted(example['all_candidates'])
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                if tuple(mention) in gold_mention_map:
                    cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
        tokens = np.array(tokens)

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        lm_emb = self.load_lm_embeddings(doc_key)

        example_tensors = (
            tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training,
            gold_starts, gold_ends, cluster_ids)

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors

    def tensorize_example(self, example, is_training):
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        if self.config['use_MRNE']:
            context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size+self.MRNE_embedding.size])
        else:
            context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                if self.config['use_MRNE']:
                    context_word_emb[i, j] = np.concatenate([self.context_embeddings[word], self.MRNE_embedding[word]], axis=None)
                else:
                    context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
        tokens = np.array(tokens)

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        lm_emb = self.load_lm_embeddings(doc_key)

        example_tensors = (
            tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training,
            gold_starts, gold_ends, cluster_ids)

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors

    def truncate_example(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids,
                         genre, is_training, gold_starts, gold_ends, cluster_ids):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = context_word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        tokens = tokens[sentence_offset:sentence_offset + max_training_sentences, :]
        context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
        char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        speaker_ids = speaker_ids[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_span_range = tf.range(k)  # [k]
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores,
                                                                                             0)  # [k, k]
        fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))  # [k, k]
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)  # [k, k]

        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
        top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1])  # [k, c]
        raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets  # [k, c]
        top_antecedents_mask = raw_top_antecedents >= 0  # [k, c]
        top_antecedents = tf.maximum(raw_top_antecedents, 0)  # [k, c]

        top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores,
                                                                                            top_antecedents)  # [k, c]
        top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask))  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len,
                                 speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(context_word_emb)[0]
        max_sentence_length = tf.shape(context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        if self.config["char_embedding_size"] > 0:
            char_emb = tf.gather(
                tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
                char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
                                                       util.shape(char_emb,
                                                                  3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
            flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config[
                "filter_size"])  # [num_sentences * max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                             util.shape(flattened_aggregated_char_emb,
                                                                                        1)])  # [num_sentences, max_sentence_length, emb]
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)
        if self.config['use_elmo']:
            if not self.lm_file:
                elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
                lm_embeddings = elmo_module(
                    inputs={"tokens": tokens, "sequence_len": text_len},
                    signature="tokens", as_dict=True)
                word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
                lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                                   lm_embeddings["lstm_outputs1"],
                                   lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
            lm_emb_size = util.shape(lm_emb, 2)
            lm_num_layers = util.shape(lm_emb, 3)
            with tf.variable_scope("lm_aggregation"):
                self.lm_weights = tf.nn.softmax(
                    tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
                self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
            flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
            flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights,
                                                                                     1))  # [num_sentences * max_sentence_length * emb, 1]
            aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
            aggregated_lm_emb *= self.lm_scaling
            context_emb_list.append(aggregated_lm_emb)

        context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentence, max_sentence_length]

        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)  # [num_words, emb]
        num_words = util.shape(context_outputs, 0)

        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]),
                              genre)  # [emb]

        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
                                   [1, max_sentence_length])  # [num_sentences, max_sentence_length]
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]

        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                   [1, self.max_span_width])  # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                           0)  # [num_words, max_span_width]

        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                     candidate_starts)  # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                          num_words - 1))  # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                             candidate_end_sentence_indices))  # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]),
                                           flattened_candidate_mask)  # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]
        # candidate_starts = gold_starts
        # candidate_ends = gold_ends

        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]),
                                                     flattened_candidate_mask)  # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          cluster_ids)  # [num_candidates]

        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts,
                                               candidate_ends)  # [num_candidates, emb]
        candidate_mention_scores = self.get_mention_scores(candidate_span_emb)  # [k, 1]
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [k]


        k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   util.shape(context_outputs, 0),
                                                   True)  # [1, k]
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]
        # golden_mask = tf.greater(candidate_mention_scores, tf.zeros([util.shape(candidate_mention_scores, 0)]))
        # top_span_indices = tf.where(golden_mask)
        # k = util.shape(top_span_indices, 0)
        # top_span_indices = tf.reshape(top_span_indices, [k])

        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
        top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices)  # [k]
        top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # [k]

        c = tf.minimum(self.config["max_top_antecedents"], k)

        if self.config["coarse_to_fine"]:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
                top_span_emb, top_span_mention_scores, c)
        else:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(
                top_span_emb, top_span_mention_scores, c)

        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        for i in range(self.config["coref_depth"]):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                                     top_antecedents,
                                                                                                     top_antecedent_emb,
                                                                                                     top_antecedent_offsets,
                                                                                                     top_span_speaker_ids,
                                                                                                     genre_emb)  # [k, c]
                top_antecedent_weights = tf.nn.softmax(
                    tf.concat([dummy_scores, top_antecedent_scores], 1))  # [k, c + 1]
                top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb],
                                               1)  # [k, c + 1, emb]
                attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb,
                                                  1)  # [k, emb]
                with tf.variable_scope("f"):
                    f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                   util.shape(top_span_emb, -1)))  # [k, emb]
                    top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]

        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]
        loss = tf.reduce_sum(loss)  # []

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedents, top_antecedent_scores], loss

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
                span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                       1)  # [k, max_span_width]
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
            with tf.variable_scope("head_scores"):
                self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                       2)  # [k, max_span_width, 1]
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb):
        with tf.variable_scope("mention_scores"):
            return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)  # [k, 1]

    def pseudo_get_mention_scores(self, candidate_starts, candidate_ends, gold_starts, gold_ends):
        same_start = tf.equal(tf.expand_dims(gold_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(gold_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        return tf.reduce_sum(tf.cast(same_span, tf.float32), 0)  # [num_candidates]

        # return candidate_labels
        # gold_mention_pairs = list()
        # for i in range(tf.shape(gold_starts)[0].eval()):
        #     gold_mention_pairs.append((gold_starts[i].eval(), gold_ends[i].eval()))
        # scores = list()
        # for i in range(tf.shape(candidate_starts)[0].eval()):
        #     if (candidate_starts[i].eval(), candidate_ends[i].eval()) in gold_mention_pairs:
        #         scores.append(1)
        #     else:
        #         scores.append(0)
        # return tf.convert_to_tensor(gold_mention_pairs, dtype=tf.float32)

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, genre_emb):
        k = util.shape(top_span_emb, 0)
        c = util.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents)  # [k, c]
            same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids)  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)  # [k, c]
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [k, c]

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]

        current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

        for layer in range(self.config["contextualization_layers"]):
            with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("fw_cell"):
                    cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                with tf.variable_scope("bw_cell"):
                    cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=current_inputs,
                    sequence_length=text_len,
                    initial_state_fw=state_fw,
                    initial_state_bw=state_bw)

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs,
                                                                                        2)))  # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False), example

            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, official_stdout=False):
        self.load_eval_data()

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                self.predictions, feed_dict=feed_dict)
            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends,
                                                                        predicted_antecedents, example["clusters"],
                                                                        coref_evaluator)
            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        summary_dict = {}
        conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"] = average_f1
        print("Average F1 (conll): {:.2f}%".format(average_f1))

        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}%".format(f * 100))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return util.make_summary(summary_dict), average_f1

    def predict_cluster_for_one_example(self, session, tmp_exmaple):
        tensorized_example = self.tensorize_example(tmp_exmaple, is_training=False)
        _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
        feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
        candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
            self.predictions, feed_dict=feed_dict)
        predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        return predicted_clusters

    def evaluate_pronoun_coreference(self, session, evaluation_data):
        data_for_analysis = list()

        # setting up
        def load_data_by_line(example):
            return self.tensorize_example(example, is_training=False), example

        self.eval_data = [load_data_by_line(e) for e in evaluation_data]
        num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
        print("Loaded {} eval examples.".format(len(self.eval_data)))
        all_coreference = 0
        predict_coreference = 0
        correct_predict_coreference = 0
        result_by_pronoun_type = dict()
        for tmp_pronoun_type in interested_pronouns:
            result_by_pronoun_type[tmp_pronoun_type] = {'all_coreference': 0, 'predict_coreference': 0,
                                                        'correct_predict_coreference': 0}

        # start to predict
        predicated_data = list()
        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            tmp_predicated_data = dict()
            tmp_data_for_analysis = list()
            _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                self.predictions, feed_dict=feed_dict)
            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            predicted_clusters = self.separate_clusters(top_span_starts, top_span_ends, predicted_antecedents, example)
            # print('number of all NP:', len(all_NPs))
            all_sentence = list()
            for s in example['sentences']:
                all_sentence += s

            word_index_to_sentence_index = list()
            for i, s in enumerate(example['sentences']):
                for w in s:
                    word_index_to_sentence_index.append(i)

            for i, pronoun_example in enumerate(example['pronoun_info']):
                tmp_pronoun = all_sentence[pronoun_example['current_pronoun'][0]]
                current_pronoun_type = get_pronoun_type(tmp_pronoun)

                tmp_pronoun_sentence_index = word_index_to_sentence_index[pronoun_example['current_pronoun'][0]]

                pronoun_position = -1
                for i in range(top_span_starts.shape[0]):
                    if top_span_starts[i] == pronoun_example['current_pronoun'][0] and top_span_ends[i] == pronoun_example['current_pronoun'][1]:
                        pronoun_position = i
                        break
                print(pronoun_position)
                if pronoun_position > 0:
                    # sorted_antecedents = top_antecedents[pronoun_position]
                    antecedence_to_score = dict()
                    for i in range(len(top_antecedents[pronoun_position])):
                        antecedence_to_score[str(top_antecedents[pronoun_position][i])] = \
                            top_antecedent_scores[pronoun_position][i + 1]
                    sorted_antecedents = sorted(antecedence_to_score, key=lambda x: antecedence_to_score[x],
                                                reverse=True)
                    # print(antecedence_to_score)
                    for i in range(len(sorted_antecedents)):
                        tmp_NP_position = int(sorted_antecedents[i])
                        if antecedence_to_score[sorted_antecedents[i]] > 0 and verify_correct_NP_match(
                                    [top_span_starts[tmp_NP_position], top_span_ends[tmp_NP_position]], pronoun_example['candidate_NPs'],
                                    'cover'):
                            predict_coreference += 1
                            result_by_pronoun_type[current_pronoun_type]['predict_coreference'] += 1
                            print([top_span_starts[tmp_NP_position], top_span_ends[tmp_NP_position]])
                            if verify_correct_NP_match(
                                    [top_span_starts[tmp_NP_position], top_span_ends[tmp_NP_position]], pronoun_example['correct_NPs'],
                                    'cover'):
                                correct_predict_coreference += 1
                                result_by_pronoun_type[current_pronoun_type]['correct_predict_coreference'] += 1
                            # break
                    all_coreference += len(pronoun_example['correct_NPs'])
                    result_by_pronoun_type[current_pronoun_type]['all_coreference'] += len(
                            pronoun_example['correct_NPs'])
                    print('candidate:', pronoun_example['candidate_NPs'])
                    print('correct', pronoun_example['correct_NPs'])
            if (example_num+1) % 10 == 0:
                print(example_num)
                p = correct_predict_coreference / predict_coreference
                r = correct_predict_coreference / all_coreference
                f1 = 2 * p * r / (p + r)
                print("Average F1 (py): {:.2f}%".format(f1 * 100))
                print("Average precision (py): {:.2f}%".format(p * 100))
                print("Average recall (py): {:.2f}%".format(r * 100))
                print('correct_predict_coreference', correct_predict_coreference)
                print('predict_coreference', predict_coreference)
                print('all_coreference', all_coreference)

        for tmp_pronoun_type in interested_pronouns:
            try:
                print('Pronoun type:', tmp_pronoun_type)
                tmp_p = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                        result_by_pronoun_type[tmp_pronoun_type]['predict_coreference']
                tmp_r = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                        result_by_pronoun_type[tmp_pronoun_type]['all_coreference']
                tmp_f1 = 2 * tmp_p * tmp_r / (tmp_p + tmp_r)
                print('p:', tmp_p)
                print('r:', tmp_r)
                print('f1:', tmp_f1)
            except:
                pass
        p = correct_predict_coreference / predict_coreference
        r = correct_predict_coreference / all_coreference
        f1 = 2 * p * r / (p + r)
        print("Average F1 (py): {:.2f}%".format(f1 * 100))
        print("Average precision (py): {:.2f}%".format(p * 100))
        print("Average recall (py): {:.2f}%".format(r * 100))
        print('correct_predict_coreference', correct_predict_coreference)
        print('predict_coreference', predict_coreference)
        print('all_coreference', all_coreference)
        print('end')

    def evaluate_pronoun_coreference_with_filter(self, session, evaluation_data, filter_span=2, rank=False):
        data_for_analysis = list()

        # setting up
        def load_data_by_line(example):
            return self.tensorize_pronoun_example(example, is_training=False), example

        self.eval_data = [load_data_by_line(e) for e in evaluation_data]
        num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
        print("Loaded {} eval examples.".format(len(self.eval_data)))
        coreference_result_by_pronoun = dict()
        for pronoun_type in interested_pronouns:
            coreference_result_by_pronoun[pronoun_type] = {'correct_coref': 0, 'all_coref': 0, 'accuracy': 0.0}
        coreference_result_by_entity_type = dict()
        for entity_type in interested_entity_types:
            coreference_result_by_entity_type[entity_type] = {'correct_coref': 0, 'all_coref': 0, 'accuracy': 0.0}
        coreference_result_by_entity_type['Others'] = {'correct_coref': 0, 'all_coref': 0, 'accuracy': 0.0}

        # start to predict
        correct_scores = list()
        wrong_scores = list()
        for example_num, (tensorized_example, example) in enumerate(self.eval_data):

            tmp_data_for_analysis = list()
            _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                self.predictions, feed_dict=feed_dict)
            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            predicted_clusters = self.separate_clusters(top_span_starts, top_span_ends, predicted_antecedents, example)

            all_NPs = list()
            for conll_NP in example['pronoun_coreference_info']['all_NP']:
                if conll_NP not in all_NPs:
                    all_NPs.append(conll_NP)
            # parsed_NPs = list()
            # for tmp_NP in example['all_NP']:
            #     found_overlap_NP = False
            #     for NP in all_NPs:
            #         if tmp_NP[0] <= NP[0] and tmp_NP[1] >= NP[1]:
            #             found_overlap_NP = True
            #             break
            #         if tmp_NP[0] >= NP[0] and tmp_NP[1] <= NP[1]:
            #             found_overlap_NP = True
            #             break
            #     if not found_overlap_NP:
            #         parsed_NPs.append(tmp_NP)
            # all_NPs += parsed_NPs

            tmp_entity_dict = dict()
            for detected_entity in example['entities']:
                tmp_entity_dict[str(detected_entity[0][0]) + '_' + str(detected_entity[0][1])] = detected_entity[1]

            for pronoun_type in interested_pronouns:
                valid_NPs = list()
                if pronoun_type == 'third_personal':
                    for NP in all_NPs:
                        if str(NP[0]) + '_' + str(NP[1]) not in tmp_entity_dict:
                            valid_NPs.append(NP)
                        else:
                            if tmp_entity_dict[str(NP[0]) + '_' + str(NP[1])] == 'PERSON':
                                valid_NPs.append(NP)
                elif pronoun_type == 'neutral':
                    for NP in all_NPs:
                        if str(NP[0]) + '_' + str(NP[1]) not in tmp_entity_dict:
                            valid_NPs.append(NP)
                        else:
                            if tmp_entity_dict[str(NP[0]) + '_' + str(NP[1])] != 'PERSON':
                                valid_NPs.append(NP)
                else:
                    for NP in all_NPs:
                        valid_NPs.append(NP)

                for pronoun_example in example['pronoun_coreference_info']['pronoun_dict'][pronoun_type]:
                    # print(pronoun_example)
                    pronoun_span = pronoun_example['pronoun']
                    correct_NPs = pronoun_example['NPs']
                    # detect entity_type
                    tmp_entity_type_match_dict = dict()
                    for NP_span in correct_NPs:
                        for detected_entity in example['entities']:
                            if NP_span[0] == detected_entity[0][0] and NP_span[1] == detected_entity[0][1]:
                                if detected_entity[0][1] not in tmp_entity_type_match_dict:
                                    tmp_entity_type_match_dict[detected_entity[1]] = 0
                                tmp_entity_type_match_dict[detected_entity[1]] += 1
                    if len(tmp_entity_type_match_dict) == 0:
                        most_entity_type = 'Others'
                    else:
                        sorted_entity_type = sorted(tmp_entity_type_match_dict,
                                                    key=lambda x: tmp_entity_type_match_dict[x])
                        most_entity_type = sorted_entity_type[0]

                    pronoun_position = -1
                    for i in range(top_span_starts.shape[0]):
                        if top_span_starts[i] == pronoun_span[0] and top_span_ends[i] == pronoun_span[1]:
                            pronoun_position = i
                            break
                    # print(pronoun_position)
                    coreference_result_by_pronoun[pronoun_type]['all_coref'] += 1
                    coreference_result_by_entity_type[most_entity_type]['all_coref'] += 1
                    if pronoun_position > 0:
                        # sorted_antecedents = top_antecedents[pronoun_position]
                        antecedence_to_score = dict()
                        for i in range(len(top_antecedents[pronoun_position])):
                            antecedence_to_score[str(top_antecedents[pronoun_position][i])] = \
                                top_antecedent_scores[pronoun_position][i + 1]
                        sorted_antecedents = sorted(antecedence_to_score, key=lambda x: antecedence_to_score[x],
                                                    reverse=True)
                        top_NPs = list()
                        for i in range(len(sorted_antecedents)):
                            tmp_NP_position = int(sorted_antecedents[i])
                            if [top_span_starts[tmp_NP_position], top_span_ends[tmp_NP_position]] in valid_NPs:
                                top_NPs.append([top_span_starts[tmp_NP_position], top_span_ends[tmp_NP_position]])
                                if len(top_NPs) >= filter_span:
                                    break
                        found_match = False
                        NP_match_scores = list()
                        if rank:
                            top_NPs, NP_match_scores = post_ranking(example, [top_span_starts[pronoun_position],
                                                                              top_span_ends[pronoun_position]], top_NPs)
                        for i, tmp_NP in enumerate(top_NPs):
                            if verify_correct_NP_match(tmp_NP, correct_NPs, 'exact'):
                                print('correct position:', i)
                                found_match = True
                                break
                            #     correct_scores.append(NP_match_scores[i])
                            # else:
                            #     wrong_scores.append(NP_match_scores[i])

                        if found_match:
                            coreference_result_by_pronoun[pronoun_type]['correct_coref'] += 1
                            coreference_result_by_entity_type[most_entity_type]['correct_coref'] += 1

                        coreference_result_by_pronoun[pronoun_type]['accuracy'] = \
                            coreference_result_by_pronoun[pronoun_type]['correct_coref'] / \
                            coreference_result_by_pronoun[pronoun_type]['all_coref']

                        coreference_result_by_entity_type[most_entity_type]['accuracy'] = \
                            coreference_result_by_entity_type[most_entity_type]['correct_coref'] / \
                            coreference_result_by_entity_type[most_entity_type]['all_coref']

            # print('length of collected example for analysis:', len(tmp_data_for_analysis))
            # data_for_analysis.append(tmp_data_for_analysis)

            # print('top_span_starts:', top_span_starts)
            # print('shape:', top_span_starts.shape)
            # print('top_span_ends:', top_span_ends)
            # print('shape:', top_span_ends.shape)
            # print('top_antecedents', top_antecedents)
            # print('shape:', top_antecedents.shape)
            # print('top_antecedent_scores:', top_antecedent_scores)
            # print('shape:', top_antecedent_scores.shape)
            # print('predicated_antecedents:', predicted_antecedents)
            # print('tmp_data', example['pronoun_coreference_info'])
            print('Result until example:', example_num, '/', len(self.eval_data))
            print(coreference_result_by_pronoun)
            print(coreference_result_by_entity_type)
            # print('correct:', sum(correct_scores)/len(correct_scores), 'wrong:', sum(wrong_scores)/len(wrong_scores))
            # break
        all_pronoun_correct_number = 0
        all_pronoun_numebr = 0
        for pronoun_type in coreference_result_by_pronoun:
            all_pronoun_correct_number += coreference_result_by_pronoun[pronoun_type]['correct_coref']
            all_pronoun_numebr += coreference_result_by_pronoun[pronoun_type]['all_coref']
        print(all_pronoun_correct_number, all_pronoun_numebr, all_pronoun_correct_number / all_pronoun_numebr)
        return data_for_analysis

    def separate_clusters(self, top_span_starts, top_span_ends, predicted_antecedents, example):
        NP_NP_clusters = list()
        NP_P_clusters = list()
        P_P_clusters = list()
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        all_clusters = list()
        all_sentence = list()
        for s in example['sentences']:
            all_sentence += s
        for c in predicted_clusters:
            tmp_c = list()
            for w in c:
                tmp_w = list()
                for token in all_sentence[w[0]:w[1] + 1]:
                    tmp_w.append(token)
                tmp_c.append((w, tmp_w))
            all_clusters.append(tmp_c)
        for c in all_clusters:
            for i in range(len(c)):
                for j in range(len(c)):
                    if i < j:
                        if len(c[i][1]) == 1 and c[i][1][0] in all_pronouns:
                            if len(c[j][1]) == 1 and c[j][1][0] in all_pronouns:
                                P_P_clusters.append((c[i][0], c[j][0]))
                            else:
                                NP_P_clusters.append((c[i][0], c[j][0]))
                        else:
                            if len(c[j][1]) == 1 and c[j][1][0] in all_pronouns:
                                NP_P_clusters.append((c[i][0], c[j][0]))
                            else:
                                NP_NP_clusters.append((c[i][0], c[j][0]))
        return NP_NP_clusters, NP_P_clusters, P_P_clusters

    def evaluate_pairwise_coref(self, predicted_clusters, gold_clusters):
        NP_NP_predict = len(predicted_clusters[0])
        NP_NP_correct = 0
        NP_NP_gold = len(gold_clusters[0])
        for p_pair in predicted_clusters[0]:
            for g_pair in gold_clusters[0]:
                if p_pair[0] == g_pair[0] and p_pair[1] == g_pair[1]:
                    NP_NP_correct += 1
                    break
                if p_pair[0] == g_pair[1] and p_pair[1] == g_pair[0]:
                    NP_NP_correct += 1
                    break

        NP_P_predict = len(predicted_clusters[1])
        NP_P_correct = 0
        NP_P_gold = len(gold_clusters[1])
        for p_pair in predicted_clusters[1]:
            for g_pair in gold_clusters[1]:
                if p_pair[0] == g_pair[0] and p_pair[1] == g_pair[1]:
                    NP_P_correct += 1
                    break
                if p_pair[0] == g_pair[1] and p_pair[1] == g_pair[0]:
                    NP_P_correct += 1
                    break

        P_P_predict = len(predicted_clusters[2])
        P_P_correct = 0
        P_P_gold = len(gold_clusters[2])
        for p_pair in predicted_clusters[2]:
            for g_pair in gold_clusters[2]:
                if p_pair[0] == g_pair[0] and p_pair[1] == g_pair[1]:
                    P_P_correct += 1
                    break
                if p_pair[0] == g_pair[1] and p_pair[1] == g_pair[0]:
                    P_P_correct += 1
                    break
        return NP_NP_predict, NP_NP_correct, NP_NP_gold, NP_P_predict, NP_P_correct, NP_P_gold, P_P_predict, P_P_correct, P_P_gold

    def evaluate_external_data(self, session, evaluation_data, official_stdout=False):
        def load_data_by_line(example):
            return self.tensorize_example(example, is_training=False), example

        self.eval_data = [load_data_by_line(e) for e in evaluation_data]
        num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
        print("Loaded {} eval examples.".format(len(self.eval_data)))

        all_NP_NP_predict_pair_counter = 0
        all_NP_NP_correct_pair_counter = 0
        all_NP_NP_gold_pair_counter = 0

        all_NP_P_predict_pair_counter = 0
        all_NP_P_correct_pair_counter = 0
        all_NP_P_gold_pair_counter = 0

        all_P_P_predict_pair_counter = 0
        all_P_P_correct_pair_counter = 0
        all_P_P_gold_pair_counter = 0

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                self.predictions, feed_dict=feed_dict)
            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            predicted_clusters = self.separate_clusters(top_span_starts, top_span_ends, predicted_antecedents, example)
            # print('predicted clusters:')
            # print(predicted_clusters)
            # print('gold clusters')
            # print((example['NP_NP_clusters'], example['NP_P_clusters'], example['P_P_clusters']))
            NP_NP_predict, NP_NP_correct, NP_NP_gold, NP_P_predict, NP_P_correct, NP_P_gold, P_P_predict, P_P_correct, P_P_gold = self.evaluate_pairwise_coref(
                predicted_clusters, (example['NP_NP_clusters'], example['NP_P_clusters'], example['P_P_clusters']))
            print('NP-NP correct', NP_NP_correct, '/', NP_NP_gold)
            print('NP-P correct', NP_P_correct, '/', NP_P_gold)
            print('P-P correct', P_P_correct, '/', P_P_gold)

            all_NP_NP_predict_pair_counter += NP_NP_predict
            all_NP_NP_correct_pair_counter += NP_NP_correct
            all_NP_NP_gold_pair_counter += NP_NP_gold

            all_NP_P_predict_pair_counter += NP_P_predict
            all_NP_P_correct_pair_counter += NP_P_correct
            all_NP_P_gold_pair_counter += NP_P_gold

            all_P_P_predict_pair_counter += P_P_predict
            all_P_P_correct_pair_counter += P_P_correct
            all_P_P_gold_pair_counter += P_P_gold
            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))
        #
        # summary_dict = {}
        # conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
        # average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        # summary_dict["Average F1 (conll)"] = average_f1
        # print("Average F1 (conll): {:.2f}%".format(average_f1))

        # p, r, f = coref_evaluator.get_prf()
        # summary_dict["Average F1 (py)"] = f
        NP_NP_p = all_NP_NP_correct_pair_counter / all_NP_NP_predict_pair_counter
        NP_NP_r = all_NP_NP_correct_pair_counter / all_NP_NP_gold_pair_counter
        NP_NP_f = 2 * NP_NP_p * NP_NP_r / (NP_NP_p + NP_NP_r)
        print('NP-NP:')
        print("Average F1 (py): {:.2f}%".format(NP_NP_f * 100))
        # summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(NP_NP_p * 100))
        # summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(NP_NP_r * 100))
        print(all_NP_NP_correct_pair_counter, '/', all_NP_NP_gold_pair_counter)

        NP_P_p = all_NP_P_correct_pair_counter / all_NP_P_predict_pair_counter
        NP_P_r = all_NP_P_correct_pair_counter / all_NP_P_gold_pair_counter
        NP_P_f = 2 * NP_P_p * NP_P_r / (NP_P_p + NP_P_r)
        print('NP-P:')
        print("Average F1 (py): {:.2f}%".format(NP_P_f * 100))
        # summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(NP_P_p * 100))
        # summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(NP_P_r * 100))
        print(all_NP_P_correct_pair_counter, '/', all_NP_P_gold_pair_counter)

        P_P_p = all_P_P_correct_pair_counter / all_P_P_predict_pair_counter
        P_P_r = all_P_P_correct_pair_counter / all_P_P_gold_pair_counter
        P_P_f = 2 * P_P_p * P_P_r / (P_P_p + P_P_r)
        print('P-P:')
        print("Average F1 (py): {:.2f}%".format(P_P_f * 100))
        # summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(P_P_p * 100))
        # summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(P_P_r * 100))
        print(all_P_P_correct_pair_counter, '/', all_P_P_gold_pair_counter)
