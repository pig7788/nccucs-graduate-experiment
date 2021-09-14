# -*- coding: UTF-8 -*-


import argparse
import ast
import datetime
import difflib
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from distutils.util import strtobool

import numpy
import pandas
import scipy
import tensorflow as tf
from nltk.corpus import wordnet
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from tf2crf import CRF
from tqdm import tqdm
from transformers import (BertConfig, BertTokenizer, T5Config, T5Tokenizer,
                          TFBertForTokenClassification, TFBertModel,
                          TFT5ForConditionalGeneration)

# import line_notifier

DOC_START_TOKEN = '-DOCSTART-'
NON_WEIGHTED_TOKEN = '[NWT]'
SUB_WORD_PREFIX = '##'
NON_WEIGHTED_COMPUTED_LABEL_ID = -100
NON_ENTITY_SCHEME = 'O'
START_SCHEME = 'B'
SINGLE_TAG = 'S'
FOLLOW_UP_SCHEME = 'I'
END_TAG = 'E'
GLOBAL_SETTING = None
SCHEME_CONFIG = None
ENTITY_CONFIG = None
INTENT_CONFIG = None
CURRENT_SCHEME_CONFIG = None
CURRENT_ENTITY_CONFIG = None
CURRENT_INTENT_CONFIG = None
SIMPLE_TO_COMPLETED = 'simple2completed'
COMPLETED_TO_SIMPLE = 'completed2simple'
DESC = 'desc'
SYNONYM_SET = 'synonym_set'
BACK_TRANSLATION = 'back_translation'
LEMMA_NAMES = 'lemma_names'
TAG_TO_LABEL = 'tag2label'
LABEL_TO_TAG = 'label2tag'
SCHEME_PREFIX = 'scheme:'
ENTITY_ONE_PREFIX = 'entity1:'
ENTITY_TWO_PREFIX = 'entity2:'
DESC_PREFIX = 'description:'
SLOT_FILLING = 'slot_filling'
INTENT_DETECTION = 'intent_detection'
INTENT_PREFIX = 'intent:'
INTENT_DETECTION_PRECIX = 'intent detection:'
SENTENCE_PREFIX = 'sentence:'
POSSIBLE_VALUES = 'possible_values'
NONE_ENTITY = 'none'
SLOT_FILLING_PATTERN = 'scheme: \w{2,} entity1: \w{2,} entity2: \w{2,}( description: \w{2,})?'
INTENT_PATTERN = 'intent: \w{2,}( description: \w{2,})?'
SCHEME_PATTERN = '{}|{}|{}'.format(
    START_SCHEME, FOLLOW_UP_SCHEME, NON_ENTITY_SCHEME)
ENTITY_PATTERN = '\w{2,}(\-\w{2,})?(\.)?(\w{2,})?'
FORMAT = '%(asctime)s [%(module)s] [%(levelname)s]: %(message)s'
log_file_name = '{}.log'.format(
    datetime.strftime(datetime.now(), '%Y-%m-%d'))
logging.basicConfig(level=logging.INFO, filename=log_file_name,
                    filemode='a', format=FORMAT)
T5_PRE_TRAINED_PATH = '/home/albeli/workspace/NCCU/Experiment/pre_trained/{}'
T5_PRE_TRAINED = 't5-base'
T5_TOKENIZER = T5Tokenizer.from_pretrained(T5_PRE_TRAINED)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)])
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
BERT_PRE_TRAINED = 'bert-base-cased'
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_PRE_TRAINED)
BERT_ENCODER = None
T5_STSB_TASK = None
EXTENDED_INTENT_RECORDER = dict()
EXTENDED_INTENT_DESC_RECORDER = dict()


def logging_counter(line_index, count_base, message):
    current_total_count = line_index + 1
    if current_total_count % count_base == 0:
        logging.info(message.format(current_total_count))


def load_all_data(path):
    sentences = list()
    sentences_tags = list()
    intents = list()
    logging.info('讀取檔案於:{}'.format(os.path.join(path)))
    with open(path, 'r') as reader:
        for line_index, line in enumerate(reader, 1):
            splitted_line = line.strip().split('\t')
            sentences.append(splitted_line[0])
            sentences_tags.append(splitted_line[1])
            intents.append(splitted_line[-1])
            logging_counter(line_index, 1e2, '目前已經讀取{}句Sentence')
    logging.info('共讀取了{}句Sentence、SentenceTags和Intents'.format(len(sentences)))
    return (numpy.array(sentences, dtype=numpy.object),
            numpy.array(sentences_tags, dtype=numpy.object),
            numpy.array(intents, dtype=numpy.object))


def load_t5_format_data(path):
    logging.info('讀取{}之T5格式檔案'.format(path))
    with open(path, 'r') as reader:
        for index, line in enumerate(reader, 1):
            logging_counter(index, 1e2, '已讀取之T5格式資料數:{}')
            yield line.strip().split('\t')


def load_tags_mapping_obj(load_path):
    with open(load_path, 'r') as reader:
        tags_mapping_obj = json.load(reader)
    logging.info('Mapping檔案讀取於:{}'.format(
        os.path.abspath(load_path)))
    return tags_mapping_obj


def load_setting(path):
    with open(path, 'r') as reader:
        return json.load(reader)


def set_global_setting(path):
    global GLOBAL_SETTING
    GLOBAL_SETTING = load_setting(path)


def set_scheme_config(path):
    global SCHEME_CONFIG
    SCHEME_CONFIG = load_setting(path)


def set_entity_config(path):
    global ENTITY_CONFIG
    ENTITY_CONFIG = load_setting(path)


def set_intent_config(path):
    global INTENT_CONFIG
    INTENT_CONFIG = load_setting(path)


def set_current_config(scheme, current_dataset):
    global CURRENT_SCHEME_CONFIG
    global CURRENT_ENTITY_CONFIG
    global CURRENT_INTENT_CONFIG
    CURRENT_SCHEME_CONFIG = SCHEME_CONFIG.get(scheme)
    CURRENT_ENTITY_CONFIG = ENTITY_CONFIG.get(current_dataset)
    CURRENT_INTENT_CONFIG = INTENT_CONFIG.get(current_dataset)


def generate_t5_x_input(sentence, target_token):
    return 'slot filling: sentence: {} word: {}'.format(sentence, target_token)


def generate_t5_y_input(scheme, entity, entity_desc, with_desc):
    entities = entity.split('.')
    entity_level = len(entities)
    entity_pattern = 'entity1: {} entity2: {}'
    if entity_level == 1:
        entities.append(NONE_ENTITY)
    all_entities_str = entity_pattern.format(*entities)

    if with_desc:
        return 'scheme: {} {} description: {}'.format(scheme, all_entities_str, entity_desc)
    return 'scheme: {} {}'.format(scheme, all_entities_str)


def generate_t5_intent_x_input(sentence):
    return 'intent detection: sentence: {}'.format(sentence)


def generate_t5_intent_y_input(intent, intent_desc, with_desc):
    if with_desc:
        return 'intent: {} description: {}'.format(intent, intent_desc)
    return 'intent: {}'.format(intent)


def update_tensors(all_tensors, encoded_tensors):
    for key, value in encoded_tensors.items():
        existed_tensors = all_tensors.get(key)
        if tf.is_tensor(existed_tensors):
            all_tensors[key] = tf.concat([existed_tensors, value], 0)
        else:
            all_tensors.setdefault(key, value)

    return all_tensors


def encode_all_from_generator(data_generator, max_seq_length, max_label_seq_length):
    inputs_tensors = list()
    attention_masks_tensors = list()
    token_types_ids_tensors = list()
    decoder_inputs_tensors = list()
    decoder_attention_masks_tensors = list()

    for t5_x_input, t5_y_input in data_generator:
        encoded_set, decoded_set = encode(
            t5_x_input, t5_y_input, max_seq_length, max_label_seq_length)
        inputs_tensors.extend(tf.convert_to_tensor(
            encoded_set.get('input_ids'), dtype=tf.int32))
        attention_masks_tensors.extend(tf.convert_to_tensor(
            encoded_set.get('attention_mask'), dtype=tf.int32))
        token_types_ids_tensors.extend(tf.convert_to_tensor(
            encoded_set.get('token_type_ids'), dtype=tf.int32))
        decoder_inputs_tensors.extend(tf.convert_to_tensor(
            decoded_set.get('input_ids'), dtype=tf.int32))
        decoder_attention_masks_tensors.extend(tf.convert_to_tensor(
            decoded_set.get('attention_mask'), dtype=tf.int32))

    inputs_tensors = tf.convert_to_tensor(inputs_tensors, dtype=tf.int32)
    attention_masks_tensors = tf.convert_to_tensor(
        attention_masks_tensors, dtype=tf.int32)
    token_types_ids_tensors = tf.convert_to_tensor(
        token_types_ids_tensors, dtype=tf.int32)
    decoder_attention_masks_tensors = tf.convert_to_tensor(
        decoder_attention_masks_tensors, dtype=tf.int32)
    decoder_inputs_tensors = tf.convert_to_tensor(
        decoder_inputs_tensors, dtype=tf.int32)

    input_tensors_set = {
        'input_ids': inputs_tensors,
        'attention_mask': attention_masks_tensors,
        'token_type_ids': token_types_ids_tensors,
        # 'decoder_input_ids': decoder_inputs_tensors,
        'decoder_attention_mask': decoder_attention_masks_tensors,
        'labels': decoder_inputs_tensors
    }

    return input_tensors_set


def get_tensor_datasets(tensors, batch_size):
    datasets = tf.data.Dataset.from_tensor_slices(tensors)
    datasets_total_length = int(
        round(datasets.cardinality().numpy() / batch_size, 0))
    datasets = datasets.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=1000,
                                                           reshuffle_each_iteration=True).batch(batch_size)

    return datasets, datasets_total_length


def encode(t5_x_input, t5_y_input, max_seq_length, max_label_seq_length):
    encoded_set = T5_TOKENIZER.encode_plus(t5_x_input,
                                           max_length=max_seq_length,
                                           padding=True,
                                           pad_to_multiple_of=max_seq_length,
                                           return_token_type_ids=True,
                                           return_tensors='tf')
    decoded_encode_set = T5_TOKENIZER.encode_plus(
        t5_y_input,
        max_length=max_label_seq_length,
        padding=True,
        pad_to_multiple_of=max_label_seq_length,
        return_token_type_ids=True,
        return_tensors='tf')

    return encoded_set, decoded_encode_set


def get_dataset_by_condition(path, sentences, sentences_tags, intents, with_desc, execute_type, co_train_info, augment_srcs, prob_threshold, extend_repeat, shuffle_times, force_generate):
    if not os.path.isfile(path) or force_generate:
        logging.info('檔案不存在於:{}，重新執行產生檔案作業'.format(path))

        all_data = list()
        if execute_type == SLOT_FILLING:
            logging.info('產生原始Slot Filling之T5格式檔案中')
            ori_slot_filling_generator = get_ori_slot_filling_data(CURRENT_ENTITY_CONFIG,
                                                                   sentences, sentences_tags, with_desc)
            for slot_filling_data in ori_slot_filling_generator:
                all_data.append(slot_filling_data)

        elif execute_type == INTENT_DETECTION:
            logging.info('產生原始Intention之T5格式檔案中')
            intent_generator = get_intent_data(
                CURRENT_INTENT_CONFIG, sentences, intents, with_desc)
            for intent_data in intent_generator:
                all_data.append(intent_data)

        for augment_type in augment_srcs:
            if execute_type == SLOT_FILLING:
                logging.info(
                    '產生自{}擴充的Slot Filling之T5格式檔案中'.format(augment_type))
                extended_slot_filling_generator = get_extended_slot_filling_data(CURRENT_ENTITY_CONFIG,
                                                                                 sentences, sentences_tags, with_desc, augment_type, prob_threshold, extend_repeat)
                for extended_slot_filling_data in extended_slot_filling_generator:
                    all_data.extend(extended_slot_filling_data)

            elif execute_type == INTENT_DETECTION:
                logging.info(
                    '產生自{}擴充的Intention之T5格式檔案中'.format(augment_type))
                extended_intent_generator = get_extended_intent_data(
                    CURRENT_INTENT_CONFIG, sentences, intents, with_desc, augment_type, prob_threshold, extend_repeat)
                for extended_intent_data in extended_intent_generator:
                    all_data.extend(extended_intent_data)

        for co_train_path, co_train_type in co_train_info:
            co_train_sentences, co_train_sentences_tags, co_train_intents = load_all_data(
                co_train_path)

            if execute_type != SLOT_FILLING:
                logging.info('產生Co-Train的Slot Filling之T5格式檔案中')
                co_train_entity_config = ENTITY_CONFIG.get(co_train_type)
                ori_slot_filling_generator = get_ori_slot_filling_data(co_train_entity_config,
                                                                       co_train_sentences, co_train_sentences_tags, with_desc)
                for slot_filling_data in ori_slot_filling_generator:
                    all_data.append(slot_filling_data)

            elif execute_type != INTENT_DETECTION:
                logging.info('產生Co-Train的Intention之T5格式檔案中')
                co_train_intent_config = INTENT_CONFIG.get(co_train_type)
                intent_generator = get_intent_data(co_train_intent_config,
                                                   co_train_sentences, co_train_intents, with_desc)
                for intent_data in intent_generator:
                    all_data.append(intent_data)

        logging.info('將所有資料隨機Shuffle，共Shuffle{}次'.format(shuffle_times))
        for times in range(shuffle_times):
            random.shuffle(all_data)

        logging.info('完成之檔案寫入於: {}'.format(path))
        with open(path, 'w') as writer:
            for index, data in enumerate(all_data, 1):
                logging_counter(index, 1e2, '已寫入之T5格式資料數:{}')
                writer.write(data + os.linesep)
            logging.info('共寫入{}筆T5格式資料數'.format(len(all_data)))
    for t5_format_data in load_t5_format_data(path):
        yield t5_format_data


def get_ori_slot_filling_data(entity_config, sentences, sentences_tags, with_desc):
    for sentence, sentence_tags in zip(sentences, sentences_tags):
        t5_x_y_input_set = get_t5_x_y_input(entity_config,
                                            sentence.split(), sentence_tags.split(), with_desc)
        yield t5_x_y_input_set


def get_intent_data(intent_config, sentences, intents, with_desc):
    for index, (sentence, intent) in enumerate(zip(sentences, intents)):
        t5_intent_x_input = generate_t5_intent_x_input(sentence)
        completed_intent = intent_config.get(
            SIMPLE_TO_COMPLETED).get(intent)
        intent_descs = intent_config.get(
            DESC).get(intent)
        if isinstance(intent_descs, list):
            for intent_desc in intent_descs:
                t5_intent_y_input = generate_t5_intent_y_input(
                    completed_intent, intent_desc, with_desc)
                yield '\t'.join([t5_intent_x_input, t5_intent_y_input])
        else:
            t5_intent_y_input = generate_t5_intent_y_input(
                completed_intent, intent_descs, with_desc)
            yield '\t'.join([t5_intent_x_input, t5_intent_y_input])


def get_extended_intent_data(intent_config, sentences, intents, with_desc, augment_type, prob_threshold, extend_repeat):
    for index, (sentence, intent) in enumerate(zip(sentences, intents)):
        extended_intent_data_per_sentence = get_extended_intent_data_per_sentence(
            intent_config, sentence, intent, with_desc, augment_type, prob_threshold, extend_repeat)
        yield extended_intent_data_per_sentence


def get_extended_intent_data_per_sentence(intent_config, sentence, intent, with_desc, augment_type, prob_threshold, extend_repeat):
    extended_t5_intent_x_y_input_sets = list()
    for times in range(extend_repeat):
        extended_t5_intent_x_y_input = get_extended_intent_t5_x_y_input(
            intent_config, sentence, intent, with_desc, augment_type, prob_threshold)

        if isinstance(extended_t5_intent_x_y_input, list):
            extended_t5_intent_x_y_input_sets.extend(
                extended_t5_intent_x_y_input)
        else:
            extended_t5_intent_x_y_input_sets.append(
                extended_t5_intent_x_y_input)

    return extended_t5_intent_x_y_input_sets


def get_extended_intent_t5_x_y_input(intent_config, sentence, intent, with_desc, augment_type, prob_threshold):
    t5_intent_x_input = generate_t5_intent_x_input(sentence)
    completed_intent = intent_config.get(
        SIMPLE_TO_COMPLETED).get(intent)

    intent_desc = None
    if with_desc:
        intent_desc = intent_config.get(DESC).get(intent)

    intent_to_augment = completed_intent
    intent_desc_or_descs_to_augment = intent_desc
    guess_prob = random.uniform(0, 1)
    if guess_prob > prob_threshold:
        intents_to_augment = intent_config.get(
            augment_type).get(LEMMA_NAMES).get(intent)
        guessed_intent_lemma_names = EXTENDED_INTENT_RECORDER.setdefault(
            intent, set())
        not_guessed_intent_lemma_names = list(set.difference(
            set(intents_to_augment), guessed_intent_lemma_names))

        if not len(not_guessed_intent_lemma_names):
            not_guessed_intent_lemma_names = list(guessed_intent_lemma_names)

        simple_intent_to_augment = random.choice(
            not_guessed_intent_lemma_names)
        guessed_intent_lemma_names.add(simple_intent_to_augment)

        intent_to_augment = intent_config.get(
            augment_type).get(SIMPLE_TO_COMPLETED).get(simple_intent_to_augment)
        if with_desc:
            intent_descs_to_augment = intent_config.get(
                augment_type).get(DESC).get(intent)
            guessed_intent_descs = EXTENDED_INTENT_DESC_RECORDER.setdefault(
                intent, set())
            not_guessed_intent_descs = list(
                set.difference(set(intent_descs_to_augment), guessed_intent_descs))

            if not len(not_guessed_intent_descs):
                not_guessed_intent_descs = list(guessed_intent_descs)

            intent_desc_or_descs_to_augment = random.choice(
                not_guessed_intent_descs)
            guessed_intent_descs.add(intent_desc_or_descs_to_augment)

    if isinstance(intent_desc_or_descs_to_augment, list):
        partial_t5_intent_x_y_input_set = list()
        for intent_desc_to_augment in intent_desc_or_descs_to_augment:
            extended_t5_intent_y_input = generate_t5_intent_y_input(
                intent_to_augment, intent_desc_to_augment, with_desc)
            partial_t5_intent_x_y_input_set.append(
                '\t'.join([t5_intent_x_input, extended_t5_intent_y_input]))
        return partial_t5_intent_x_y_input_set

    extended_t5_intent_y_input = generate_t5_intent_y_input(
        intent_to_augment, intent_desc_or_descs_to_augment, with_desc)

    return '\t'.join([t5_intent_x_input, extended_t5_intent_y_input])


def get_extended_slot_filling_data(entity_config, sentences, sentences_tags, with_desc, augment_type, prob_threshold, extend_repeat):
    for sentence, sentence_tags in zip(sentences, sentences_tags):
        extended_slot_filling_data_per_sentence = get_extended_slot_filling_data_per_sentence(entity_config,
                                                                                              sentence.split(), sentence_tags.split(), with_desc, augment_type, prob_threshold, extend_repeat)

        yield extended_slot_filling_data_per_sentence


def get_extended_slot_filling_data_per_sentence(entity_config, sentence, sentence_tags, with_desc, augment_type, prob_threshold, extend_repeat):
    extended_t5_x_y_input_sets = list()
    for times in range(extend_repeat):
        extended_t5_x_y_input_sets.append(get_extended_t5_x_y_input(entity_config,
                                                                    sentence, sentence_tags, with_desc, augment_type, prob_threshold))
    return extended_t5_x_y_input_sets


def get_t5_x_y_input(entity_config, src_sentence, src_sentence_tags, with_desc):
    t5_x_y_input = list()
    for src_token, src_tag in zip(src_sentence, src_sentence_tags):
        original_scheme = find_target_from_pattern(SCHEME_PATTERN, src_tag)
        original_entity = find_target_from_pattern(ENTITY_PATTERN, src_tag)
        completed_scheme = CURRENT_SCHEME_CONFIG.get(
            SIMPLE_TO_COMPLETED).get(original_scheme)
        completed_entity = entity_config.get(
            SIMPLE_TO_COMPLETED).get(original_entity)

        if original_scheme == NON_ENTITY_SCHEME and not original_entity:
            completed_entity = entity_config.get(
                SIMPLE_TO_COMPLETED).get(original_scheme)

        completed_descs = None
        if with_desc:
            completed_descs = entity_config.get(
                DESC).get(original_entity)

            if original_scheme == NON_ENTITY_SCHEME and not original_entity:
                completed_descs = entity_config.get(
                    DESC).get(original_scheme)

        sentence_to_generate = ' '.join(src_sentence)
        t5_x_input = generate_t5_x_input(sentence_to_generate, src_token)
        t5_y_input = generate_t5_y_input(
            completed_scheme, completed_entity, completed_descs, with_desc)
        t5_x_y_input.append((t5_x_input, t5_y_input))
    formator = '{}\t{}'
    t5_x_y_input_set = os.linesep.join(list(map(lambda item: formator.format(
        item[0], item[-1]), t5_x_y_input)))

    return t5_x_y_input_set


def get_extended_t5_x_y_input(entity_config, src_sentence, src_sentence_tags, with_desc, augment_type, prob_threshold):
    extended_t5_x_y_input = list()
    for src_token, src_tag in zip(src_sentence, src_sentence_tags):
        original_scheme = find_target_from_pattern(SCHEME_PATTERN, src_tag)
        original_entity = find_target_from_pattern(ENTITY_PATTERN, src_tag)
        completed_scheme = CURRENT_SCHEME_CONFIG.get(
            SIMPLE_TO_COMPLETED).get(original_scheme)
        completed_entity = entity_config.get(
            SIMPLE_TO_COMPLETED).get(original_entity)
        if original_scheme == NON_ENTITY_SCHEME and not original_entity:
            completed_entity = entity_config.get(
                SIMPLE_TO_COMPLETED).get(original_scheme)

        completed_descs = None
        if with_desc:
            completed_descs = entity_config.get(
                DESC).get(original_entity)

            if original_scheme == NON_ENTITY_SCHEME and not original_entity:
                completed_descs = entity_config.get(
                    DESC).get(original_scheme)

        entity_to_augment = completed_entity
        desc_to_augment = completed_descs
        if original_scheme != NON_ENTITY_SCHEME:
            guess_prob = random.uniform(0, 1)
            if guess_prob > prob_threshold:
                augment_entities = entity_config.get(
                    augment_type).get(LEMMA_NAMES).get(original_entity)
                simple_entity_to_augment = random.choice(augment_entities)
                entity_to_augment = entity_config.get(
                    augment_type).get(SIMPLE_TO_COMPLETED).get(simple_entity_to_augment)

                if with_desc:
                    augment_descs = entity_config.get(
                        augment_type).get(DESC).get(original_entity)
                    desc_to_augment = random.choice(augment_descs)

        sentence_to_generate = ' '.join(src_sentence)
        extended_t5_x_input = generate_t5_x_input(
            sentence_to_generate, src_token)
        extended_t5_y_input = generate_t5_y_input(
            completed_scheme, entity_to_augment, desc_to_augment, with_desc)
        extended_t5_x_y_input.append(
            (extended_t5_x_input, extended_t5_y_input))
    formator = '{}\t{}'
    extended_t5_x_y_input_set = os.linesep.join(
        list(map(lambda item: formator.format(item[0], item[-1]), extended_t5_x_y_input)))

    return extended_t5_x_y_input_set


def comput_all_input_sequence_length(path, length):
    x_input_length_recorder = numpy.empty(length, dtype=numpy.int)
    y_input_length_recorder = numpy.empty(length, dtype=numpy.int)
    with open(path, 'r') as reader:
        for index, line in enumerate(reader):
            splitted_line = line.strip().split('\t')
            current_t5_x_input = splitted_line[0]
            current_t5_y_input = splitted_line[-1]
            current_t5_x_input_length = compute_sequence_length(
                current_t5_x_input)
            current_t5_y_input_length = compute_sequence_length(
                current_t5_y_input)
            x_input_length_recorder[index] = current_t5_x_input_length
            y_input_length_recorder[index] = current_t5_y_input_length
    return x_input_length_recorder.max(), y_input_length_recorder.max()


def compute_sequence_length(input):
    return len(T5_TOKENIZER.tokenize(input))


def change_label_weight(input_ids, changing_token_id, changed_token_id):
    return tf.where(
        tf.equal(input_ids, changing_token_id),
        changed_token_id,
        input_ids)


def clear_t5_decoded_useless_tokens(string):
    return string.replace(T5_TOKENIZER.eos_token, '').replace(T5_TOKENIZER.pad_token, '').strip()


def change_t5_to_conll_eval_format(item, is_zero_shot):
    token = item[0].split('word:')[-1].strip()

    answer_item_index = 1
    answer_item_splitor = item[answer_item_index].split()
    prediction_item_index = -1 if not is_zero_shot else -2
    prediction_item_splitor = item[prediction_item_index].split()
    scheme_index = 1

    simple_scheme = CURRENT_SCHEME_CONFIG.get(
        COMPLETED_TO_SIMPLE).get(answer_item_splitor[scheme_index])
    simple_entity_info = get_entity_info(
        item[answer_item_index])
    simple_entity = CURRENT_ENTITY_CONFIG.get(
        COMPLETED_TO_SIMPLE).get(parse_entity(simple_entity_info))

    predicted_scheme = CURRENT_SCHEME_CONFIG.get(
        COMPLETED_TO_SIMPLE).get(prediction_item_splitor[scheme_index])
    prediction_entity_info = get_entity_info(
        item[prediction_item_index])
    prediction_entity = CURRENT_ENTITY_CONFIG.get(
        COMPLETED_TO_SIMPLE).get(parse_entity(prediction_entity_info))

    conll_entity_format = '{}-{}'
    answer = conll_entity_format.format(simple_scheme, simple_entity)
    if simple_scheme == NON_ENTITY_SCHEME:
        answer = conll_entity_format.replace(
            '-{}', '').format(simple_scheme)

    prediction = conll_entity_format.format(
        predicted_scheme, prediction_entity)
    if predicted_scheme == NON_ENTITY_SCHEME:
        prediction = conll_entity_format.replace(
            '-{}', '').format(predicted_scheme)

    return ' '.join([token, '.', answer, prediction]) + os.linesep


def change_bert_to_conll_eval_format(item):
    tokens = item[0].split(' ')
    answer_tags = item[1].split(' ')
    predict_tags = item[2].split(' ')
    pairs = list()
    for (token, answer_tag, predict_tag) in zip(tokens, answer_tags, predict_tags):
        predict_tag = predict_tag if predict_tag != NON_WEIGHTED_TOKEN else NON_ENTITY_SCHEME
        pairs.append(
            ' '.join([token, '.', answer_tag, predict_tag]) + os.linesep)

    return pairs


def encode_all(max_seq_length, sentences, sentences_tags, tags_mapping_obj, crf_mode=False):
    all_encoded_inputs_ids = list()
    all_encoded_attention_masks = list()
    all_encoded_token_types_ids = list()
    all_encoded_labels = list()
    all_encoded_sample_weights = list()
    all_encoded_input_ids_length = list()
    for sentence, sentence_tags in zip(sentences, sentences_tags):
        token_ids, attention_mask, token_type_ids, labels, sample_weights, current_input_ids_length = encode_by_sentence(max_seq_length, sentence.split(' '),
                                                                                                                         sentence_tags.split(' '), tags_mapping_obj, crf_mode)
        all_encoded_inputs_ids.append(token_ids)
        all_encoded_attention_masks.append(attention_mask)
        all_encoded_token_types_ids.append(token_type_ids)
        all_encoded_sample_weights.append(sample_weights)
        all_encoded_labels.append(labels)
        all_encoded_input_ids_length.append(current_input_ids_length)

    input_tensors_set = {
        'input_ids': tf.convert_to_tensor(all_encoded_inputs_ids, dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor(all_encoded_attention_masks, dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor(all_encoded_token_types_ids, dtype=tf.int32),
        'sample_weights': tf.convert_to_tensor(all_encoded_sample_weights, dtype=tf.int32),
        'labels': tf.convert_to_tensor(all_encoded_labels, dtype=tf.int32)
    }

    return (input_tensors_set, max(all_encoded_input_ids_length))


def encode_by_sentence(max_seq_length, sentence_split, sentence_tags_split, tags_mapping_obj, crf_mode):
    tokenized_tokens = list()
    tags_as_tokens_length = list()
    for index, (token, tag) in enumerate(zip(sentence_split, sentence_tags_split)):
        tokenized_token = BERT_TOKENIZER.tokenize(token)
        tokenized_token_length = len(tokenized_token)
        tokenized_tokens.extend(tokenized_token)
        if tokenized_token_length > 1:
            tags_as_tokens_length.extend([tag] * tokenized_token_length)
        else:
            tags_as_tokens_length.append(tag)

    to_convert_tokenized_tokens = list()
    to_convert_tokenized_tokens.extend(tokenized_tokens)
    to_convert_tokenized_tokens.insert(0, BERT_TOKENIZER.cls_token)
    to_convert_tokenized_tokens.append(BERT_TOKENIZER.sep_token)

    to_padding_token_ids = BERT_TOKENIZER.convert_tokens_to_ids(
        to_convert_tokenized_tokens)
    attention_mask = [1] * len(to_padding_token_ids)
    token_type_ids = [0] * len(to_padding_token_ids)

    token_ids = tf.squeeze(tf.keras.preprocessing.sequence.pad_sequences(
        [to_padding_token_ids], padding='post', maxlen=max_seq_length))
    attention_mask = tf.squeeze(tf.keras.preprocessing.sequence.pad_sequences(
        [attention_mask], padding='post', maxlen=max_seq_length))
    token_type_ids = tf.squeeze(tf.keras.preprocessing.sequence.pad_sequences(
        [token_type_ids], padding='post', maxlen=max_seq_length))
    labels_by_sentence = list(map(lambda tag: tags_mapping_obj.get(
        TAG_TO_LABEL).get(tag), tags_as_tokens_length))
    labels_by_sentence.insert(0, tags_mapping_obj.get(
        TAG_TO_LABEL).get(NON_WEIGHTED_TOKEN))
    labels_by_sentence.append(tags_mapping_obj.get(
        TAG_TO_LABEL).get(NON_WEIGHTED_TOKEN))
    labels_by_sentence = tf.squeeze(tf.keras.preprocessing.sequence.pad_sequences(
        [labels_by_sentence], padding='post', maxlen=max_seq_length))
    if not crf_mode:
        labels_by_sentence = change_label_weight(labels_by_sentence,
                                                 tags_mapping_obj.get(TAG_TO_LABEL).get(
                                                     NON_WEIGHTED_TOKEN),
                                                 NON_WEIGHTED_COMPUTED_LABEL_ID)
    sample_weights = list()
    sample_weights.extend([1] * len(tokenized_tokens))
    sample_weights.insert(0, 0)
    sample_weights.append(0)
    sample_weights = tf.squeeze(tf.keras.preprocessing.sequence.pad_sequences(
        [sample_weights], padding='post', maxlen=max_seq_length))

    return (token_ids, attention_mask, token_type_ids, labels_by_sentence, sample_weights, len(to_padding_token_ids))


def get_compiled_bert_crf_model(bert_model, max_seq_length, learning_rate):
    input_ids_layer = tf.keras.layers.Input(
        shape=(max_seq_length,), name='input_ids', dtype=tf.int32)
    attention_mask_layer = tf.keras.layers.Input(
        shape=(max_seq_length,), name='attention_mask', dtype=tf.int32)
    token_type_ids_layer = tf.keras.layers.Input(
        shape=(max_seq_length,), name='token_type_ids', dtype=tf.int32)

    output = bert_model.bert(inputs=input_ids_layer,
                             attention_mask=attention_mask_layer,
                             token_type_ids=token_type_ids_layer)[0]
    output = bert_model.dropout(output, training=False)
    output = bert_model.classifier(output)
    crf = CRF()
    output = crf(output)
    model = tf.keras.Model(
        inputs=[input_ids_layer, attention_mask_layer, token_type_ids_layer],
        outputs=output,
        name='BERT-CRF-NER')
    model.build(input_shape=(max_seq_length,))
    loss = crf.loss
    metric = crf.accuracy
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, epsilon=1e-8, clipnorm=1.0)
    sample_weight_mode = 'temporal'
    model.compile(optimizer=optimizer, loss=loss, metrics=[
                  metric], sample_weight_mode=sample_weight_mode)

    return model


def get_compiled_bert_model(bert_model, learning_rate):
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, epsilon=1e-8, clipnorm=1.0)
    loss = bert_model.compute_loss
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return bert_model


def decode_all(sentences_tokens_ids, sentences_answers, sentences_predictions, tags_mapping_obj):
    all_decoded_sentences = list()
    all_decoded_sentences_answers = list()
    all_decoded_sentences_predictions = list()
    for sentence_token_ids, sentences_answer, sentence_prediction in zip(sentences_tokens_ids, sentences_answers, sentences_predictions):
        tokens, answer, prediction = decode_by_sentence(
            sentence_token_ids, sentences_answer, sentence_prediction, tags_mapping_obj)
        all_decoded_sentences.append(' '.join(tokens))
        all_decoded_sentences_answers.append(' '.join(answer))
        all_decoded_sentences_predictions.append(' '.join(prediction))

    return (all_decoded_sentences, all_decoded_sentences_answers, all_decoded_sentences_predictions)


def decode_by_sentence(token_ids, answer, prediction, tags_mapping_obj):
    tokens = list(filter(lambda token: token != BERT_TOKENIZER.pad_token,
                         BERT_TOKENIZER.convert_ids_to_tokens(token_ids)))

    full_sentence = dict()
    nearest_token_index = 0
    for token_index, token in enumerate(tokens):
        if token.startswith(SUB_WORD_PREFIX):
            full_sentence.get(nearest_token_index)['token'] = ''.join(
                [full_sentence.get(nearest_token_index).get('token'), token.replace(SUB_WORD_PREFIX, '')])
        else:
            nearest_token_index = token_index
            answer_label = answer[token_index]
            predict_label = prediction[token_index]
            full_sentence.setdefault(token_index, {
                'token': token,
                'answer_tag': tags_mapping_obj.get(LABEL_TO_TAG).get(str(answer_label)),
                'predict_tag': tags_mapping_obj.get(LABEL_TO_TAG).get(str(predict_label))
            })

    decoded_tokens = list()
    decoded_answers = list()
    decoded_predictions = list()
    for index, token_item in sorted(full_sentence.items(), key=lambda item: item[0]):
        decoded_tokens.append(token_item.get('token'))
        decoded_answers.append(token_item.get('answer_tag'))
        decoded_predictions.append(token_item.get('predict_tag'))
    decoded_tokens.pop(0)
    decoded_tokens.pop(-1)
    decoded_answers.pop(0)
    decoded_answers.pop(-1)
    decoded_predictions.pop(0)
    decoded_predictions.pop(-1)

    return (decoded_tokens, decoded_answers, decoded_predictions)


def set_zero_shot_encoder(encoder_type):
    global BERT_ENCODER
    global T5_STSB_TASK
    if encoder_type == 'bert':
        BERT_ENCODER = TFBertModel.from_pretrained(BERT_PRE_TRAINED)
    elif encoder_type == 't5':
        config = T5Config.from_pretrained(T5_PRE_TRAINED)
        T5_STSB_TASK = TFT5ForConditionalGeneration.from_pretrained(
            T5_PRE_TRAINED, config=config)
    elif encoder_type == 'mixed':
        BERT_ENCODER = TFBertModel.from_pretrained(BERT_PRE_TRAINED)
        config = T5Config.from_pretrained(T5_PRE_TRAINED)
        T5_STSB_TASK = TFT5ForConditionalGeneration.from_pretrained(
            T5_PRE_TRAINED, config=config)


def get_intent_info(string, with_desc=False):
    splitted_items = string.split()
    if INTENT_PREFIX in splitted_items:
        intent_index = splitted_items.index(INTENT_PREFIX)

        description_index = None
        if not with_desc and DESC_PREFIX in splitted_items:
            description_index = splitted_items.index(DESC_PREFIX)

        return ' '.join(splitted_items[intent_index: description_index])


def get_entity_info(string, with_desc=False):
    splitted_items = string.split()
    if ENTITY_ONE_PREFIX in splitted_items:
        entity_index = splitted_items.index(ENTITY_ONE_PREFIX)

        description_index = None
        if not with_desc and DESC_PREFIX in splitted_items:
            description_index = splitted_items.index(
                DESC_PREFIX)

        return ' '.join(splitted_items[entity_index: description_index])


def parse_entity(t5_entity_input):
    splitted_t5_entity_input = t5_entity_input.split()
    entity_one_prefix_index = splitted_t5_entity_input.index(ENTITY_ONE_PREFIX)
    entity_two_prefix_index = splitted_t5_entity_input.index(ENTITY_TWO_PREFIX)

    entity_one = splitted_t5_entity_input[entity_one_prefix_index +
                                          1:entity_two_prefix_index]
    entity_two = splitted_t5_entity_input[entity_two_prefix_index + 1:]

    entity_one_str = ' '.join(entity_one)
    entity_two_str = ' '.join(entity_two)

    if entity_two_str == NONE_ENTITY:
        return entity_one_str
    return '.'.join([entity_one_str, entity_two_str])


def change_output(prediction, target_zero_shot_entity):
    # target_result = find_target_from_pattern(SLOT_FILLING_PATTERN, prediction)
    # if target_result:
    splitted_items = prediction.split()
    if ENTITY_ONE_PREFIX in splitted_items:
        entity_index = splitted_items.index(ENTITY_ONE_PREFIX)
        scheme_items = splitted_items[: entity_index]
        target_zero_shot_items = target_zero_shot_entity.split()

        return ' '.join(scheme_items + target_zero_shot_items)


def get_zero_shot_classes(execute_type, with_desc):
    zero_shot_classes = list()

    if execute_type == SLOT_FILLING:
        for simple_scheme, completed_scheme in CURRENT_SCHEME_CONFIG.get(SIMPLE_TO_COMPLETED).items():
            for simple_entity_type, completed_entity_type in CURRENT_ENTITY_CONFIG.get(SIMPLE_TO_COMPLETED).items():
                desc = CURRENT_ENTITY_CONFIG.get(DESC).get(simple_entity_type)
                t5_y_input = generate_t5_y_input(
                    completed_scheme, completed_entity_type, desc, with_desc)
                zero_shot_classes.append(get_entity_info(
                    clear_t5_decoded_useless_tokens(t5_y_input), with_desc=with_desc))
    elif execute_type == INTENT_DETECTION:
        for simple_intent_type, completed_intent_type in CURRENT_INTENT_CONFIG.get(SIMPLE_TO_COMPLETED).items():
            intent_descs = CURRENT_INTENT_CONFIG.get(
                DESC).get(simple_intent_type)
            if isinstance(intent_descs, list):
                for intent_desc in intent_descs:
                    t5_y_input = generate_t5_intent_y_input(
                        completed_intent_type, intent_desc, with_desc)
                    zero_shot_classes.append(get_intent_info(
                        clear_t5_decoded_useless_tokens(t5_y_input), with_desc=with_desc))
            elif isinstance(intent_descs, str):
                t5_y_input = generate_t5_intent_y_input(
                    completed_intent_type, intent_descs, with_desc)
                zero_shot_classes.append(get_intent_info(
                    clear_t5_decoded_useless_tokens(t5_y_input), with_desc=with_desc))

    return zero_shot_classes


def batch_transform_to_bert_encoded_input(cleared_decoded_sentences, entities_to_compute, zero_shot_classes, max_seq_length, is_basic_transferring):
    sentences_size = len(cleared_decoded_sentences)
    encoded_predicted_inputs = None
    predicted_inputs_to_encode = list()
    encoded_zero_shot_inputs = list()
    for item_index, (cleared_decoded_sentence, entity_to_compute) in enumerate(tqdm(zip(cleared_decoded_sentences, entities_to_compute),
                                                                                    desc='Transform to BERT Inputs',
                                                                                    total=sentences_size,
                                                                                    dynamic_ncols=True)):

        new_predicted_input = entity_to_compute
        if not is_basic_transferring:
            cleared_decoded_sentence = clear_prefixies(
                cleared_decoded_sentence)
            cleared_entity_to_compute = clear_prefixies(entity_to_compute)
            new_predicted_input = '"{}" {}'.format(
                cleared_decoded_sentence, cleared_entity_to_compute)

        predicted_inputs_to_encode.append(new_predicted_input)

        zero_shot_input_to_encode = list()
        for zero_shot_index, zero_shot_class in enumerate(zero_shot_classes):
            new_zero_shot_input = zero_shot_class
            if not is_basic_transferring:
                cleared_zero_shot_class = clear_prefixies(zero_shot_class)
                new_zero_shot_input = '"{}" {}'.format(
                    cleared_decoded_sentence, cleared_zero_shot_class)

            zero_shot_input_to_encode.append(new_zero_shot_input)

        encoded_zero_shot_inputs.append(BERT_TOKENIZER.batch_encode_plus(zero_shot_input_to_encode,
                                                                         return_tensors='tf',
                                                                         padding=True,
                                                                         pad_to_multiple_of=max_seq_length))

    encoded_predicted_inputs = BERT_TOKENIZER.batch_encode_plus(predicted_inputs_to_encode,
                                                                return_tensors='tf',
                                                                padding=True,
                                                                pad_to_multiple_of=max_seq_length)

    return encoded_predicted_inputs, encoded_zero_shot_inputs


def transform_to_bert_encoded_input(input, max_seq_length):
    return BERT_TOKENIZER.encode_plus(input,
                                      return_tensors='tf',
                                      padding=True,
                                      pad_to_multiple_of=max_seq_length)


def encode_to_embedding(encoded_entities, encoded_zero_shot_classes, batch_size):
    all_prediction_embeddings = list()
    datasets, datasets_total_length = get_tensor_datasets(
        encoded_entities, batch_size)
    for batch_index, batch_input in enumerate(tqdm(datasets,
                                                   desc='Encode Predictions to BERT embeddings',
                                                   total=datasets_total_length,
                                                   dynamic_ncols=True)):
        outputs = BERT_ENCODER(input_ids=batch_input.get('input_ids'),
                               attention_mask=batch_input.get(
                                   'attention_mask'),
                               token_type_ids=batch_input.get(
                                   'token_type_ids'),
                               return_dict=True)

        all_prediction_embeddings.extend(outputs.get(
            'pooler_output').numpy().tolist())

    all_zero_shot_classes_embeddings = list()
    for encoded_zero_shot_class_per_prediction in tqdm(encoded_zero_shot_classes,
                                                       desc='Encode Zero-Shot Classes to BERT embeddings',
                                                       total=len(
                                                           encoded_zero_shot_classes),
                                                       dynamic_ncols=True):
        datasets, datasets_total_length = get_tensor_datasets(
            encoded_zero_shot_class_per_prediction, batch_size)
        all_zero_shot_class_embedding = list()
        for batch_index, batch_input in enumerate(datasets):
            outputs = BERT_ENCODER(input_ids=batch_input.get('input_ids'),
                                   attention_mask=batch_input.get(
                'attention_mask'),
                token_type_ids=batch_input.get(
                'token_type_ids'),
                return_dict=True)

            all_zero_shot_class_embedding.extend(outputs.get(
                'pooler_output').numpy().tolist())

        all_zero_shot_classes_embeddings.append(all_zero_shot_class_embedding)

    return all_prediction_embeddings, all_zero_shot_classes_embeddings


def compute_all_cosine_similarity(predictions_embedding, zero_shot_classes_embedding):
    records = list()
    for predict_index, prediction_embedding in enumerate(tqdm(predictions_embedding,
                                                              desc='Compute Cosine Similarity',
                                                              total=len(
                                                                  predictions_embedding),
                                                              dynamic_ncols=True)):
        record = list()
        zero_shot_classes_embedding_per_prediction = zero_shot_classes_embedding[
            predict_index]
        for zero_shot_index, zero_shot_class_embedding in enumerate(zero_shot_classes_embedding_per_prediction):
            similarity = compute_cosine_similarity(
                prediction_embedding, zero_shot_class_embedding)

            record.append((zero_shot_index, similarity))

        records.append(record)

    return records


def compute_all_cosine_intent_string_similarity(predictions_embedding, zero_shot_classes_embedding, intent_string_similarities):
    records = list()
    for predict_index, prediction_embedding in enumerate(tqdm(predictions_embedding,
                                                              desc='Compute Cosine Similarity with Intent String Similarity',
                                                              total=len(
                                                                  predictions_embedding),
                                                              dynamic_ncols=True)):
        intent_string_similarity_per_prediction = intent_string_similarities[predict_index]
        zero_shot_classes_embedding_per_prediction = zero_shot_classes_embedding[
            predict_index]
        record = list()

        for zero_shot_index, zero_shot_class_embedding in enumerate(zero_shot_classes_embedding_per_prediction):
            intent_string_similarity = intent_string_similarity_per_prediction[zero_shot_index]
            similarity = compute_cosine_similarity(
                prediction_embedding, zero_shot_class_embedding)

            record.append(
                (zero_shot_index, intent_string_similarity + similarity))

        records.append(record)

    return records


def compute_cosine_similarity(comparing_input, compared_input):
    return 1 - scipy.spatial.distance.cosine(comparing_input, compared_input)


def find_target_from_pattern(pattern, input):
    search_result = re.search(pattern, input)
    if search_result:
        return search_result.group()


def parse_intent(completed_intent):
    splitted_completed_intent = completed_intent.split()
    if INTENT_PREFIX in splitted_completed_intent:
        start = splitted_completed_intent.index(INTENT_PREFIX) + 1
        end = None
        if DESC_PREFIX in splitted_completed_intent:
            end = splitted_completed_intent.index(DESC_PREFIX)

        return ' '.join(splitted_completed_intent[start: end])


def convert_to_normal_format(results, is_zero_shot):
    all_normal_format_results = list()
    for index, result in enumerate(results):
        splitted_result = result.strip().split('\t')
        sentence = splitted_result[0]
        original_ground_truth = splitted_result[1]
        original_prediction = splitted_result[2]

        prediction_to_parse = original_prediction
        if is_zero_shot:
            original_transferred_prediction = splitted_result[3]
            similarity = splitted_result[-1]
            prediction_to_parse = original_transferred_prediction
        parsed_ground_truth = CURRENT_INTENT_CONFIG.get(
            COMPLETED_TO_SIMPLE).get(parse_intent(original_ground_truth))
        parsed_prediction = CURRENT_INTENT_CONFIG.get(
            COMPLETED_TO_SIMPLE).get(parse_intent(prediction_to_parse))

        if parsed_prediction:
            all_normal_format_results.append(
                (parsed_ground_truth, parsed_prediction))

    return all_normal_format_results


def extend_for_one_level(part_of_entry, is_desc):
    splitted_items = part_of_entry.split()
    existed_splitted_items = list()
    format_items = list()
    detail_record = dict()

    for index, word in enumerate(splitted_items):
        synsets = wordnet.synsets(word)
        if not len(synsets):
            format_items.append(word)
        else:
            format_items.append('{}')
            existed_splitted_items.append(word)

        for syns in synsets:
            detail_record.setdefault(word, dict()).setdefault(
                POSSIBLE_VALUES, list()).extend(syns.lemma_names())

    for word, possible_values in detail_record.items():
        unique_values = sorted(set(possible_values.get(POSSIBLE_VALUES)))
        detail_record[word][POSSIBLE_VALUES] = unique_values

    set_to_product = [None] * len(existed_splitted_items)
    for index, word in enumerate(existed_splitted_items):
        possible_values = detail_record.get(word).get(POSSIBLE_VALUES)
        set_to_product[index] = possible_values

    pattern = '_'.join(format_items)
    if is_desc:
        pattern = ' '.join(format_items)
    for possible_set in itertools.product(*set_to_product):
        yield pattern.format(*possible_set)


def extend_per_entry(entry, is_desc, force_one_level):
    replaced_entry = entry.replace('_', ' ')
    splitted_entry_words_by_level = replaced_entry.split('.')

    levels = len(splitted_entry_words_by_level)
    one_level_entry = replaced_entry
    original_first_level_entry = None
    if levels > 1 and force_one_level and not is_desc:
        levels = 1
        original_first_level_entry = splitted_entry_words_by_level[0]
        original_second_level_entry = splitted_entry_words_by_level[-1]
        one_level_entry = original_second_level_entry

    if levels > 1 and not is_desc:
        level_format_items = ['{}'] * levels
        level_pattern = '.'.join(level_format_items)
        level_record = [None] * levels

        for level_index, splitted_entry_word_by_level in enumerate(splitted_entry_words_by_level):
            synsets = wordnet.synsets(splitted_entry_word_by_level)
            if not len(synsets):
                level_record[level_index] = extend_for_one_level(
                    splitted_entry_word_by_level, is_desc)
            else:
                for syns in synsets:
                    level_record[level_index] = syns.lemma_names()
        for possible_set in itertools.product(*level_record):
            yield level_pattern.format(*possible_set)

    else:
        synsets = wordnet.synsets(entry)
        if not len(synsets):
            for possible_set in extend_for_one_level(one_level_entry, is_desc):
                if force_one_level and not is_desc:
                    yield '.'.join([original_first_level_entry, possible_set]).replace(' ', '_')
                elif is_desc:
                    yield possible_set.replace('_', ' ')
                else:
                    yield possible_set
        else:
            for syns in synsets:
                for lemma_name in syns.lemma_names():
                    yield lemma_name


def generate_stsb_t5_input(sentence_one, sentence_two):
    return 'stsb sentence1: {} sentence2: {}'.format(sentence_one, sentence_two)


def clear_prefixies(string):
    return string.replace(INTENT_DETECTION_PRECIX + ' ', '').replace(SENTENCE_PREFIX + ' ', '').replace(INTENT_PREFIX + ' ', '').replace(DESC_PREFIX + ' ', '; ').strip()


def get_encoded_stsb_inputs(cleared_decoded_sentences, entities_to_compute, zero_shot_classes, max_seq_length, is_basic_transferring):
    encoded_stsb_all_inputs = list()
    for cleared_decoded_sentence, entity_to_compute in tqdm(zip(cleared_decoded_sentences, entities_to_compute),
                                                            desc='Transform to T5 STSB inputs',
                                                            total=len(
                                                                cleared_decoded_sentences),
                                                            dynamic_ncols=True):
        stsb_inputs_per_prediction = list()
        sentence_one = entity_to_compute
        if not is_basic_transferring:
            cleared_decoded_sentence = clear_prefixies(
                cleared_decoded_sentence)
            cleared_entity_to_compute = clear_prefixies(
                entity_to_compute)
            sentence_one = '"{}" {}'.format(cleared_decoded_sentence,
                                            cleared_entity_to_compute)

        for zero_shot_index, zero_shot_class in enumerate(zero_shot_classes):
            sentence_two = zero_shot_class
            if not is_basic_transferring:
                cleared_zero_shot_class = clear_prefixies(
                    zero_shot_class)
                sentence_two = '"{}" {}'.format(cleared_decoded_sentence,
                                                cleared_zero_shot_class)

            stsb_input = generate_stsb_t5_input(
                sentence_one, sentence_two)

            stsb_inputs_per_prediction.append(stsb_input)

        encoded_stsb_inputs_per_prediction = T5_TOKENIZER.batch_encode_plus(stsb_inputs_per_prediction,
                                                                            max_length=max_seq_length,
                                                                            padding=True,
                                                                            pad_to_multiple_of=max_seq_length,
                                                                            return_token_type_ids=True,
                                                                            return_tensors='tf')
        encoded_stsb_all_inputs.append(encoded_stsb_inputs_per_prediction)

    return encoded_stsb_all_inputs


def compute_all_intent_string_similarity(to_compute_entities, zero_shot_classes):
    intent_string_similarities = list()
    predictions_size = len(to_compute_entities)
    for prediction_index, to_compute_entity in enumerate(tqdm(to_compute_entities,
                                                              desc='Compute Intent String Similarity',
                                                              total=predictions_size,
                                                              dynamic_ncols=True)):
        intent_string_similaritiy = list()
        to_compute_intent = parse_intent(to_compute_entity)

        for zero_shot_index, zero_shot_class in enumerate(zero_shot_classes):
            zero_shot_intent = parse_intent(zero_shot_class)
            intent_string_similaritiy.append(
                difflib.SequenceMatcher(None, to_compute_intent, zero_shot_intent).ratio())

        intent_string_similarities.append(intent_string_similaritiy)

    return intent_string_similarities


def compute_all_stsb_similarity(encoded_stsb_all_inputs):
    records = list()
    for prediction_index, encoded_stsb_inputs_per_prediction in enumerate(tqdm(encoded_stsb_all_inputs,
                                                                               desc='Compute T5 STSB Similarity',
                                                                               total=len(
                                                                                   encoded_stsb_all_inputs),
                                                                               dynamic_ncols=True)):
        record = list()
        similarities_per_prediction = T5_STSB_TASK.generate(
            encoded_stsb_inputs_per_prediction.get('input_ids'))
        decoded_similarities_per_prediction = T5_TOKENIZER.batch_decode(
            similarities_per_prediction)
        cleared_decoded_similarities_per_prediction = list(
            map(clear_t5_decoded_useless_tokens, decoded_similarities_per_prediction))

        for zero_shot_index, cleared_decoded_similarity_per_prediction in enumerate(cleared_decoded_similarities_per_prediction):
            new_cleared_decoded_similarity = 0.0
            if is_float(cleared_decoded_similarity_per_prediction):
                new_cleared_decoded_similarity = float(
                    cleared_decoded_similarity_per_prediction) / 5

            record.append((zero_shot_index, new_cleared_decoded_similarity))

        records.append(record)

    return records


def compute_all_stsb_intent_string_similarity(encoded_stsb_all_inputs, intent_string_similarities):
    records = list()
    for prediction_index, encoded_stsb_inputs_per_prediction in enumerate(tqdm(encoded_stsb_all_inputs,
                                                                               desc='Compute T5 STSB Similarity with Intent String Similarity',
                                                                               total=len(
                                                                                   encoded_stsb_all_inputs),
                                                                               dynamic_ncols=True)):
        record = list()
        intent_string_similarity_per_prediction = intent_string_similarities[prediction_index]
        similarities_per_prediction = T5_STSB_TASK.generate(
            encoded_stsb_inputs_per_prediction.get('input_ids'))
        decoded_similarities_per_prediction = T5_TOKENIZER.batch_decode(
            similarities_per_prediction)
        cleared_decoded_similarities_per_prediction = list(
            map(clear_t5_decoded_useless_tokens, decoded_similarities_per_prediction))

        for zero_shot_index, cleared_decoded_similarity_per_prediction in enumerate(cleared_decoded_similarities_per_prediction):
            intent_string_similarity = intent_string_similarity_per_prediction[zero_shot_index]

            new_cleared_decoded_similarity = 0.0
            if is_float(cleared_decoded_similarity_per_prediction):
                new_cleared_decoded_similarity = float(
                    cleared_decoded_similarity_per_prediction) / 5

            record.append((zero_shot_index, intent_string_similarity +
                           new_cleared_decoded_similarity))

        records.append(record)

    return records


def is_float(string):
    return string.replace('.', '').isdigit()


def compute_all_mixed_similarity(src_dataset, tgt_dataset, model_type, bert_cosine_similarities, t5_stsb_similarities, intent_string_similarities):
    all_similarity_results = list()
    inference_target = '{}_to_{}'.format(src_dataset, tgt_dataset)
    inference_weights = GLOBAL_SETTING.get(model_type).get(
        'inference_hyperparameters').get(inference_target)
    bert_cosine_similarity_weight = inference_weights.get(
        'bert_cosine_similarity_weight')
    t5_stsb_similarity_weight = inference_weights.get(
        't5_stsb_similarity_weight')
    intent_string_similarity_weight = inference_weights.get(
        'intent_string_similarity_weight')

    for prediction_index, bert_cosine_similarities_per_prediction in enumerate(tqdm(bert_cosine_similarities,
                                                                                    desc='Compute Mixed Similarity',
                                                                                    total=len(
                                                                                        bert_cosine_similarities),
                                                                                    dynamic_ncols=True)):
        all_similarity_result = list()
        t5_stsb_similarities_per_prediction = t5_stsb_similarities[prediction_index]
        intent_string_similarities_per_prediction = intent_string_similarities[
            prediction_index]

        for zero_shot_item_index, (zero_shot_index, bert_cosine_similarity) in enumerate(bert_cosine_similarities_per_prediction):
            t5_stsb_similarity = t5_stsb_similarities_per_prediction[zero_shot_item_index][1]
            intent_string_similarity = intent_string_similarities_per_prediction[
                zero_shot_item_index]

            total_similarity = bert_cosine_similarity_weight * bert_cosine_similarity + t5_stsb_similarity_weight * \
                t5_stsb_similarity + intent_string_similarity_weight * intent_string_similarity

            all_similarity_result.append((zero_shot_index, total_similarity))

        all_similarity_results.append(all_similarity_result)

    return all_similarity_results


def record_all_mixed_similarity(src_daataset, tgt_dataset, bert_cosine_similarities, t5_stsb_similarities, intent_string_similarities):
    all_similarity_results = list()
    for prediction_index, bert_cosine_similarities_per_prediction in enumerate(tqdm(bert_cosine_similarities,
                                                                                    desc='Record Mixed Similarity',
                                                                                    total=len(
                                                                                        bert_cosine_similarities),
                                                                                    dynamic_ncols=True)):
        all_similarity_result = list()
        t5_stsb_similarities_per_prediction = t5_stsb_similarities[prediction_index]
        intent_string_similarities_per_prediction = intent_string_similarities[
            prediction_index]

        for zero_shot_item_index, (zero_shot_index, bert_cosine_similarity) in enumerate(bert_cosine_similarities_per_prediction):
            t5_stsb_similarity = t5_stsb_similarities_per_prediction[zero_shot_item_index][1]
            intent_string_similarity = intent_string_similarities_per_prediction[
                zero_shot_item_index]
            similarity_detail = {
                'zero_shot_index': zero_shot_index,
                'bert_cosine_similarity': bert_cosine_similarity,
                't5_stsb_similarity': t5_stsb_similarity,
                'intent_string_similarity': intent_string_similarity
            }
            all_similarity_result.append(similarity_detail)

        all_similarity_results.append(all_similarity_result)

    file_name = 'test_mixed_hyperparameters_similarities.json'
    fila_path = os.path.join(
        '/home/albeli/workspace/NCCU/Experiment/', file_name)
    data = None
    if os.path.isfile(fila_path):
        with open(fila_path, 'r') as reader:
            data = json.load(reader)
    else:
        data = dict()

    data['{}_to_{}'.format(src_daataset, tgt_dataset)
         ] = all_similarity_results

    with open(fila_path, 'w') as writer:
        json.dump(data, writer)


def load_similarities(src_dataset, tgt_dataset, bert_cosine_similarity_weight, t5_stsb_similarity_weight, intent_string_similarity_weight):
    file_name = 'test_mixed_hyperparameters_similarities.json'
    fila_path = os.path.join(
        '/home/albeli/workspace/NCCU/Experiment/', file_name)
    data = None
    with open(fila_path, 'r') as reader:
        data = json.load(reader)

    inference_target = '{}_to_{}'.format(src_dataset, tgt_dataset)
    # inference_weights = GLOBAL_SETTING.get('t5_based').get(
    #     'inference_hyperparameters').get(inference_target)
    # bert_cosine_similarity_weight = inference_weights.get(
    #     'bert_cosine_similarity_weight')
    # t5_stsb_similarity_weight = inference_weights.get(
    #     't5_stsb_similarity_weight')
    # intent_string_similarity_weight = inference_weights.get(
    #     'intent_string_similarity_weight')

    target_similarities = data.get(inference_target)

    all_similarity_results = list()
    for similarities_detail_per_prediction in tqdm(target_similarities,
                                                   desc='Comupte Mixed Similarity',
                                                   total=len(target_similarities),
                                                   dynamic_ncols=True):
        all_similarity_result = list()

        for similarity_detail in similarities_detail_per_prediction:
            zero_shot_index = similarity_detail.get('zero_shot_index')
            bert_cosine_similarity = similarity_detail.get(
                'bert_cosine_similarity')
            t5_stsb_similarity = similarity_detail.get('t5_stsb_similarity')
            intent_string_similarity = similarity_detail.get(
                'intent_string_similarity')

            total_similarity = bert_cosine_similarity_weight * bert_cosine_similarity + t5_stsb_similarity_weight * \
                t5_stsb_similarity + intent_string_similarity_weight * intent_string_similarity

            all_similarity_result.append((zero_shot_index, total_similarity))

        all_similarity_results.append(all_similarity_result)

    return all_similarity_results
