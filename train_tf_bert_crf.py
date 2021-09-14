# -*- coding: UTF-8 -*-

import argparse
import datetime
import math
import os
import sys
import time

import tensorflow as tf
from tf2crf import CRF
from transformers import BertConfig, TFBertForTokenClassification

import utility

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute_type', required=True, type=str)
    parser.add_argument('--which_dataset', required=True, type=str)
    parser.add_argument('--setting_path', required=True, type=str)
    parser.add_argument('--init_sentence_size', required=False,
                        default=math.inf, type=int)
    parser.add_argument('--epoch', required=True, type=int)

    args = parser.parse_args()
    EXECUTE_TYPE = args.execute_type
    WHICH_DATASET = args.which_dataset
    SETTING_PATH = args.setting_path
    SETTING = utility.load_setting(SETTING_PATH)
    INIT_SENTENCES_SIZE = args.init_sentence_size
    DATASETS_PATH = SETTING.get(WHICH_DATASET).get('dataset_path')
    RAW_DATA_PATH = SETTING.get(WHICH_DATASET).get('train_dataset_path')

    MODEL_TYPE = 'bert_crf'
    MAX_SEQ_LENGTH = SETTING.get(MODEL_TYPE).get('max_seq_length')
    BATCH_SIZE = SETTING.get(MODEL_TYPE).get('train_batch_size')
    LEARNING_RATE = SETTING.get(MODEL_TYPE).get('learning_rate')
    EPOCHS = args.epoch
    LABEL_LENGTH = SETTING.get(MODEL_TYPE).get('label_length')

    train_sentences, train_sentences_tags = utility.load_data(DATASETS_PATH + '{}',
                                                              RAW_DATA_PATH, EXECUTE_TYPE)
    valid_sentences, valid_sentences_tags = utility.load_data(DATASETS_PATH + '{}',
                                                              RAW_DATA_PATH, 'valid')
    tags_mapping_obj_path = os.path.join(DATASETS_PATH, 'tags_mapping.json')
    tags_mapping_obj = utility.load_tags_mapping_obj(tags_mapping_obj_path)

    if math.isfinite(INIT_SENTENCES_SIZE):
        train_sentences = train_sentences[: INIT_SENTENCES_SIZE]
        train_sentences_tags = train_sentences_tags[: INIT_SENTENCES_SIZE]
        valid_sentences = valid_sentences[: INIT_SENTENCES_SIZE]
        valid_sentences_tags = valid_sentences_tags[: INIT_SENTENCES_SIZE]

    train_tensors, train_max_token_length = utility.encode_all(MAX_SEQ_LENGTH, train_sentences,
                                                               train_sentences_tags, tags_mapping_obj, crf_mode=True)
    valid_tensors, valid_max_token_length = utility.encode_all(MAX_SEQ_LENGTH, valid_sentences,
                                                               valid_sentences_tags, tags_mapping_obj, crf_mode=True)

    config = BertConfig.from_pretrained(utility.BERT_PRE_TRAINED,
                                        num_labels=len(tags_mapping_obj.get(utility.TAG_TO_LABEL)))
    bert_model = TFBertForTokenClassification.from_pretrained(
        utility.BERT_PRE_TRAINED, config=config)
    model = utility.get_compiled_bert_crf_model(
        bert_model, MAX_SEQ_LENGTH, LEARNING_RATE)
    model.summary()

    train_x = {
        'input_ids': train_tensors.get('input_ids'),
        'attention_mask': train_tensors.get('attention_mask'),
        'token_type_ids': train_tensors.get('token_type_ids')
    }
    train_y = train_tensors.get('labels')
    train_sample_weights = train_tensors.get('sample_weights')

    valid_x = {
        'input_ids': valid_tensors.get('input_ids'),
        'attention_mask': valid_tensors.get('attention_mask'),
        'token_type_ids': valid_tensors.get('token_type_ids')
    }
    valid_y = valid_tensors.get('labels')
    valid_sample_weights = valid_tensors.get('sample_weights')

    start = time.time()
    model.fit(x=train_x,
              y=train_y,
              sample_weight=train_sample_weights,
              validation_data=(valid_x, valid_y, valid_sample_weights),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS)
    end = time.time()
    consuming_time = datetime.timedelta(seconds=end - start).__str__()
    utility.logging.info('Total training time: {}'.format(consuming_time))

    all_model_folders = os.listdir(
        SETTING.get(MODEL_TYPE).get('model_path'))
    all_model_folders.sort(reverse=True)
    current_max_folder_num = int(all_model_folders[0].split('-')[-1])
    model_saved_path = os.path.join(SETTING.get(MODEL_TYPE).get(
        'model_path'), '{}-{:02}/'.format(MODEL_TYPE, current_max_folder_num + 1))
    if not os.path.isdir(model_saved_path):
        os.makedirs(model_saved_path)
    config.save_pretrained(model_saved_path)
    model.save_weights(model_saved_path + '/tf_model.h5')
    message = '\n模型權重儲存於: {}\n總共訓練: {}'.format(
        model_saved_path, consuming_time)

    utility.line_notifier.sendMessage('苦命研究生', MODEL_TYPE.upper(), message)
