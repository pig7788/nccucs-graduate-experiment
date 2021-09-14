# -*- coding: UTF-8 -*-

from transformers import T5Config, TFT5ForConditionalGeneration

import utility
from utility import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute_type', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str)
    parser.add_argument('--which_dataset', required=True, type=str)
    parser.add_argument('--setting_path', required=True, type=str)
    parser.add_argument('--with_desc', required=True, type=str)
    parser.add_argument('--augment_src',
                        required=False, type=str)
    parser.add_argument('--prob_threshold',
                        required=False, default=1, type=float)
    parser.add_argument('--extend_repeat',
                        required=False, default=0, type=int)
    parser.add_argument('--shuffle_times',
                        required=False, default=0, type=int)
    parser.add_argument('--data_src_path', required=False,
                        default=None, type=str)
    parser.add_argument('--co_train', required=False,
                        default=None, type=str)

    args = parser.parse_args()
    WHICH_DATASET = args.which_dataset
    SETTING_PATH = args.setting_path
    SETTING = utility.load_setting(SETTING_PATH)
    utility.set_scheme_config(SETTING.get('scheme_config_path'))
    utility.set_entity_config(SETTING.get('entity_config_path'))
    utility.set_intent_config(SETTING.get('intent_config_path'))
    EXECUTE_TYPE = args.execute_type
    DATA_TYPE = args.data_type
    AUGMENTATION_SRC = args.augment_src.split(
        ',') if args.augment_src else list()
    CO_TRAIN = args.co_train.split(
        ',') if args.co_train else list()
    AUGMENTATION_SRC_FILE_NAME = '_'.join(AUGMENTATION_SRC)
    CO_TRAIN_FILE_NAME = '_'.join(
        ['co_train'] + CO_TRAIN) if len(CO_TRAIN) else 'none'

    DATA_SRC_PATH = args.data_src_path
    ROOT_DATASETS_PATH = SETTING.get('dataset_path')
    TEST_DATA_PATH = ROOT_DATASETS_PATH.get(
        '{}_dataset_path'.format(DATA_TYPE)).format(WHICH_DATASET)
    EXTENDED_DATA_PATH = ROOT_DATASETS_PATH.get('extended_dataset_path')
    T5_DATA_PATH = ROOT_DATASETS_PATH.get(
        't5_format_dataset_path').format(WHICH_DATASET, EXECUTE_TYPE, DATA_TYPE, CO_TRAIN_FILE_NAME)

    WITH_DESC = utility.strtobool(args.with_desc)
    PROB_THRESHOLD = args.prob_threshold
    EXTEND_REPEAT = args.extend_repeat
    SHUFFLE_TIMES = args.shuffle_times
    CO_TRAIN_INFO = list()
    for co_train_type in CO_TRAIN:
        path = ROOT_DATASETS_PATH.get(
            '{}_dataset_path'.format(DATA_TYPE)).format(co_train_type)
        CO_TRAIN_INFO.append((path, co_train_type))
    COMPLETED_EXTENDED_DATA_PATH = EXTENDED_DATA_PATH.format(
        WHICH_DATASET,
        EXECUTE_TYPE,
        AUGMENTATION_SRC_FILE_NAME,
        CO_TRAIN_FILE_NAME,
        PROB_THRESHOLD,
        EXTEND_REPEAT,
        WITH_DESC)

    utility.set_current_config('BIO', WHICH_DATASET)

    RAW_DATA_PATH = TEST_DATA_PATH
    if DATA_SRC_PATH:
        RAW_DATA_PATH = DATA_SRC_PATH
    train_sentences, train_sentences_tags, train_intents = utility.load_all_data(
        RAW_DATA_PATH)

    DATA_PATH = None
    if len(AUGMENTATION_SRC):
        DATA_PATH = COMPLETED_EXTENDED_DATA_PATH
    else:
        DATA_PATH = T5_DATA_PATH

    utility.logging.info('正在執行產生資料作業⋯⋯')
    t5_format_data_generator = utility.get_dataset_by_condition(DATA_PATH,
                                                                train_sentences,
                                                                train_sentences_tags,
                                                                train_intents,
                                                                WITH_DESC,
                                                                EXECUTE_TYPE,
                                                                CO_TRAIN_INFO,
                                                                AUGMENTATION_SRC,
                                                                PROB_THRESHOLD,
                                                                EXTEND_REPEAT,
                                                                SHUFFLE_TIMES,
                                                                False)

    for item in t5_format_data_generator:
        pass
    utility.logging.info('已完成執行產生資料作業⋯⋯')
