# -*- coding: UTF-8 -*-

import utility
from utility import argparse, os, random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_path', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str)
    parser.add_argument('--which_dataset', required=True, type=str)
    parser.add_argument('--is_backup', required=False, default=None, type=str)

    args = parser.parse_args()
    SETTING_PATH = args.setting_path
    SETTING = utility.load_setting(SETTING_PATH)
    DATA_TYPE = args.data_type
    WHICH_DATASET = args.which_dataset
    IS_BACKUP = args.is_backup
    # SHUFFLE_TIMES = args.shuffle_times

    ROOT_DATASETS_PATH = SETTING.get('dataset_path')
    DATA_COUNT_PATH = ROOT_DATASETS_PATH.get(
        '{}_dataset_intent_count_path'.format(DATA_TYPE)).format(WHICH_DATASET)
    TARGET_DATA_PATH = ROOT_DATASETS_PATH.get(
        '{}_dataset_path'.format(DATA_TYPE)).format(WHICH_DATASET)

    if IS_BACKUP:
        DATA_COUNT_PATH = ROOT_DATASETS_PATH.get(
            '{}_dataset_backup_intent_count_path'.format(DATA_TYPE)).format(WHICH_DATASET)
        TARGET_DATA_PATH = ROOT_DATASETS_PATH.get(
            '{}_dataset_backup_path'.format(DATA_TYPE)).format(WHICH_DATASET)

    # TARGET_DATA_BACKUP_PATH = ROOT_DATASETS_PATH.get(
    #     '{}_dataset_backup_path'.format(DATA_TYPE)).format(WHICH_DATASET)
    # DATASET_REDUCTION = SETTING.get('dataset_reduction')

    all_data = list()
    with open(TARGET_DATA_PATH, 'r') as reader:
        for line in reader:
            all_data.append(tuple(line.strip().split('\t')))

    all_intent = set()
    intent_data = dict()
    for data in all_data:
        intent = data[-1]
        intent_data.setdefault(intent, dict())
        intent_data_obj = intent_data.setdefault(intent, dict())
        intent_data_obj.setdefault('data', list()).append(data)
        if intent_data_obj.get('data_count'):
            intent_data_obj['data_count'] += 1
        else:
            intent_data_obj['data_count'] = 1

    data_to_record = list()
    for intent, intent_data_obj in intent_data.items():
        data_to_record.append((intent, intent_data_obj.get('data_count')))
    data_to_record.sort(key=lambda item: item[0])

    with open(DATA_COUNT_PATH, 'w') as writer:
        for intent, data_count in data_to_record:
            writer.write(
                '\t'.join([intent, '{}'.format(data_count)]) + os.linesep)

    # sample_size_by_intent = DATASET_REDUCTION.get(WHICH_DATASET).get(DATA_TYPE)

    # sampled_data = list()
    # for itent, intent_data_obj in intent_data.items():
    #     new_sample_size_by_intent = sample_size_by_intent
    #     if intent_data_obj.get('data_count') < sample_size_by_intent:
    #         new_sample_size_by_intent = intent_data_obj.get('data_count')

        # samples = random.sample(intent_data_obj.get(
        #     'data'), new_sample_size_by_intent)
        # sampled_data.extend(samples)

    # for times in range(SHUFFLE_TIMES):
    #     random.shuffle(sampled_data)

    # with open(TARGET_DATA_PATH, 'w') as writer:
    #     for sample_data in sampled_data:
    #         writer.write('\t'.join(sample_data) + os.linesep)
