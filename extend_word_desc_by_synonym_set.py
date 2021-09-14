# -*- coding: UTF-8 -*-

import utility
from utility import argparse, os, strtobool

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_path', required=True, type=str)
    parser.add_argument('--execute_type', required=True, type=str)
    parser.add_argument('--which_dataset', required=True, type=str)
    parser.add_argument('--with_desc', required=True, type=str)

    args = parser.parse_args()
    SETTING_PATH = args.setting_path
    SETTING = utility.load_setting(SETTING_PATH)
    EXECUTE_TYPE = args.execute_type
    WHICH_DATASET = args.which_dataset
    WITH_DESC = strtobool(args.with_desc)
    CONFIG = None

    if EXECUTE_TYPE == utility.SLOT_FILLING:
        utility.set_entity_config(SETTING.get('entity_config_path'))
        CONFIG = utility.ENTITY_CONFIG
    elif EXECUTE_TYPE == utility.INTENTION:
        utility.set_intent_config(SETTING.get('intent_config_path'))
        CONFIG = utility.INTENT_CONFIG

    file_name = './extend/{}_{}_synonym_set_extend.txt'.format(
        EXECUTE_TYPE, WHICH_DATASET)
    desc_file_name = './extend/{}_{}_desc_synonym_set_extend.txt'.format(
        EXECUTE_TYPE, WHICH_DATASET)

    force_one_level = EXECUTE_TYPE == utility.SLOT_FILLING and WHICH_DATASET in [
        'multiwoz', 'sgd']

    for entry, desc in CONFIG.get(WHICH_DATASET).get(utility.DESC).items():
        if entry != utility.NON_ENTITY_SCHEME:
            completed_entry = CONFIG.get(WHICH_DATASET).get(
                utility.SIMPLE_TO_COMPLETED).get(entry)
            if isinstance(completed_entry, list):
                for item in completed_entry:
                    for extended_entry in utility.extend_per_entry(item, False, force_one_level):
                        with open(file_name, 'a') as writer:
                            writer.write('{}|{}|{}|{}'.format(
                                WHICH_DATASET, utility.LEMMA_NAMES, entry, extended_entry.lower()) + os.linesep)
            else:
                for extended_entry in utility.extend_per_entry(completed_entry, False, force_one_level):
                    with open(file_name, 'a') as writer:
                        writer.write('{}|{}|{}|{}'.format(
                            WHICH_DATASET, utility.LEMMA_NAMES, entry, extended_entry.lower()) + os.linesep)

            if WITH_DESC:
                for extended_desc in utility.extend_per_entry(desc, True, force_one_level):
                    with open(desc_file_name, 'a') as writer:
                        writer.write('{}|{}|{}|{}'.format(
                            WHICH_DATASET, utility.DESC, entry, extended_desc) + os.linesep)
