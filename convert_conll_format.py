# -*- coding: UTF-8 -*-

import utility
from utility import (argparse, datetime, line_notifier, math, os, sys, time,
                     traceback)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute_type', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str)
    parser.add_argument('--tgt_dataset', required=True, type=str)
    parser.add_argument('--setting_path', required=True, type=str)
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--selected_model', required=True, type=str)
    parser.add_argument('--convert_from', required=True, type=str)
    parser.add_argument('--is_zero_shot', required=True, type=str)
    parser.add_argument('--given_model_path',
                        required=False, default=None, type=str)

    args = parser.parse_args()
    EXECUTE_TYPE = args.execute_type
    DATA_TYPE = args.data_type
    TGT_DATASET = args.tgt_dataset
    SETTING_PATH = args.setting_path
    utility.set_global_setting(SETTING_PATH)
    SETTING = utility.GLOBAL_SETTING
    utility.set_scheme_config(SETTING.get('scheme_config_path'))
    utility.set_entity_config(SETTING.get('entity_config_path'))
    utility.set_intent_config(SETTING.get('intent_config_path'))

    SELECTED_MODEL = args.selected_model
    MODEL_TYPE = args.model_type
    MODEL_PATH = os.path.join(SETTING.get(
        MODEL_TYPE).get('model_path'), SELECTED_MODEL)
    GIVEN_MODEL_PATH = args.given_model_path
    CONVERT_FROM = args.convert_from

    LOAD_MODEL_PATH = MODEL_PATH
    if GIVEN_MODEL_PATH:
        LOAD_MODEL_PATH = GIVEN_MODEL_PATH

    PREDICT_RESULT_PATH = os.path.join(
        LOAD_MODEL_PATH, '{}_{}_{}_results.txt'.format(TGT_DATASET, EXECUTE_TYPE, DATA_TYPE))
    CONVERT_RESULT_PATH = os.path.join(
        LOAD_MODEL_PATH, '{}_{}_{}_results_conll_format.txt'.format(TGT_DATASET, EXECUTE_TYPE, DATA_TYPE))

    IS_ZERO_SHOT = utility.strtobool(args.is_zero_shot)
    utility.set_current_config('BIO', TGT_DATASET)

    conll_format_results = list()
    error_records = list()
    with open(PREDICT_RESULT_PATH, 'r') as reader:
        for index, line in enumerate(reader, 1):
            line_strip_splitor = line.strip().split('\t')
            if CONVERT_FROM == 't5':
                try:
                    conll_format_results.extend(
                        list(map(lambda item: utility.change_t5_to_conll_eval_format(item, IS_ZERO_SHOT), [line_strip_splitor])))
                except:
                    message = '第{}筆資料有問題:\n{}\nexception:\n{}'.format(
                        index, line_strip_splitor, traceback.format_exc())
                    error_records.append(message + os.linesep)
                    utility.logging.info(message)
            elif CONVERT_FROM == 'bert':
                conll_format_results.extend(
                    utility.change_bert_to_conll_eval_format(line_strip_splitor))

    with open(CONVERT_RESULT_PATH, 'w') as writer:
        writer.writelines(conll_format_results)

    error_count = len(error_records)
    error_records_path = os.path.join(MODEL_PATH, 'convert_errors.record')
    error_message = ''
    if error_count:
        with open(error_records_path, 'w') as writer:
            writer.writelines(error_records)
        error_message = '\n發現轉換錯誤: {}\n轉換錯誤記錄寫入於: {}'.format(
            error_count, error_records_path)
    result_count = len(conll_format_results)
    utility.logging.info('已寫入{}筆測試結果'.format(result_count))
    message = '\n已寫入測試結果: {}\n轉換結果寫入於:{}{}'.format(
        result_count, CONVERT_RESULT_PATH, error_message)
    line_notifier.sendMessage('苦命研究生', MODEL_TYPE.upper(), message)
