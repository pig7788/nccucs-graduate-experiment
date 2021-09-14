# -*- coding: UTF-8 -*-


import utility
from utility import (accuracy_score, argparse, ast, classification_report,
                     confusion_matrix, f1_score, numpy, os, pandas,
                     precision_score, recall_score)

if __name__ == '__main__':
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
    parser.add_argument('--encoder_type', required=True, type=str)

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
    ENCODER_TYPE = args.encoder_type

    LOAD_MODEL_PATH = MODEL_PATH
    if GIVEN_MODEL_PATH:
        LOAD_MODEL_PATH = GIVEN_MODEL_PATH

    RAW_RESULTS_FILE_NAME = '{}_{}_{}_{}_results.txt'.format(
        TGT_DATASET, EXECUTE_TYPE, DATA_TYPE, ENCODER_TYPE)
    RAW_RESULTS_NORMAL_FORMAT_FILE_NAME = '{}_{}_{}_{}_results_normal_format.txt'.format(
        TGT_DATASET, EXECUTE_TYPE, DATA_TYPE, ENCODER_TYPE)
    FINAL_RESULTS_FILE_NAME = '{}_{}_{}_{}_final_results.txt'.format(
        TGT_DATASET, EXECUTE_TYPE, DATA_TYPE, ENCODER_TYPE)
    CONFUSION_MATRIX_EXCEL_FILE_NAME = '{}_{}_{}_{}_confusion_matrix.xlsx'.format(
        TGT_DATASET, EXECUTE_TYPE, DATA_TYPE, ENCODER_TYPE)
    CONFUSION_MATRIX_CSV_FILE_NAME = '{}_{}_{}_{}_confusion_matrix.csv'.format(
        TGT_DATASET, EXECUTE_TYPE, DATA_TYPE, ENCODER_TYPE)

    PREDICT_RESULT_PATH = os.path.join(
        LOAD_MODEL_PATH, RAW_RESULTS_FILE_NAME)
    CONVERT_RESULT_PATH = os.path.join(
        LOAD_MODEL_PATH, RAW_RESULTS_NORMAL_FORMAT_FILE_NAME)
    FINAL_RESULT_PATH = os.path.join(
        LOAD_MODEL_PATH, FINAL_RESULTS_FILE_NAME)
    CONFUSION_MATRIX_EXCEL_PATH = os.path.join(
        LOAD_MODEL_PATH, CONFUSION_MATRIX_EXCEL_FILE_NAME)
    CONFUSION_MATRIX_CSV_PATH = os.path.join(
        LOAD_MODEL_PATH, CONFUSION_MATRIX_CSV_FILE_NAME)

    IS_ZERO_SHOT = utility.strtobool(args.is_zero_shot)
    utility.set_current_config('BIO', TGT_DATASET)

    pandas.set_option('display.max_rows', None, 'display.max_columns', None)

    all_normal_format_results = list()
    with open(PREDICT_RESULT_PATH, 'r') as reader:
        all_normal_format_results.extend(
            utility.convert_to_normal_format(reader, IS_ZERO_SHOT))

    with open(CONVERT_RESULT_PATH, 'w') as writer:
        for normal_format_result in all_normal_format_results:
            writer.write('\t'.join(normal_format_result) + os.linesep)

    ground_truthes = list()
    predictions = list()
    score_pattern = '{}: {:.4%}{}'
    with open(FINAL_RESULT_PATH, 'w') as writer:
        for ground_truth, prediction in all_normal_format_results:
            ground_truthes.append(ground_truth)
            predictions.append(prediction)

        target_names = sorted(set.union(set(ground_truthes), set(predictions)))
        raw_confusion_matrix = confusion_matrix(ground_truthes, predictions)
        new_confusion_matrix = numpy.empty(
            (raw_confusion_matrix.shape[0], raw_confusion_matrix.shape[-1] + 1), dtype=numpy.int32)
        for row_index, row in enumerate(raw_confusion_matrix):
            for col_index, col in enumerate(row):
                new_confusion_matrix[row_index][col_index] = col
            new_confusion_matrix[row_index][-1] = numpy.sum(row)

        confustionMatrix = pandas.DataFrame(
            new_confusion_matrix, columns=target_names + ['Total'])
        confustionMatrix.index = target_names

        writer.write(classification_report(
            ground_truthes, predictions) + os.linesep)
        confustionMatrix.to_excel(CONFUSION_MATRIX_EXCEL_PATH)
        CONFUSION_MATRIX_EXCEL_PATH
        confustionMatrix.to_csv(CONFUSION_MATRIX_CSV_PATH)
        # writer.write('Confustion Matrix:\n{}'.format(
        #     confustionMatrix) + os.linesep * 2)
        writer.write(score_pattern.format('Accuracy', accuracy_score(
            ground_truthes, predictions), os.linesep))
        writer.write(score_pattern.format('Macro Precision', precision_score(
            ground_truthes, predictions, average='macro'), os.linesep))
        writer.write(score_pattern.format('Macro Recall', recall_score(
            ground_truthes, predictions, average='macro'), os.linesep))
        writer.write(score_pattern.format('Macro F1', f1_score(
            ground_truthes, predictions, average='macro'), os.linesep))
        writer.write(score_pattern.format('Micro Precision', precision_score(
            ground_truthes, predictions, average='micro'), os.linesep))
        writer.write(score_pattern.format('Micro Recall', recall_score(
            ground_truthes, predictions, average='micro'), os.linesep))
        writer.write(score_pattern.format('Micro F1', f1_score(
            ground_truthes, predictions, average='micro'), os.linesep))
