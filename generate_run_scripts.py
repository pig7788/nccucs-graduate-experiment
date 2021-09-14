# -*- coding: UTF-8 -*-

from utility import argparse, os


def generate_zero_shot_run_scripts(inference_method, given_model_path_prefix):
    augment_methods = ['baseline', 'back_translation', 'synonym_set', 'mixed']
    extended_timeses = ['extended_5', 'extended_10']
    datasets = ['snips', 'atis', 'multiwoz', 'sgd']
    run_script_pattern = 'sh run_predict_convert_eval.sh intent_detection {} {} t5_based test None t5 True True {} {}'
    if not inference_method:
        run_script_pattern = 'sh run_predict_convert_eval.sh intent_detection {} {} t5_based test None t5 True False {} {}'
    unextenced_transfer_pattern = '{} -> {} to {}'
    extended_transfer_pattern = '{} -> {} to {} for {} {}'
    unextended_log_record_script_pattern = 'echo "{} -> {} to {}: done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'
    extended_log_record_script_pattern = 'echo "{} -> {} to {} for {} {}: done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'

    with open('run_zero_shot_predict_eval_scripts_{}.sh'.format(inference_method), 'w') as writer:
        writer.write('#!/bin/bash' + os.linesep * 2)
        writer.write('rm -f running.log' + os.linesep * 2)
        for augment_method in augment_methods:
            if augment_method == 'baseline':
                baseline_comment = '# ' + unextenced_transfer_pattern
                for src_dataset in datasets:
                    for tgt_dataset in datasets:
                        if src_dataset != tgt_dataset:
                            run_script = run_script_pattern.format(src_dataset, tgt_dataset, inference_method, os.path.join(
                                given_model_path_prefix, augment_method, src_dataset))
                            unextended_log_record_script = unextended_log_record_script_pattern.format(
                                inference_method, src_dataset, tgt_dataset)

                            writer.write(baseline_comment.format(
                                inference_method, src_dataset, tgt_dataset) + os.linesep)
                            writer.write(run_script + os.linesep)
                            writer.write(
                                unextended_log_record_script + os.linesep)
            else:
                extended_comment = '# ' + extended_transfer_pattern
                for extended_times in extended_timeses:
                    for src_dataset in datasets:
                        for tgt_dataset in datasets:
                            if src_dataset != tgt_dataset:
                                run_script = run_script_pattern.format(src_dataset, tgt_dataset, inference_method, os.path.join(
                                    given_model_path_prefix, augment_method, extended_times, src_dataset))
                                extended_log_record_script = extended_log_record_script_pattern.format(
                                    inference_method, src_dataset, tgt_dataset, extended_times, augment_method)

                                writer.write(extended_comment.format(
                                    inference_method, src_dataset, tgt_dataset, extended_times, augment_method) + os.linesep)
                                writer.write(run_script + os.linesep)
                                writer.write(
                                    extended_log_record_script + os.linesep)


def generate_zero_shot_run_scripts_test_mixed_hyperparameters(inference_method, given_model_path_prefix):
    augment_methods = ['baseline']
    extended_timeses = ['extended_5', 'extended_10']
    datasets = ['snips', 'atis', 'multiwoz', 'sgd']
    run_script_pattern = 'sh run_predict_convert_eval_test_mixed_hyperparameters.sh intent_detection {} {} t5_based test None t5 True True {} {} {} {} {}'
    if not inference_method:
        run_script_pattern = 'sh run_predict_convert_eval_test_mixed_hyperparameters.sh intent_detection {} {} t5_based test None t5 True False {} {} {} {} {}'
    unextenced_transfer_pattern = '{} -> {} to {} bert:{} t5:{} string:{}'
    extended_transfer_pattern = '{} -> {} to {} for {} {} bert:{} t5:{} string:{}'
    unextended_log_record_script_pattern = 'echo "{} -> {} to {} bert:{} t5:{} string:{} : done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'
    extended_log_record_script_pattern = 'echo "{} -> {} to {} for {} {} bert:{} t5:{} string:{} : done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'

    with open('run_zero_shot_predict_eval_scripts_{}_test_mixed_hyperparameters.sh'.format(inference_method), 'w') as writer:
        writer.write('#!/bin/bash' + os.linesep * 2)
        writer.write('rm -f running.log' + os.linesep * 2)
        for augment_method in augment_methods:
            if augment_method == 'baseline':
                baseline_comment = '# ' + unextenced_transfer_pattern
                for src_dataset in datasets:
                    for tgt_dataset in datasets:
                        if src_dataset != tgt_dataset:
                            for lambda_bert in range(11):
                                for lambda_t5 in range(11):
                                    if lambda_bert + lambda_t5 <= 10:
                                        lambda_string = 10 - lambda_bert - lambda_t5
                                        hyperparameter = (
                                            lambda_bert / 10, lambda_t5 / 10, lambda_string / 10)
                                        run_script = run_script_pattern.format(src_dataset, tgt_dataset, inference_method, os.path.join(
                                            given_model_path_prefix, augment_method, src_dataset), hyperparameter[0], hyperparameter[1], hyperparameter[2])
                                        unextended_log_record_script = unextended_log_record_script_pattern.format(
                                            inference_method, src_dataset, tgt_dataset, hyperparameter[0], hyperparameter[1], hyperparameter[2])

                                        writer.write(baseline_comment.format(
                                            inference_method, src_dataset, tgt_dataset, hyperparameter[0], hyperparameter[1], hyperparameter[2]) + os.linesep)
                                        writer.write(run_script + os.linesep)
                                        writer.write(
                                            unextended_log_record_script + os.linesep)
            else:
                extended_comment = '# ' + extended_transfer_pattern
                for extended_times in extended_timeses:
                    for src_dataset in datasets:
                        for tgt_dataset in datasets:
                            if src_dataset != tgt_dataset:
                                for lambda_bert in range(11):
                                    for lambda_t5 in range(11):
                                        if lambda_bert + lambda_t5 <= 10:
                                            lambda_string = 10 - lambda_bert - lambda_t5
                                            hyperparameter = (
                                                lambda_bert / 10, lambda_t5 / 10, lambda_string / 10)
                                            run_script = run_script_pattern.format(src_dataset, tgt_dataset, inference_method, os.path.join(
                                                given_model_path_prefix, augment_method, extended_times, src_dataset), hyperparameter[0], hyperparameter[1], hyperparameter[2])
                                            extended_log_record_script = extended_log_record_script_pattern.format(
                                                inference_method, src_dataset, tgt_dataset, extended_times, augment_method, hyperparameter[0], hyperparameter[1], hyperparameter[2])

                                            writer.write(extended_comment.format(
                                                inference_method, src_dataset, tgt_dataset, extended_times, augment_method, hyperparameter[0], hyperparameter[1], hyperparameter[2]) + os.linesep)
                                            writer.write(
                                                run_script + os.linesep)
                                            writer.write(
                                                extended_log_record_script + os.linesep)


def generate_in_domain_run_scripts(inference_method, given_model_path_prefix):
    augment_methods = ['baseline', 'back_translation', 'synonym_set', 'mixed']
    extended_timeses = ['extended_5', 'extended_10']
    datasets = ['snips', 'atis', 'multiwoz', 'sgd']
    run_script_pattern = 'sh run_predict_convert_eval.sh intent_detection {} {} t5_based test None t5 True True {} {}'
    if not inference_method:
        run_script_pattern = 'sh run_predict_convert_eval.sh intent_detection {} {} t5_based test None t5 True False {} {}'
    unextenced_transfer_pattern = '{} -> {} to {}'
    extended_transfer_pattern = '{} -> {} to {} for {} {}'
    unextended_log_record_script_pattern = 'echo "{} -> {} to {}: done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'
    extended_log_record_script_pattern = 'echo "{} -> {} to {} for {} {}: done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'
    with open('run_in_domain_predict_eval_scripts_{}.sh'.format(inference_method), 'w') as writer:
        writer.write('#!/bin/bash' + os.linesep * 2)
        writer.write('rm -f running.log' + os.linesep * 2)
        for augment_method in augment_methods:
            if augment_method == 'baseline':
                baseline_comment = '# ' + unextenced_transfer_pattern
                for src_dataset in datasets:
                    run_script = run_script_pattern.format(src_dataset, src_dataset, inference_method, os.path.join(
                        given_model_path_prefix, augment_method, src_dataset))
                    unextended_log_record_script = unextended_log_record_script_pattern.format(
                        inference_method, src_dataset, src_dataset)

                    writer.write(baseline_comment.format(
                        inference_method, src_dataset, src_dataset) + os.linesep)
                    writer.write(run_script + os.linesep)
                    writer.write(
                        unextended_log_record_script + os.linesep)
            else:
                extended_comment = '# ' + extended_transfer_pattern
                for extended_times in extended_timeses:
                    for src_dataset in datasets:
                        run_script = run_script_pattern.format(src_dataset, src_dataset, inference_method, os.path.join(
                            given_model_path_prefix, augment_method, extended_times, src_dataset))
                        extended_log_record_script = extended_log_record_script_pattern.format(
                            inference_method, src_dataset, src_dataset, extended_times, augment_method)

                        writer.write(extended_comment.format(
                            inference_method, src_dataset, src_dataset, extended_times, augment_method) + os.linesep)
                        writer.write(run_script + os.linesep)
                        writer.write(
                            extended_log_record_script + os.linesep)


def generate_in_domain_run_scripts_test_mixed_hyperparameters(inference_method, given_model_path_prefix):
    augment_methods = ['baseline']
    extended_timeses = ['extended_5', 'extended_10']
    datasets = ['snips', 'atis', 'multiwoz', 'sgd']
    run_script_pattern = 'sh run_predict_convert_eval_test_mixed_hyperparameters.sh intent_detection {} {} t5_based test None t5 True True {} {} {} {} {}'
    if not inference_method:
        run_script_pattern = 'sh run_predict_convert_eval_test_mixed_hyperparameters.sh intent_detection {} {} t5_based test None t5 True False {} {} {} {} {}'
    unextenced_transfer_pattern = '{} -> {} to {} bert:{} t5:{} string:{}'
    extended_transfer_pattern = '{} -> {} to {} for {} {} bert:{} t5:{} string:{}'
    unextended_log_record_script_pattern = 'echo "{} -> {} to {} bert:{} t5:{} string:{} : done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'
    extended_log_record_script_pattern = 'echo "{} -> {} to {} for {} {} bert:{} t5:{} string:{} : done at $(date "+%Y-%m-%d %H:%M:%S")" >> running.log'
    with open('run_in_domain_predict_eval_scripts_{}_test_mixed_hyperparameters.sh'.format(inference_method), 'w') as writer:
        writer.write('#!/bin/bash' + os.linesep * 2)
        writer.write('rm -f running.log' + os.linesep * 2)
        for augment_method in augment_methods:
            if augment_method == 'baseline':
                baseline_comment = '# ' + unextenced_transfer_pattern
                for src_dataset in datasets:
                    for lambda_bert in range(11):
                        for lambda_t5 in range(11):
                            if lambda_bert + lambda_t5 <= 10:
                                lambda_string = 10 - lambda_bert - lambda_t5
                                hyperparameter = (
                                    lambda_bert / 10, lambda_t5 / 10, lambda_string / 10)
                                run_script = run_script_pattern.format(src_dataset, src_dataset, inference_method, os.path.join(
                                    given_model_path_prefix, augment_method, src_dataset), hyperparameter[0], hyperparameter[1], hyperparameter[2])
                                unextended_log_record_script = unextended_log_record_script_pattern.format(
                                    inference_method, src_dataset, src_dataset, hyperparameter[0], hyperparameter[1], hyperparameter[2])

                                writer.write(baseline_comment.format(
                                    inference_method, src_dataset, src_dataset, hyperparameter[0], hyperparameter[1], hyperparameter[2]) + os.linesep)
                                writer.write(run_script + os.linesep)
                                writer.write(
                                    unextended_log_record_script + os.linesep)
            else:
                extended_comment = '# ' + extended_transfer_pattern
                for extended_times in extended_timeses:
                    for src_dataset in datasets:
                        for lambda_bert in range(11):
                            for lambda_t5 in range(11):
                                if lambda_bert + lambda_t5 <= 10:
                                    lambda_string = 10 - lambda_bert - lambda_t5
                                    hyperparameter = (
                                        lambda_bert / 10, lambda_t5 / 10, lambda_string / 10)
                                    run_script = run_script_pattern.format(src_dataset, src_dataset, inference_method, os.path.join(
                                        given_model_path_prefix, augment_method, extended_times, src_dataset), hyperparameter[0], hyperparameter[1], hyperparameter[2])
                                    extended_log_record_script = extended_log_record_script_pattern.format(
                                        inference_method, src_dataset, src_dataset, extended_times, augment_method, hyperparameter[0], hyperparameter[1], hyperparameter[2])

                                    writer.write(extended_comment.format(
                                        inference_method, src_dataset, src_dataset, extended_times, augment_method, hyperparameter[0], hyperparameter[1], hyperparameter[2]) + os.linesep)
                                    writer.write(run_script + os.linesep)
                                    writer.write(
                                        extended_log_record_script + os.linesep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_method', required=True, type=str)
    parser.add_argument('--given_model_path_prefix', required=True, type=str)
    parser.add_argument('--test_mixed_hyperparameters',
                        required=False, default=None, type=str)

    args = parser.parse_args()

    INFERENCE_METHOD = args.inference_method
    if args.inference_method == 'None':
        INFERENCE_METHOD = None

    GIVEN_MODEL_PATH_PREFIX = args.given_model_path_prefix

    TEST_HYPERPARAMETERS = args.test_mixed_hyperparameters
    if TEST_HYPERPARAMETERS:
        generate_in_domain_run_scripts_test_mixed_hyperparameters(
            INFERENCE_METHOD, GIVEN_MODEL_PATH_PREFIX)
        generate_zero_shot_run_scripts_test_mixed_hyperparameters(
            INFERENCE_METHOD, GIVEN_MODEL_PATH_PREFIX)
    else:
        generate_in_domain_run_scripts(
            INFERENCE_METHOD, GIVEN_MODEL_PATH_PREFIX)
        generate_zero_shot_run_scripts(
            INFERENCE_METHOD, GIVEN_MODEL_PATH_PREFIX)
