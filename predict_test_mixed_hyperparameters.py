# -*- coding: UTF-8 -*-


import utility
from utility import (CRF, BertConfig, T5Config, TFBertForTokenClassification,
                     TFT5ForConditionalGeneration, argparse, ast, datetime,
                     math, os, sys, tf, time, timedelta, tqdm)


def predict_by_t5_based(tensors, encoder_type, src_dataset, tgt_dataset, predicted_file_name):
    start = time.time()
    all_cleared_decoded_sentences = list()
    all_cleared_decoded_anwsers = list()
    all_cleared_decoded_predictions = list()

    if not IS_PREDICTED_FILE_EXISTED:
        datasets, datasets_total_length = utility.get_tensor_datasets(
            tensors, BATCH_SIZE)

        config = T5Config.from_json_file(
            os.path.join(LOAD_MODEL_PATH, 'config.json'))
        model = TFT5ForConditionalGeneration.from_pretrained(
            os.path.join(LOAD_MODEL_PATH, 'tf_model.h5'), config=config)
        model.summary()

        for batch_index, batch_inputs in enumerate(tqdm(datasets,
                                                        desc='Predicting',
                                                        total=datasets_total_length), 1):
            predictions = model.generate(batch_inputs.get('input_ids'))
            decoded_sentences = utility.T5_TOKENIZER.batch_decode(
                batch_inputs.get('input_ids'))
            decoded_anwsers = utility.T5_TOKENIZER.batch_decode(utility.change_label_weight(batch_inputs.get(
                'labels'), utility.NON_WEIGHTED_COMPUTED_LABEL_ID, utility.T5_TOKENIZER.pad_token_id))
            decoded_predictions = utility.T5_TOKENIZER.batch_decode(
                predictions)

            cleared_decoded_sentences = list(
                map(utility.clear_t5_decoded_useless_tokens, decoded_sentences))
            cleared_decoded_anwsers = list(map(
                utility.clear_t5_decoded_useless_tokens, decoded_anwsers))
            cleared_decoded_predictions = list(
                map(utility.clear_t5_decoded_useless_tokens, decoded_predictions))

            all_cleared_decoded_sentences.extend(cleared_decoded_sentences)
            all_cleared_decoded_anwsers.extend(cleared_decoded_anwsers)
            all_cleared_decoded_predictions.extend(cleared_decoded_predictions)

            utility.logging.info(
                '目前已經預測{}筆測試資料'.format(batch_index * BATCH_SIZE))

        with open(predicted_result_path, 'w') as writer:
            for cleared_decoded_sentence, cleared_decoded_anwser, cleared_decoded_prediction in zip(all_cleared_decoded_sentences, all_cleared_decoded_anwsers, all_cleared_decoded_predictions):
                writer.write('\t'.join(
                    [cleared_decoded_sentence, cleared_decoded_anwser, cleared_decoded_prediction]) + os.linesep)

    all_cleared_decoded_sentences.clear()
    all_cleared_decoded_anwsers.clear()
    all_cleared_decoded_predictions.clear()
    with open(predicted_result_path, 'r') as reader:
        for line in reader:
            splitted_items = line.strip().split('\t')
            cleared_decoded_sentence = splitted_items[0]
            cleared_decoded_anwser = splitted_items[1]
            cleared_decoded_prediction = splitted_items[2]

            all_cleared_decoded_sentences.append(cleared_decoded_sentence)
            all_cleared_decoded_anwsers.append(cleared_decoded_anwser)
            all_cleared_decoded_predictions.append(cleared_decoded_prediction)

    all_results = list()
    result_pairs = None
    if IS_ZERO_SHOT:
        utility.set_zero_shot_encoder(encoder_type)
        zero_shot_classes = utility.get_zero_shot_classes(
            EXECUTE_TYPE, WITH_DESC)

        entities_to_compute = None
        if EXECUTE_TYPE == utility.SLOT_FILLING:
            entities_to_compute = list(filter(lambda item: item, map(lambda item: utility.get_entity_info(
                item, with_desc=WITH_DESC), all_cleared_decoded_predictions)))
        elif EXECUTE_TYPE == utility.INTENT_DETECTION:
            entities_to_compute = list(filter(lambda item: item, map(lambda item: utility.get_intent_info(
                item, with_desc=WITH_DESC), all_cleared_decoded_predictions)))

        similarity_results = None
        if ENCODER_TYPE == 'bert':
            encoded_predicted_entities, encoded_zero_shot_classes = utility.batch_transform_to_bert_encoded_input(
                all_cleared_decoded_sentences,
                entities_to_compute,
                zero_shot_classes,
                MAX_SEQ_LENGTH,
                IS_BASIC_TRANSFERRING)
            predicted_entities_embeddings, zero_shot_embeddings = utility.encode_to_embedding(
                encoded_predicted_entities,
                encoded_zero_shot_classes,
                BATCH_SIZE)

            if not IS_BASIC_TRANSFERRING:
                intent_string_similarities = utility.compute_all_intent_string_similarity(
                    entities_to_compute, zero_shot_classes)
                similarity_results = utility.compute_all_cosine_intent_string_similarity(
                    predicted_entities_embeddings,
                    zero_shot_embeddings,
                    intent_string_similarities)
            else:
                similarity_results = utility.compute_all_cosine_similarity(
                    predicted_entities_embeddings, zero_shot_embeddings)

        elif ENCODER_TYPE == 't5':
            encoded_stsb_inputs = utility.get_encoded_stsb_inputs(
                all_cleared_decoded_sentences,
                entities_to_compute,
                zero_shot_classes,
                MAX_SEQ_LENGTH,
                IS_BASIC_TRANSFERRING)

            if not IS_BASIC_TRANSFERRING:
                intent_string_similarities = utility.compute_all_intent_string_similarity(
                    entities_to_compute, zero_shot_classes)
                similarity_results = utility.compute_all_stsb_intent_string_similarity(
                    encoded_stsb_inputs, intent_string_similarities)
            else:
                similarity_results = utility.compute_all_stsb_similarity(
                    encoded_stsb_inputs)

        elif ENCODER_TYPE == 'mixed':
            if not IS_PREDICTED_FILE_EXISTED:
                encoded_predicted_entities, encoded_zero_shot_classes = utility.batch_transform_to_bert_encoded_input(
                    all_cleared_decoded_sentences,
                    entities_to_compute,
                    zero_shot_classes,
                    MAX_SEQ_LENGTH,
                    IS_BASIC_TRANSFERRING)
                predicted_entities_embeddings, zero_shot_embeddings = utility.encode_to_embedding(
                    encoded_predicted_entities,
                    encoded_zero_shot_classes,
                    BATCH_SIZE)
                bert_cosine_similarity_results = utility.compute_all_cosine_similarity(
                    predicted_entities_embeddings, zero_shot_embeddings)

                encoded_stsb_inputs = utility.get_encoded_stsb_inputs(
                    all_cleared_decoded_sentences,
                    entities_to_compute,
                    zero_shot_classes,
                    MAX_SEQ_LENGTH,
                    IS_BASIC_TRANSFERRING)
                t5_stsb_similarity_results = utility.compute_all_stsb_similarity(
                    encoded_stsb_inputs)

                intent_string_similarities = utility.compute_all_intent_string_similarity(
                    entities_to_compute, zero_shot_classes)

                # similarity_results = utility.compute_all_mixed_similarity(
                #     src_dataset,
                #     tgt_dataset,
                #     model_type,
                #     bert_cosine_similarity_results,
                #     t5_stsb_similarity_results,
                #     intent_string_similarities)
                utility.record_all_mixed_similarity(src_dataset,
                                                    tgt_dataset,
                                                    bert_cosine_similarity_results,
                                                    t5_stsb_similarity_results,
                                                    intent_string_similarities)

            similarity_results = utility.load_similarities(
                src_dataset,
                tgt_dataset,
                BERT_COSINE_SIMILARITY_WEIGHT,
                T5_STSB_SIMILARITY_WEIGHT,
                INTENT_STRING_SIMILARITY_WEIGHT)

        for similarity_result_index, record_per_prediction in enumerate(similarity_results):
            record_per_prediction.sort(
                key=lambda item: item[1], reverse=True)

        most_similar_class_info = list(
            map(lambda item: item[0], similarity_results))
        most_similar_classes = list(map(
            lambda item: zero_shot_classes[item[0]], most_similar_class_info))
        most_similarities = list(
            map(lambda item: str(item[1]), most_similar_class_info))

        if EXECUTE_TYPE == utility.SLOT_FILLING:
            changed_predictions = list(filter(lambda item: item, map(lambda item: utility.change_output(item[0], item[1]), list(
                zip(all_cleared_decoded_predictions, most_similar_classes)))))
        elif EXECUTE_TYPE == utility.INTENT_DETECTION:
            changed_predictions = most_similar_classes

        result_pairs = list(zip(all_cleared_decoded_sentences, all_cleared_decoded_anwsers,
                                all_cleared_decoded_predictions, changed_predictions, most_similarities))
    else:
        result_pairs = list(
            zip(all_cleared_decoded_sentences, all_cleared_decoded_anwsers, all_cleared_decoded_predictions))

    all_results.extend(list(map(lambda decoded_item: '\t'.join(
        list(decoded_item)) + os.linesep, result_pairs)))

    end = time.time()
    utility.logging.info('Total predicting time: {}'.format(
        timedelta(seconds=end - start).__str__()))

    return all_results


def get_encoded_tensors(target_type, data_generator, tags_mapping_obj):
    if target_type.startswith('t5'):
        return utility.encode_all_from_generator(
            data_generator,
            MAX_SEQ_LENGTH,
            MAX_LABEL_SEQ_LENGTH)
    elif target_type.startswith('bert'):
        pass
        # return utility.encode_all(MAX_SEQ_LENGTH, sentences,
        #                           sentences_tags, tags_mapping_obj)


def get_target_bert_model(target_type):
    config = BertConfig.from_json_file(
        os.path.join(MODEL_PATH, 'config.json'))
    bert_model = TFBertForTokenClassification.from_pretrained(
        os.path.join(MODEL_PATH, 'tf_model.h5'), config=config)

    model = None
    if target_type == 'bert_based':
        model = utility.get_compiled_bert_model(
            bert_model, LEARNING_RATE)
    elif target_type == 'bert_crf':
        model = utility.get_compiled_bert_crf_model(
            bert_model, MAX_SEQ_LENGTH, LEARNING_RATE)
    model.summary()

    return model


def convert_to_consistency_labels(target_type, prediction):
    if target_type == 'bert_based':
        predictions_labels = utility.numpy.argmax(
            tf.squeeze(prediction), axis=-1).tolist()
    elif target_type == 'bert_crf':
        predictions_labels = tf.cast(prediction, tf.int32).numpy().tolist()

    return predictions_labels


def predict_by_bert_model(target_type, tensors, tags_mapping_obj):
    model = get_target_bert_model(target_type)

    x = {
        'input_ids': tensors.get('input_ids'),
        'attention_mask': tensors.get('attention_mask'),
        'token_type_ids': tensors.get('token_type_ids')
    }
    sentences_tokens_ids = tensors.get('input_ids').numpy().tolist()
    answers_labels = tensors.get('labels').numpy().tolist()
    predictions_labels = convert_to_consistency_labels(
        target_type, model.predict(x))
    sentences, sentences_answers, sentences_predictions = utility.decode_all(
        sentences_tokens_ids, answers_labels, predictions_labels, tags_mapping_obj)
    all_results = list(map(lambda pairs: '\t'.join(
        [pairs[0], pairs[1], pairs[2]]) + os.linesep, list(zip(sentences, sentences_answers, sentences_predictions))))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute_type', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str)
    parser.add_argument('--src_dataset', required=True, type=str)
    parser.add_argument('--tgt_dataset', required=True, type=str)
    parser.add_argument('--setting_path', required=True, type=str)
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--selected_model', required=True, type=str)
    parser.add_argument('--init_sentence_size', required=False,
                        default=math.inf, type=int)
    parser.add_argument('--is_zero_shot', required=True, type=str)
    parser.add_argument('--with_desc', required=False,
                        default='False', type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--given_model_path',
                        required=False, default=None, type=str)
    parser.add_argument('--encoder_type', required=True, type=str)
    parser.add_argument('--bert_cosine_similarity_weight',
                        required=True, type=str)
    parser.add_argument('--t5_stsb_similarity_weight', required=True, type=str)
    parser.add_argument('--intent_string_similarity_weight',
                        required=True, type=str)

    args = parser.parse_args()
    EXECUTE_TYPE = args.execute_type
    DATA_TYPE = args.data_type
    SRC_DATASET = args.src_dataset
    TGT_DATASET = args.tgt_dataset
    SETTING_PATH = args.setting_path
    utility.set_global_setting(SETTING_PATH)
    SETTING = utility.GLOBAL_SETTING
    utility.set_scheme_config(SETTING.get('scheme_config_path'))
    utility.set_entity_config(SETTING.get('entity_config_path'))
    utility.set_intent_config(SETTING.get('intent_config_path'))

    SELECTED_MODEL = args.selected_model
    INIT_SENTENCES_SIZE = args.init_sentence_size
    IS_ZERO_SHOT = utility.strtobool(args.is_zero_shot)
    WITH_DESC = utility.strtobool(args.with_desc)
    DATA_PATH = args.data_path

    FULL_ENCODER_TYPE = args.encoder_type
    SPLITTED_ENCODER_TYPE = FULL_ENCODER_TYPE.split('_')
    ENCODER_TYPE = SPLITTED_ENCODER_TYPE[0]
    IS_BASIC_TRANSFERRING = FULL_ENCODER_TYPE.endswith('baseline')

    TAGS_MAPPING_OBJ_PATH = None

    MODEL_TYPE = args.model_type
    MODEL_PATH = os.path.join(SETTING.get(
        MODEL_TYPE).get('model_path'), SELECTED_MODEL)
    GIVEN_MODEL_PATH = args.given_model_path
    MAX_SEQ_LENGTH = SETTING.get(MODEL_TYPE).get('max_seq_length')
    MAX_LABEL_SEQ_LENGTH = SETTING.get(MODEL_TYPE).get('max_label_seq_length')
    BATCH_SIZE = SETTING.get(MODEL_TYPE).get('predict_batch_size')
    LEARNING_RATE = SETTING.get(MODEL_TYPE).get('learning_rate')
    LABEL_LENGTH = SETTING.get(MODEL_TYPE).get('label_length')

    utility.set_current_config('BIO', TGT_DATASET)
    data_generator = utility.load_t5_format_data(DATA_PATH)

    # tags_mapping_obj = utility.load_tags_mapping_obj(TAGS_MAPPING_OBJ_PATH)
    tags_mapping_obj = None

    BERT_COSINE_SIMILARITY_WEIGHT = float(args.bert_cosine_similarity_weight)
    T5_STSB_SIMILARITY_WEIGHT = float(args.t5_stsb_similarity_weight)
    INTENT_STRING_SIMILARITY_WEIGHT = float(
        args.intent_string_similarity_weight)

    LOAD_MODEL_PATH = MODEL_PATH
    if GIVEN_MODEL_PATH:
        LOAD_MODEL_PATH = GIVEN_MODEL_PATH

    predicted_file_name = '{}_{}_{}_{}_predicted_results.txt'.format(
        TGT_DATASET, EXECUTE_TYPE, DATA_TYPE, FULL_ENCODER_TYPE)
    predicted_result_path = os.path.join(
        LOAD_MODEL_PATH, predicted_file_name)
    IS_PREDICTED_FILE_EXISTED = os.path.isfile(predicted_result_path)

    tensors = None
    if not IS_PREDICTED_FILE_EXISTED:
        tensors = get_encoded_tensors(
            MODEL_TYPE, data_generator, tags_mapping_obj)

    all_results = None
    if MODEL_TYPE == 't5_based':
        all_results = predict_by_t5_based(
            tensors,
            ENCODER_TYPE,
            SRC_DATASET,
            TGT_DATASET,
            predicted_result_path)
    elif MODEL_TYPE.startswith('bert'):
        all_results = predict_by_bert_model(
            MODEL_TYPE, tensors, tags_mapping_obj)

    result_count = len(all_results)
    result_file_name = '{}_{}_{}_{}_results.txt'.format(
        TGT_DATASET, EXECUTE_TYPE, DATA_TYPE, FULL_ENCODER_TYPE)
    result_path = os.path.join(
        LOAD_MODEL_PATH, result_file_name)
    with open(result_path, 'w') as writer:
        writer.writelines(all_results)
    utility.logging.info('已寫入{}筆測試結果'.format(result_count))
    message = '\n已寫入測試結果: {}\n預測結果寫入於:{}'.format(result_count, result_path)
    # line_notifier.sendMessage('苦命研究生', MODEL_TYPE.upper(), message)
