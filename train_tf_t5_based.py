# -*- coding: UTF-8 -*-

import utility
from utility import (T5Config, TFT5ForConditionalGeneration, argparse,
                     datetime, math, os, sys, tf, time, timedelta, traceback)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute_type', required=True, type=str)
    parser.add_argument('--src_dataset', required=True, type=str)
    parser.add_argument('--setting_path', required=True, type=str)
    parser.add_argument('--init_sentence_size', required=False,
                        default=math.inf, type=int)
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
    SRC_DATASET = args.src_dataset
    SETTING_PATH = args.setting_path
    utility.set_global_setting(SETTING_PATH)
    SETTING = utility.GLOBAL_SETTING
    utility.set_scheme_config(SETTING.get('scheme_config_path'))
    utility.set_entity_config(SETTING.get('entity_config_path'))
    utility.set_intent_config(SETTING.get('intent_config_path'))
    INIT_SENTENCES_SIZE = args.init_sentence_size
    EXECUTE_TYPE = args.execute_type
    DATA_TYPE = 'train'
    AUGMENTATION_SRC = args.augment_src.split(
        ',') if args.augment_src else list()
    CO_TRAIN = args.co_train.split(
        ',') if args.co_train else list()
    AUGMENTATION_SRC_FILE_NAME = '_'.join(AUGMENTATION_SRC)
    CO_TRAIN_FILE_NAME = '_'.join(
        ['co_train'] + CO_TRAIN) if len(CO_TRAIN) else 'none'

    DATA_SRC_PATH = args.data_src_path
    ROOT_DATASETS_PATH = SETTING.get('dataset_path')
    TRAIN_DATA_PATH = ROOT_DATASETS_PATH.get(
        '{}_dataset_path'.format(DATA_TYPE)).format(SRC_DATASET)
    EXTENDED_DATA_PATH = ROOT_DATASETS_PATH.get('extended_dataset_path')
    T5_DATA_PATH = ROOT_DATASETS_PATH.get(
        't5_format_dataset_path').format(SRC_DATASET, EXECUTE_TYPE, DATA_TYPE, CO_TRAIN_FILE_NAME)
    CO_TRAIN_INFO = list()
    for co_train_type in CO_TRAIN:
        path = ROOT_DATASETS_PATH.get(
            '{}_dataset_path'.format(DATA_TYPE)).format(co_train_type)
        CO_TRAIN_INFO.append((path, co_train_type))

    WITH_DESC = utility.strtobool(args.with_desc)
    PROB_THRESHOLD = args.prob_threshold
    EXTEND_REPEAT = args.extend_repeat
    SHUFFLE_TIMES = args.shuffle_times
    COMPLETED_EXTENDED_DATA_PATH = EXTENDED_DATA_PATH.format(
        SRC_DATASET,
        EXECUTE_TYPE,
        AUGMENTATION_SRC_FILE_NAME,
        CO_TRAIN_FILE_NAME,
        PROB_THRESHOLD,
        EXTEND_REPEAT,
        WITH_DESC)

    MODEL_TYPE = 't5_based'
    MAX_SEQ_LENGTH = SETTING.get(MODEL_TYPE).get('max_seq_length')
    MAX_LABEL_SEQ_LENGTH = SETTING.get(MODEL_TYPE).get('max_label_seq_length')
    BATCH_SIZE = SETTING.get(MODEL_TYPE).get('train_batch_size')
    EPOCHS = SETTING.get(MODEL_TYPE).get(
        'train_hyperparameters').get(SRC_DATASET).get('epoch')
    LEARNING_RATE = SETTING.get(MODEL_TYPE).get('train_hyperparameters').get(
        SRC_DATASET).get('learning_rate')
    LABEL_LENGTH = list(range(MAX_LABEL_SEQ_LENGTH))

    utility.set_current_config('BIO', SRC_DATASET)

    RAW_DATA_PATH = TRAIN_DATA_PATH
    if DATA_SRC_PATH:
        RAW_DATA_PATH = DATA_SRC_PATH

    train_sentences, train_sentences_tags, train_intents = utility.load_all_data(
        RAW_DATA_PATH)

    if math.isfinite(INIT_SENTENCES_SIZE):
        train_sentences = train_sentences[: INIT_SENTENCES_SIZE]
        train_sentences_tags = train_sentences_tags[: INIT_SENTENCES_SIZE]
        train_intents = train_intents[: INIT_SENTENCES_SIZE]

    if len(AUGMENTATION_SRC):
        DATA_PATH = COMPLETED_EXTENDED_DATA_PATH
    else:
        DATA_PATH = T5_DATA_PATH

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

    # for item in t5_format_data_generator:
    #     pass
    train_tensors = utility.encode_all_from_generator(
        t5_format_data_generator, MAX_SEQ_LENGTH, MAX_LABEL_SEQ_LENGTH)

    all_model_folders = os.listdir(
        SETTING.get(MODEL_TYPE).get('model_path'))
    all_model_folders.sort(reverse=True)
    current_max_folder_num = 0
    if len(all_model_folders):
        current_max_folder_num = int(all_model_folders[0].split('-')[-1])
    model_saved_path = os.path.join(SETTING.get(MODEL_TYPE).get(
        'model_path'), '{}-{:02}/'.format(MODEL_TYPE, current_max_folder_num + 1))
    if not os.path.isdir(model_saved_path):
        os.makedirs(model_saved_path)

    with open(os.path.join(model_saved_path, 'training_info.txt'), 'w') as writer:
        writer.write(os.linesep.join(['model: {}'.format(MODEL_TYPE),
                                      'data_src_path: {}'.format(
                                          RAW_DATA_PATH),
                                      't5_format_data_path: {}'.format(
                                          DATA_PATH),
                                      'execute_type: {}'.format(EXECUTE_TYPE),
                                      'src_dataset: {}'.format(
                                          SRC_DATASET),
                                      'co_train: {}'.format(CO_TRAIN),
                                      'with_desc: {}'.format(WITH_DESC),
                                      'augmentation_src: {}'.format(
                                          AUGMENTATION_SRC),
                                      'prob_threshold: {}'.format(
                                          PROB_THRESHOLD),
                                      'extend_repeat: {}'.format(
                                          EXTEND_REPEAT),
                                      'epoch: {}'.format(EPOCHS),
                                      'batch_size: {}'.format(BATCH_SIZE),
                                      'learning_rate: {}'.format(
                                          LEARNING_RATE),
                                      'max_seq_length: {}'.format(
                                          MAX_SEQ_LENGTH),
                                      'max_label_seq_length: {}'.format(MAX_LABEL_SEQ_LENGTH)]))

    train_datasets, datasets_total_length = utility.get_tensor_datasets(
        train_tensors, BATCH_SIZE)
    # train_datasets = train_datasets.cache(os.path.join(model_saved_path, 'train.cache')).prefetch(
    #     buffer_size=tf.data.experimental.AUTOTUNE).cache().batch(BATCH_SIZE)

    config = T5Config.from_pretrained(utility.T5_PRE_TRAINED)
    model = TFT5ForConditionalGeneration.from_pretrained(
        utility.T5_PRE_TRAINED, config=config)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, epsilon=1e-8, clipnorm=1.0)
    train_metrics = tf.keras.metrics.Accuracy()
    metric_names = ['Loss', train_metrics.__class__.__name__]
    model.summary()

    current_epoch = 0
    current_step = 0
    exception_happened = False
    consuming_time = None
    try:
        start = time.time()
        for epoch in range(1, EPOCHS + 1):
            progress_bar = tf.keras.utils.Progbar(
                datasets_total_length, stateful_metrics=metric_names)
            current_epoch = epoch
            final_loss = 0
            final_accuracy = 0
            print('Epoch: {}/{}'.format(epoch, EPOCHS))
            for step, batch_train_inputs in enumerate(train_datasets, 1):
                current_step = step
                with tf.GradientTape() as tape:
                    outputs = model(batch_train_inputs, return_dict=True)
                    loss = tf.reduce_mean(outputs.get('loss'))
                    prediction_from_logits = tf.argmax(
                        outputs.get('logits'), axis=-1)
                    train_labels = tf.gather(batch_train_inputs.get(
                        'labels'), LABEL_LENGTH, axis=-1)
                    train_prediction = tf.cast(
                        tf.gather(prediction_from_logits, LABEL_LENGTH, axis=-1), tf.int32)

                train_metrics.update_state(train_labels, train_prediction)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                accuracy = train_metrics.result()
                values = [('Loss', loss),
                          (train_metrics.__class__.__name__, accuracy)]
                progress_bar.update(step, values=values)
                utility.logging.info('Epoch: {}/{}, Step: {}/{}, Progress: {:.4%}, Loss: {}, {}: {:.4%}'.format(
                    epoch,
                    EPOCHS,
                    step,
                    datasets_total_length,
                    step / datasets_total_length,
                    loss,
                    train_metrics.__class__.__name__,
                    accuracy))
                final_loss = loss
                final_accuracy = accuracy

            train_metrics.reset_states()
            final_values = [('Loss', final_loss),
                            (train_metrics.__class__.__name__, final_accuracy)]
            progress_bar.update(datasets_total_length,
                                values=final_values, finalize=True)

            # if epoch in [3, 5, 7]:
            #     end = time.time()
            #     consuming_time = timedelta(
            #         seconds=end - start).__str__()

            #     all_model_folders = os.listdir(
            #         SETTING.get(MODEL_TYPE).get('model_path'))
            #     all_model_folders.sort(reverse=True)
            #     current_max_folder_num = 0
            #     if len(all_model_folders):
            #         current_max_folder_num = int(
            #             all_model_folders[0].split('-')[-1])
            #     model_saved_path = os.path.join(SETTING.get(MODEL_TYPE).get(
            #         'model_path'), '{}-{:02}/'.format(MODEL_TYPE, current_max_folder_num + 1))
            #     if not os.path.isdir(model_saved_path):
            #         os.makedirs(model_saved_path)
            #     model.save_pretrained(model_saved_path)
            #     message = '\n模型權重儲存於: {}\n總共訓練: {}'.format(
            #         model_saved_path, consuming_time)
            #     line_notifier.sendMessage(
            #         '苦命研究生', MODEL_TYPE.upper(), message)
        end = time.time()
        consuming_time = timedelta(seconds=end - start).__str__()
        utility.logging.info('Total training time: {}'.format(consuming_time))
    except Exception:
        exception_happened = True
        utility.logging.error('錯誤發生', exc_info=True)
        # line_notifier.sendMessage('苦命研究生', MODEL_TYPE.upper(), '\n模型訓練停止於Epoch: {}, Step: {}\n錯誤訊息: {}'.format(
        #     current_epoch, current_step, traceback.format_exc()))
    finally:
        model.save_pretrained(model_saved_path)

        message = None
        if not exception_happened:
            message = '\n模型權重儲存於: {}\n總共訓練: {}'.format(
                model_saved_path, consuming_time)
        else:
            exception_record_path = os.path.join(
                model_saved_path, 'exception.record')
            with open(exception_record_path, 'w') as writer:
                writer.write(traceback.format_exc())

            data_stop_record_path = os.path.join(
                model_saved_path, 'data_stop.record')
            with open(data_stop_record_path, 'w') as writer:
                writer.write(str(current_step * BATCH_SIZE))
            message = '\n模型權重儲存於: {}\n錯誤資訊寫入於: {}\n資料終止數記錄寫入於: {}'.format(
                model_saved_path, exception_record_path, data_stop_record_path)

        # line_notifier.sendMessage('苦命研究生', MODEL_TYPE.upper(), message)
