#!/bin/bash

EXECUTE_TYPE=$1
SRC_DATASET=$2
TGT_DATASET=$3
SETTING_PATH=/home/albeli/workspace/NCCU/Experiment/config/setting.json
MODEL_TYPE=$4
DATA_TYPE=$5
SELECTED_MODEL=$6
CONVERT_FROM=$7
WITH_DESC=$8
IS_ZERO_SHOT=$9
ENCODER_TYPE=${10}
GIVEN_MODEL_PATH=${11}
BERT_COSINE_SIMILARITY_WEIGHT=${12}
T5_STSB_SIMILARITY_WEIGHT=${13}
INTENT_STRING_SIMILARITY_WEIGHT=${14}
DATA_FILE_NAME=t5_format_"$EXECUTE_TYPE"_"$DATA_TYPE"_none_data.txt
DATA_PATH=/home/albeli/workspace/NCCU/Experiment/datasets/"$TGT_DATASET"/"$DATA_FILE_NAME"

if [ ! -f "${DATA_PATH}" ]; then
    python generate_t5_format_data.py \
    --execute_type=$EXECUTE_TYPE \
    --data_type=$DATA_TYPE \
    --which_dataset=$TGT_DATASET \
    --setting_path=./config/setting.json \
    --shuffle_times=10 \
    --with_desc=$WITH_DESC
fi

if [ "${GIVEN_MODEL_PATH}" != "False" ]; then
    python predict_test_mixed_hyperparameters.py \
    --execute_type=$EXECUTE_TYPE \
    --data_type=$DATA_TYPE \
    --src_dataset=$SRC_DATASET \
    --tgt_dataset=$TGT_DATASET \
    --setting_path=$SETTING_PATH \
    --model_type=$MODEL_TYPE \
    --selected_model=$SELECTED_MODEL \
    --data_path=$DATA_PATH \
    --is_zero_shot=$IS_ZERO_SHOT \
    --with_desc=$WITH_DESC \
    --encoder_type=$ENCODER_TYPE \
    --given_model_path=$GIVEN_MODEL_PATH \
    --bert_cosine_similarity_weight=$BERT_COSINE_SIMILARITY_WEIGHT \
    --t5_stsb_similarity_weight=$T5_STSB_SIMILARITY_WEIGHT \
    --intent_string_similarity_weight=$INTENT_STRING_SIMILARITY_WEIGHT 
    # --init_sentence_size=1

    if [ "${EXECUTE_TYPE}" = "slot_filling" ]; then
        python convert_conll_format.py \
        --execute_type=$EXECUTE_TYPE \
        --data_type=$DATA_TYPE \
        --tgt_dataset=$TGT_DATASET \
        --setting_path=$SETTING_PATH \
        --model_type=$MODEL_TYPE \
        --selected_model=$SELECTED_MODEL \
        --convert_from=$CONVERT_FROM \
        --is_zero_shot=$IS_ZERO_SHOT \
        --given_model_path=$GIVEN_MODEL_PATH

        python conlleval.py \
        < "$GIVEN_MODEL_PATH"/"$TGT_DATASET"_"$EXECUTE_TYPE"_"$DATA_TYPE"_"$ENCODER_TYPE"_results_conll_format.txt > \
        "$GIVEN_MODEL_PATH"/"$TGT_DATASET"_"$EXECUTE_TYPE"_"$DATA_TYPE"_"$ENCODER_TYPE"_final_results.txt
    
    elif [ "${EXECUTE_TYPE}" = "intent_detection" ]; then
        python intent_detection_eval_test_mixed_hyperparameters.py \
        --execute_type=$EXECUTE_TYPE \
        --data_type=$DATA_TYPE \
        --src_dataset=$SRC_DATASET \
        --tgt_dataset=$TGT_DATASET \
        --setting_path=$SETTING_PATH \
        --model_type=$MODEL_TYPE \
        --selected_model=$SELECTED_MODEL \
        --convert_from=$CONVERT_FROM \
        --is_zero_shot=$IS_ZERO_SHOT \
        --encoder_type=$ENCODER_TYPE \
        --given_model_path=$GIVEN_MODEL_PATH \
        --bert_cosine_similarity_weight=$BERT_COSINE_SIMILARITY_WEIGHT \
        --t5_stsb_similarity_weight=$T5_STSB_SIMILARITY_WEIGHT \
        --intent_string_similarity_weight=$INTENT_STRING_SIMILARITY_WEIGHT 
    fi
else
    python predict_test_mixed_hyperparameters.py \
    --execute_type=$EXECUTE_TYPE \
    --data_type=$DATA_TYPE \
    --src_dataset=$SRC_DATASET \
    --tgt_dataset=$TGT_DATASET \
    --setting_path=$SETTING_PATH \
    --model_type=$MODEL_TYPE \
    --selected_model=$SELECTED_MODEL \
    --data_path=$DATA_PATH \
    --is_zero_shot=$IS_ZERO_SHOT \
    --encoder_type=$ENCODER_TYPE \
    --with_desc=$WITH_DESC \
    --bert_cosine_similarity_weight=$BERT_COSINE_SIMILARITY_WEIGHT \
    --t5_stsb_similarity_weight=$T5_STSB_SIMILARITY_WEIGHT \
    --intent_string_similarity_weight=$INTENT_STRING_SIMILARITY_WEIGHT 
    # --init_sentence_size=1

    if [ "${EXECUTE_TYPE}" = "slot_filling" ]; then
        python convert_conll_format.py \
        --execute_type=$EXECUTE_TYPE \
        --data_type=$DATA_TYPE \
        --tgt_dataset=$TGT_DATASET \
        --setting_path=$SETTING_PATH \
        --model_type=$MODEL_TYPE \
        --selected_model=$SELECTED_MODEL \
        --convert_from=$CONVERT_FROM \
        --is_zero_shot=$IS_ZERO_SHOT

        python conlleval.py \
        < /home/albeli/workspace/NCCU/Experiment/models/$MODEL_TYPE/$SELECTED_MODEL/"$TGT_DATASET"_"$EXECUTE_TYPE"_"$DATA_TYPE"_"$ENCODER_TYPE"_results_conll_format.txt > \
        /home/albeli/workspace/NCCU/Experiment/models/$MODEL_TYPE/$SELECTED_MODEL/"$TGT_DATASET"_"$EXECUTE_TYPE"_"$DATA_TYPE"_"$ENCODER_TYPE"_final_results.txt
    elif [ "${EXECUTE_TYPE}" = "intent_detection" ]; then
        python intent_detection_eval_test_mixed_hyperparameters.py \
        --execute_type=$EXECUTE_TYPE \
        --data_type=$DATA_TYPE \
        --src_dataset=$SRC_DATASET \
        --tgt_dataset=$TGT_DATASET \
        --setting_path=$SETTING_PATH \
        --model_type=$MODEL_TYPE \
        --selected_model=$SELECTED_MODEL \
        --convert_from=$CONVERT_FROM \
        --encoder_type=$ENCODER_TYPE \
        --is_zero_shot=$IS_ZERO_SHOT \
        --bert_cosine_similarity_weight=$BERT_COSINE_SIMILARITY_WEIGHT \
        --t5_stsb_similarity_weight=$T5_STSB_SIMILARITY_WEIGHT \
        --intent_string_similarity_weight=$INTENT_STRING_SIMILARITY_WEIGHT 
    fi
fi