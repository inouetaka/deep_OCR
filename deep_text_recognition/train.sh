set -ue

source activate pytorch_p36

TRAIN_DATA=$train_data
VALID_DATA=$valid_data
EXPER_NAME=$experiment_name
NUM_ITER=$num_iter
SAVED_MODELS=$saved_models

if [ -n "$SAVED_MODELS" ]; then
    echo "$SAVED_MODELS で再学習を行います。"

    nohup python3 train.py --train_data data_lmdb_release/train/MJ/$TRAIN_DATA/ \
    --valid_data data_lmdb_release/validation/$VALID_DATA/ --select_data MJ \
    --batch_ratio 1.0 --Transformation TPS --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM --Prediction CTC --experiment_name $EXPER_NAME \
    --num_iter $NUM_ITER --valInterval 100  --sensitive --rgb --workers 10 \
    --continue_model saved_models/$SAVED_MODELS &

    echo -e "train_data:$TRAIN_DATA\\nvaldation_data:$VALID_DATA\\nmodel_name:$EXPER_NAME\\niteration:$NUM_ITER"
    echo "nohup.outにlogが出ます。"
fi
if [ -z "$SAVED_MODELS" ]; then
    
    nohup python3 train.py --train_data data_lmdb_release/train/MJ/$TRAIN_DATA/ \
    --valid_data data_lmdb_release/validation/$VALID_DATA/ --select_data MJ \
    --batch_ratio 1.0 --Transformation TPS --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM --Prediction CTC --experiment_name $EXPER_NAME \
    --num_iter $NUM_ITER --valInterval 100  --sensitive --rgb --workers 10 &

    echo -e "train_data:$TRAIN_DATA\\nvaldation_data:$VALID_DATA\\nmodel_name:$EXPER_NAME\\niteration:$NUM_ITER"
    echo "nohup.outにlogが出ます。"
fi
