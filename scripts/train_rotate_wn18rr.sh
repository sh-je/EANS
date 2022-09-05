#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

MODE=train
MODEL=RotatE
DATASET=wn18rr
GPU_DEVICE=0
SAVE_ID=0

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

# Best Configs
BATCH_SIZE=512
NEGATIVE_SAMPLE_SIZE=1024
HIDDEN_DIM=500
GAMMA=6.0
ALPHA=0.5
LEARNING_RATE=0.00005
REGULARIZATION=0.0
MAX_STEPS=80000
TEST_BATCH_SIZE=8
SAMPLING_METHOD=gaussian
NCLUSTERS=100
VARIANCE=800
REORDER_STEPS=1000
SUB_LOSS_WEIGHT=0.05
SUB_REGULARIZATION=0.01


if [ $MODE == "train" ]
then

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -r $REGULARIZATION \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    -sm $SAMPLING_METHOD \
    -k $NCLUSTERS  -v $VARIANCE --reorder_steps $REORDER_STEPS \
    -sub -subl $SUB_LOSS_WEIGHT -subr $SUB_REGULARIZATION \
    -adv \
    -de

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE

elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

else
   echo "Unknown MODE" $MODE
fi
