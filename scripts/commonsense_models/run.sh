#!/bin/bash

source ./config/params.config

LOG_FOLDER="./checkpoints_6lendict_wcontains/"

mkdir -p $LOG_FOLDER
# nohup
CUDA_VISIBLE_DEVICES=2,1 python -u main.py \
    --data_dir $data_dir \
    --save_dir $save_dir \
    --model $model \
    --learning_rate $learning_rate \
    --warmup_steps $warmup_steps \
    --weight_decay $weight_decay \
    --num_epoch $num_epoch \
    --batch_size $batch_size \
    --gpu_device 1
#    > $LOG_FOLDER/$params.log 2>&1 &

