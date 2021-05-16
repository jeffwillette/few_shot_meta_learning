#!/bin/bash

ROOT=$DATADIR
GPUS=(0 0 0)
DATASET="miniimagenet"
VSHOT=15

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --run $RUN \
    --ml-algorithm=vampire \
    --num-models=2 \
    --no-batchnorm \
    --n-way=5 \
    --k-shot=5 \
    --v-shot=$VSHOT \
    --minibatch 2 \
    --inner-lr 0.01 \
    --num-epochs=40 \
    --num-episodes-per-epoch 10000 \
    --resume-epoch=0 \
    --train
