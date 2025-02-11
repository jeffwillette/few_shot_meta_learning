#!/bin/bash

ROOT=$DATADIR
GPUS=(0 0 0)
DATASET="miniimagenet"
VSHOT=15

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --run $RUN \
    --ml-algorithm=MAML \
    --num-models=1 \
    --minibatch 4 \
    --inner-lr 0.01 \
    --no-batchnorm \
    --n-way=5 \
    --k-shot=1 \
    --v-shot=$VSHOT \
    --num-epochs=40 \
    --num-episodes-per-epoch 10000 \
    --resume-epoch=0 \
    --train
