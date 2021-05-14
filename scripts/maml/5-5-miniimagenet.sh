#!/bin/bash

ROOT=$DATADIR
GPUS=(0 0 0)
DATASET="miniimagenet"
VSHOT=15

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --ml-algorithm=MAML \
    --num-models=1 \
    --no-batchnorm \
    --minibatch 2 \
    --n-way=5 \
    --k-shot=5 \
    --v-shot=$VSHOT \
    --inner-lr 0.01 \
    --num-epochs=40 \
    --inner-lr 0.01 \
    --num-episodes-per-epoch 10000 \
    --resume-epoch=0 \
    --train
