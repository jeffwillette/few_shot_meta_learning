#!/bin/bash

ROOT=$DATADIR
GPUS=(0 0 0)
DATASET="omniglot"
VSHOT=15

for RUN in 0 1 2
do
  CUDA_VISIBLE_DEVICES=${GPUS[RUN]} PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --ml-algorithm=abml \
    --num-models=2 \
    --no-batchnorm \
    --n-way=20 \
    --k-shot=1 \
    --v-shot=$VSHOT \
    --num-epochs=40 \
    --num-episodes-per-epoch 10000 \
    --resume-epoch=0 \
    --train
done
