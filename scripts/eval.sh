#!/bin/bash

ROOT=$DATADIR
GPUS=(4 4 4)
DATASET="omniglot"
# NWAY=20
# KSHOT=1
VSHOT=15
ALGORITHM="maml"

for NWAY in 5 20
do
    for KSHOT in 1 5
    do
        for RUN in 0 1 2
        do
          CUDA_VISIBLE_DEVICES=${GPUS[RUN]} PYTHONPATH=. python main.py \
            --datasource=$DATASET \
            --ds-folder $ROOT \
            --run $RUN \
            --ml-algorithm=$ALGORITHM \
            --num-models=2 \
            --minibatch 16 \
            --no-batchnorm \
            --n-way=$NWAY \
            --k-shot=$KSHOT \
            --v-shot=$VSHOT \
            --num-epochs=40 \
            --num-episodes-per-epoch 10000 \
            --resume-epoch=0 \
            --test

          CUDA_VISIBLE_DEVICES=${GPUS[RUN]} PYTHONPATH=. python main.py \
            --datasource=$DATASET \
            --ds-folder $ROOT \
            --run $RUN \
            --ml-algorithm=$ALGORITHM \
            --num-models=2 \
            --minibatch 16 \
            --no-batchnorm \
            --n-way=$NWAY \
            --k-shot=$KSHOT \
            --v-shot=$VSHOT \
            --num-epochs=40 \
            --num-episodes-per-epoch 10000 \
            --resume-epoch=0 \
            --ood-test \
            --test

          CUDA_VISIBLE_DEVICES=${GPUS[RUN]} PYTHONPATH=. python main.py \
            --datasource=$DATASET \
            --ds-folder $ROOT \
            --run $RUN \
            --ml-algorithm=$ALGORITHM \
            --num-models=2 \
            --minibatch 16 \
            --no-batchnorm \
            --n-way=$NWAY \
            --k-shot=$KSHOT \
            --v-shot=$VSHOT \
            --num-epochs=40 \
            --num-episodes-per-epoch 10000 \
            --resume-epoch=0 \
            --corrupt \
            --test
        done
    done
done
