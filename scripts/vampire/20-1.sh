ROOT=$DATADIR
GPUS=(0 0 0)
DATASET="omniglot"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --ml-algorithm=vampire \
    --num-models=2 \
    --first-order \
    --network-architecture=CNN \
    --no-batchnorm \
    --n-way=20 \
    --k-shot=1 \
    --v-shot=15 \
    --num-epochs=60 \
    --num-episodes-per-epoch 1000 \
    --resume-epoch=0 \
    --train
