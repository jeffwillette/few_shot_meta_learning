ROOT=$DATADIR
GPUS=(0 0 0)
DATASET="omniglot"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --ml-algorithm=MAML \
    --num-models=1 \
    --first-order \
    --network-architecture=CNN \
    --no-batchnorm \
    --n-way=20 \
    --k-shot=5 \
    --v-shot=15 \
    --num-epochs=1 \
    --num-episodes-per-epoch 100 \
    --resume-epoch=0 \
    --train
