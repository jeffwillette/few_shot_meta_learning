ROOT=$DATADIR
GPUS=(0 0 0)
DATASET="miniimagenet"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --ml-algorithm=vampire \
    --num-models=2 \
    --first-order \
    --network-architecture=CNN \
    --no-batchnorm \
    --n-way=5 \
    --k-shot=1 \
    --v-shot=15 \
    --num-epochs=1 \
    --num-episodes-per-epoch 10 \
    --resume-epoch=0 \
    --train
