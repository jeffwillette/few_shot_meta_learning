ROOT=$DATADIR
GPUS=(2 2 2)
DATASET="omniglot"

for RUN in 0 1 2
do
  CUDA_VISIBLE_DEVICES=${GPUS[RUN]} PYTHONPATH=. python main.py \
    --datasource=$DATASET \
    --ds-folder $ROOT \
    --ml-algorithm=vampire \
    --num-models=2 \
    --first-order \
    --network-architecture=CNN \
    --no-batchnorm \
    --n-way=5 \
    --k-shot=5 \
    --v-shot=15 \
    --num-epochs=40 \
    --num-episodes-per-epoch 10000 \
    --resume-epoch=0 \
    --train
done
