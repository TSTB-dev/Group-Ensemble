export CUDA_VISIBLE_DEVICES=0
python3 ./src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --batch_size 64 \
    --num_epochs 200 \
    --num_snapshots 10 \
    --scheduler_type cyclic_cosine \
    --weight_decay 0.0005 \
    --momentum 0.9 \
    --max_lr 0.1 \
    --min_lr 0.0 \
    --optimizer sgd \
    --cycle_length -1 \
    --cycle_length_decay 1 \
    --device cuda \
    --seed 0 \
    --log_interval 1 \
    --proj_name snapshot_ensemble
