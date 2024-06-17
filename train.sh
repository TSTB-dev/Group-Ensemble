export CUDA_VISIBLE_DEVICES=0
python3 ./src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --batch_size 64 \
    --num_epochs 200 \
    --num_ensembles 8 \
    --scheduler_type cyclic_cosine \
    --weight_decay 0.0005 \
    --momentum 0.9 \
    --max_lr 0.1 \
    --min_lr 0.0 \
    --optimizer sgd \
    --device cuda \
    --seed 0 \
    --log_interval 100 \
    --proj_name group_ensemble \
    --save_dir models
