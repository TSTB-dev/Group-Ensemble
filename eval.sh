export CUDA_VISIBLE_DEVICES=0
python3 ./src/evaluate.py \
    --dataset cifar10 \
    --model resnet18 \
    --num_ensembles 8 \
    --device cuda \
    --eval_single \
    --batch_size 64 \
    --save_dir # TODO: save_dir