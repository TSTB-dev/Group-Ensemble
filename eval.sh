export CUDA_VISIBLE_DEVICES=0
python3 ./src/evaluate.py \
    --dataset cifar10 \
    --model resnet18 \
    --num_ensembles 8 \
    --device cuda \
    --eval_single \
    --batch_size 64 \
    --save_dir /home/sakai/projects/NDE/Group-Ensemble/wandb/run-20240617_134019-9i05ebro/files/