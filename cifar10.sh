save_dir="."

devices="0"
port=7296
n_gpu=1

backPruneRatio=0.9
lr=0.01

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name cifar10-lr${lr}-B128-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-Ti_16 --pretrained_dir ${save_dir}/pretrain/ViT-Ti_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 128 --eval_batch_size 32 \
--num_steps 3000 --eval_every 500