#CUDA_VISIBLE_DEVICES=3,1 torchrun --nproc_per_node=2 parallel_inference.py

torchrun --nnodes=$SLURM_NNODES \
         --nproc_per_node=$SLURM_GPUS_ON_NODE \
         --node_rank=$SLURM_NODEID \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         parallel_inference.py