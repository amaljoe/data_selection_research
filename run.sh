#CUDA_VISIBLE_DEVICES=3,1 torchrun --nproc_per_node=2 parallel_inference.py

#srun torchrun \
#  --nnodes=$SLURM_NNODES \
#  --nproc_per_node=$SLURM_GPUS_ON_NODE \
#  --node_rank=$SLURM_NODEID \
#  --rdzv_id=$SLURM_JOB_ID \
#  --rdzv_backend=c10d \
#  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#  --max_restarts=3 \
#  parallel_inference.py

rsync -az --delete -e ssh amal@10.129.6.206:/ide_docker/workspace ~/the_happening_place/workspace
python3 test.py
