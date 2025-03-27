ssh -t -p 4422 $HPC_USER@$HPC_HOST "srun --job-name=the_happening_place \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --mem=48G \
     --time=04:00:00 \
     --partition=gpu \
     --pty bash"

