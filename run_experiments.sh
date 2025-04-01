#!/bin/bash

# Experiment 1
echo "Starting experiment 1..."
CUDA_VISIBLE_DEVICES=0 python3 end2end.py --method=random --train_seed=49 &
PID1=$!

# Experiment 2
echo "Starting experiment 2..."
CUDA_VISIBLE_DEVICES=1 python3 end2end.py --method=delift-se --train_seed=49 &
PID2=$!

# Wait for both experiments to finish
wait $PID1
echo "Experiment 1 finished."

wait $PID2
echo "Experiment 2 finished."

# Queue for subsequent experiments
# Experiment 3 (queued)
echo "Starting experiment 3..."
CUDA_VISIBLE_DEVICES=0 python3 end2end.py --method=full --train_seed=49
echo "Experiment 3 finished."
