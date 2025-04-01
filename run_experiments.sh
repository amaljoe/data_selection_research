#!/bin/bash

# Experiment 1
echo "Starting experiment 1..."
CUDA_VISIBLE_DEVICES=0 python3 end2end.py --method=random --train_seed=49 > cache/logs/experiment1.log 2>&1 &
PID1=$!

# Experiment 2
echo "Starting experiment 2..."
CUDA_VISIBLE_DEVICES=1 python3 end2end.py --method=delift-se --train_seed=49 > cache/logs/experiment2.log 2>&1 &
PID2=$!

# Wait for both experiments to finish
wait $PID1
echo "Experiment 1 finished."

wait $PID2
echo "Experiment 2 finished."

# Queue for subsequent experiments
# Experiment 3
echo "Starting experiment 3..."
CUDA_VISIBLE_DEVICES=0 python3 end2end.py --method=full --train_seed=49 > cache/logs/experiment3.log 2>&1 &
PID3=$!

# Experiment 4
echo "Starting experiment 4..."
CUDA_VISIBLE_DEVICES=1 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=initial --train_seed=49 > cache/logs/experiment4.log 2>&1  &
PID4=$!

# Wait for both experiments to finish
wait $PID3
echo "Experiment 3 finished."

wait $PID4
echo "Experiment 4 finished."