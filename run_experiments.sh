#!/bin/bash

# Experiment 1
echo "Starting experiment 9..."
CUDA_VISIBLE_DEVICES=3 python3 end2end.py --method=delift-se --train_seed=50 > cache/logs/experiment9.log 2>&1 &
PID1=$!

# Experiment 2
echo "Starting experiment 10..."
CUDA_VISIBLE_DEVICES=1 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=initial --train_seed=49 > cache/logs/experiment10.log 2>&1 &
PID2=$!

# Wait for both experiments to finish
wait $PID1
echo "Experiment 9 finished."

wait $PID2
echo "Experiment 10 finished."

# Queue for subsequent experiments
# Experiment 3
echo "Starting experiment 11..."
CUDA_VISIBLE_DEVICES=0 python3 end2end.py --method=random --train_seed=49 > cache/logs/experiment11.log 2>&1 &
PID3=$!

# Experiment 4
echo "Starting experiment 12..."
CUDA_VISIBLE_DEVICES=1 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=full --train_seed=49 > cache/logs/experiment12.log 2>&1  &
PID4=$!

# Wait for both experiments to finish
wait $PID3
echo "Experiment 11 finished."

wait $PID4
echo "Experiment 12 finished."