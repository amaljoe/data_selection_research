#!/bin/bash

# Experiment 1
echo "Starting experiment 13..."
CUDA_VISIBLE_DEVICES=0 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=full --train_seed=50 > cache/logs/experiment13.log 2>&1 &
PID1=$!

# Experiment 2
echo "Starting experiment 14..."
CUDA_VISIBLE_DEVICES=1 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=delift-se --train_seed=50 > cache/logs/experiment14.log 2>&1 &
PID2=$!

# Wait for both experiments to finish
wait $PID1
echo "Experiment 13 finished."

wait $PID2
echo "Experiment 14 finished."