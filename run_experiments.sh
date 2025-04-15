#!/bin/bash

# Experiment 1
echo "Starting experiment 15..."
CUDA_VISIBLE_DEVICES=0 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=delift-se --train_seed=49 --generation_batch_size=1 > cache/logs/experiment15.log 2>&1 &
PID1=$!

# Experiment 2
echo "Starting experiment 16..."
CUDA_VISIBLE_DEVICES=1 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=initial --train_seed=49 --generation_batch_size=1 > cache/logs/experiment16.log 2>&1 &
PID2=$!


# Queue for subsequent experiments
# Experiment 3
echo "Starting experiment 17..."
CUDA_VISIBLE_DEVICES=2 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=random --train_seed=49 --generation_batch_size=1 > cache/logs/experiment17.log 2>&1 &
PID3=$!

# Experiment 4
echo "Starting experiment 18..."
CUDA_VISIBLE_DEVICES=3 python3 end2end.py --model=microsoft/Phi-3-mini-128k-instruct --method=full --train_seed=49 --generation_batch_size=1 > cache/logs/experiment18.log 2>&1  &
PID4=$!

# Wait for both experiments to finish
wait $PID1
echo "Experiment 15 finished."

wait $PID2
echo "Experiment 16 finished."


# Wait for both experiments to finish
wait $PID3
echo "Experiment 17 finished."

wait $PID4
echo "Experiment 18 finished."