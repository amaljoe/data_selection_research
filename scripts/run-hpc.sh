#!/bin/bash

# Set environment variables: HPC_USER, HPC_HOST, HPC_PATH

echo "Working directory: $HPC_USER@$HPC_HOST:$HPC_PATH"

# Sync files
echo "Starting Rsync..."
rsync -avz -e "ssh -p 4422" ./hpc-job.slurm $HPC_USER@$HPC_HOST:${HPC_PATH}/ || { echo "Rsync failed for hpc-job.slurm. Exiting."; exit 1; }
if [[ ! " $@ " =~ " --no-rsync " ]]; then
  rsync -avz --delete -e "ssh -p 4422" ~/workspace $HPC_USER@$HPC_HOST:${HPC_PATH}/ || { echo "Rsync failed for envs. Exiting."; exit 1; }
  rsync -avz --delete -e "ssh -p 4422" ~/miniconda3/envs $HPC_USER@$HPC_HOST:${HPC_PATH}/ || { echo "Rsync failed for workspace. Exiting."; exit 1; }
else
  echo "Skipping Rsync for envs and workspace as --no-rsync was provided."
fi

echo "Rsync completed successfully. Submitting the job..."

# Submit job
ssh -p 4422 $HPC_USER@$HPC_HOST "cd $HPC_PATH && sbatch hpc-job.slurm" | tee job_submission.log

# Extract Job ID and stream output
JOB_ID=$(grep -oP '\d+' job_submission.log | tail -1)
echo "Job submitted with ID: $JOB_ID. Waiting for output..."

LOGS_FILE=${HPC_PATH}/logs/slurm-${JOB_ID}.log
# Wait for either temp log or final log to appear
while ! ssh -p 4422 $HPC_USER@$HPC_HOST "[ -f ${LOGS_FILE} ]"; do
    echo "Waiting..."
    sleep 10
done

echo -e "Job running. Streaming output...\n\n"
JOB_COMPLETE_MSG="-----Job completed-----"
ssh -p 4422 $HPC_USER@$HPC_HOST "tail -f ${LOGS_FILE} | awk '/${JOB_COMPLETE_MSG}/ {print; exit} {print}'"