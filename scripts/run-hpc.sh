#!/bin/bash

# Set environment variables: HPC_USER, HPC_HOST, HPC_PATH

echo "Working directory: $HPC_USER@$HPC_HOST:$HPC_PATH"

# Sync files
echo "Starting Rsync..."
rsync -avz -e "ssh -p 4422" ./hpc-job.slurm $HPC_USER@$HPC_HOST:${HPC_PATH}/ || { echo "Rsync failed for hpc-job.slurm. Exiting."; exit 1; }
if [[ ! " $@ " =~ " --no-rsync " ]]; then
  echo "Syncing workspace"
  rsync -avz --delete -e "ssh -p 4422" ~/workspace $HPC_USER@$HPC_HOST:${HPC_PATH}/ || { echo "Rsync failed for envs. Exiting."; exit 1; }
#  rsync -avz --delete -e "ssh -p 4422" ~/miniconda3/envs $HPC_USER@$HPC_HOST:${HPC_PATH}/ || { echo "Rsync failed for workspace. Exiting."; exit 1; }
  echo "Syncing .cache directory..."
  rsync -avz --delete -e "ssh -p 4422" ~/.cache/huggingface/ $HPC_USER@$HPC_HOST:.cache/huggingface/ || { echo "Rsync failed for .cache. Exiting."; exit 1; }
else
  echo "Skipping Rsync for envs and workspace as --no-rsync was provided."
fi

echo "Rsync completed successfully. Submitting the job..."

# Submit job
ssh -p 4422 $HPC_USER@$HPC_HOST "cd $HPC_PATH && sbatch hpc-job.slurm" | tee job_submission.log

# Extract Job ID and stream output
JOB_ID=$(grep -oP '\d+' job_submission.log | tail -1)
echo "Job submitted with ID: $JOB_ID. Waiting for output..."

# Trap interrupt signal which will cancel job (ctrl+c)
trap "echo -e '\nInterrupt detected. Cancelling job $JOB_ID...'; ssh -p 4422 $HPC_USER@$HPC_HOST 'scancel $JOB_ID'; exit 1" SIGINT

LOGS_FILE=${HPC_PATH}/logs/slurm-${JOB_ID}.log
# Wait for either temp log or final log to appear
while ! ssh -p 4422 $HPC_USER@$HPC_HOST "[ -f ${LOGS_FILE} ]"; do
    echo "Waiting..."
    sleep 10
done

echo -e "Job running. Streaming output...\n"
JOB_COMPLETE_MSG="-----Job completed-----"
#ssh -p 4422 $HPC_USER@$HPC_HOST "tail -n +1 -f ${LOGS_FILE} | awk '/${JOB_COMPLETE_MSG}/ {print; exit} {print}'"
#ssh -p 4422 $HPC_USER@$HPC_HOST "tail -n +1 -f ${LOGS_FILE} | awk '{print; fflush()} /${JOB_COMPLETE_MSG}/ {print; exit}'"
#ssh -p 4422 $HPC_USER@$HPC_HOST "tail -n +1 -f ${LOGS_FILE} | awk '{print} /${JOB_COMPLETE_MSG}/ {print; exit}'"
#ssh -p 4422 $HPC_USER@$HPC_HOST "tail -n +1 -f ${LOGS_FILE} | grep --line-buffered -e '.*' -e '${JOB_COMPLETE_MSG}' && echo '${JOB_COMPLETE_MSG}'"
#ssh -p 4422 $HPC_USER@$HPC_HOST "tail -n +1 -f ${LOGS_FILE} | stdbuf -o0 grep -e '.*' -e '${JOB_COMPLETE_MSG}' && echo '${JOB_COMPLETE_MSG}'"
ssh -p 4422 $HPC_USER@$HPC_HOST "tail -n +1 -f ${LOGS_FILE}"


echo -e "\nJob completed. Syncing back workspace files..."
rsync -avz --ignore-existing -e "ssh -p 4422" $HPC_USER@$HPC_HOST:${HPC_PATH}/workspace/ ~/workspace/ || { echo "Rsync back failed. Exiting."; exit 1; }

echo "Workspace files synced back successfully."
