#!/bin/bash
#SBATCH --job-name=the_happening_place        # Job name
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks=1                            # Number of tasks (processes)
#SBATCH --gres=gpu:1                          # Number of GPUs per node
#SBATCH --mem=16G                             # Total memory per node
#SBATCH --time=00:02:00                       # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                       # Partition name
#SBATCH --output=logs/slurm-%j.out            # Output file

# Load Miniconda
module load miniconda

# Initialize Conda
conda init
source ~/.bashrc

PLACE="$HOME/the_happening_place"

# Create or activate environment in home directory
ENV_NAME="llm-general2"
ENV_PATH="$PLACE/envs/$ENV_NAME"

if [ -d "$ENV_PATH" ]; then
    echo "Activating existing environment: $ENV_NAME"
    conda activate $ENV_PATH
else
    echo "Creating new environment at $ENV_PATH"
    conda create -y -p $ENV_PATH python=3.10
    conda activate $ENV_PATH
fi

# Navigate to project workspace
PROJECT="data_selection_research"
cd $PLACE/workspace/$PROJECT

SCRIPT="run.sh"

# Run the script and log results
LOGFILE=${PLACE}/logs/slurm-${SLURM_JOB_ID}.log
echo "-----Job started-----" > $LOGFILE
bash $SCRIPT >> $LOGFILE 2>&1
echo "-----Job completed-----" >> $LOGFILE

echo "Job completed"
