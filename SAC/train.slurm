#!/bin/bash

#SBATCH --job-name=sac
#SBATCH --cpus-per-task=4  # max 24 per node
#SBATCH --partition=day   
#SBATCH --mem-per-cpu=3G   # max 251GB per node
#SBATCH --gres=gpu:1        # each node has 4 gpus
#SBATCH --time=24:00:00  # 24 hours
#SBATCH --error=slurm/%J-%x.err
#SBATCH --output=slurm/%J-%x.out

CONFIG_FILE="/home/stud438/RL-Hockey/SAC/config.yaml"
CONTAINER="/home/stud438/RL-Hockey/SAC/container.sif"

echo "Job started on $(hostname) at $(date)"
echo "Current working directory: $(pwd)"

# Ensure config.yaml exists
echo "Checking if config.yaml exists..."
ls -lh "$CONFIG_FILE" || { echo "Error: config.yaml not found"; exit 1; }

cd /home/stud438/RL-Hockey/SAC || { echo "Error: Directory not found"; exit 1; }
mkdir -p slurm

# Ensure container exists
echo "Checking Singularity container..."
ls -lh "$CONTAINER" || { echo "Error: Singularity container missing"; exit 1; }

# Check Python version inside Singularity
echo "Checking Python installation inside Singularity..."
singularity run "$CONTAINER" python3 --version

# Run training script inside Singularity, passing the SLURM Job ID
echo "Running training script with Job ID: $SLURM_JOB_ID"
singularity run "$CONTAINER" python3 run_training.py "$CONFIG_FILE" "$SLURM_JOB_ID"

echo "Job completed at $(date)"
