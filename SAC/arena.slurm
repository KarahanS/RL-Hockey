#!/bin/bash

#SBATCH --job-name=arena_advanced
#SBATCH --cpus-per-task=4         # Maximum 24 per node
#SBATCH --partition=week   
#SBATCH --mem-per-cpu=3G          # Maximum memory per CPU
#SBATCH --gres=gpu:1              # Request one GPU (adjust if necessary)
#SBATCH --time=72:00:00           # Maximum runtime: 72 hours
#SBATCH --error=slurm/%J-%x.err
#SBATCH --output=slurm/%J-%x.out


# Full path to your Singularity container.
CONTAINER="/home/stud438/RL-Hockey/SAC/container.sif"

# Full path to your tournament wrapper script.
ARENA_SCRIPT="/home/stud438/RL-Hockey/SAC/run_arena.py"

echo "Job started on $(hostname) at $(date)"
echo "Current working directory: $(pwd)"

# Ensure Singularity container exists.
echo "Checking Singularity container..."
ls -lh "$CONTAINER" || { echo "Error: Singularity container missing"; exit 1; }

# Check Python version inside Singularity.
echo "Checking Python installation inside Singularity..."
singularity run "$CONTAINER" python3 --version

# Change to your working directory.
cd /home/stud438/RL-Hockey/SAC || { echo "Error: Directory not found"; exit 1; }
mkdir -p slurm

# Run the arena_advanced.py script inside the Singularity container.
echo "Running arena.py with Job ID: $SLURM_JOB_ID"
singularity run "$CONTAINER" python3 "$ARENA_SCRIPT" "$SLURM_JOB_ID"

echo "Job completed at $(date)"
