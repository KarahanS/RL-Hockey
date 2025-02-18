#!/bin/bash

#SBATCH --job-name=sac_elo
#SBATCH --cpus-per-task=4        # Adjust CPU usage
#SBATCH --partition=week         # or whichever partition
#SBATCH --mem-per-cpu=3G         # total memory depends on cpus
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --time=48:00:00          # 48 hours
#SBATCH --error=slurm/%J-%x.err
#SBATCH --output=slurm/%J-%x.out

# Adjust these paths as needed:
MODEL_FOLDER="/home/stud438/RL-Hockey/SAC/slurm" 
AGENT_CONFIG="/home/stud438/RL-Hockey/SAC/slurm" 
CONTAINER="/home/stud438/RL-Hockey/SAC/container.sif"

# Additional "episodes_per_match"
EPISODES_PER_MATCH=30

echo "Job started on $(hostname) at $(date)"
echo "Current working directory: $(pwd)"

# Check existence of model folder & config
echo "Checking if model folder and agent config exist..."
ls -lh "$MODEL_FOLDER" || { echo "Error: model folder not found"; exit 1; }
ls -lh "$AGENT_CONFIG" || { echo "Error: agent config not found"; exit 1; }

# HPC environment: go to your project directory
cd /home/stud438/RL-Hockey/SAC || { echo "Error: Directory not found"; exit 1; }
mkdir -p slurm

# Check Singularity container
echo "Checking Singularity container..."
ls -lh "$CONTAINER" || { echo "Error: Singularity container missing"; exit 1; }

# Display Python version inside the container
echo "Checking Python inside container..."
singularity run "$CONTAINER" python3 --version

# We'll do a final run of evaluate_elo.py. 
# We pass your model folder, agent config, optional environment mode, episodes per match, etc.
echo "Running ELO evaluation with Job ID: $SLURM_JOB_ID"
singularity run "$CONTAINER" python3 src/evaluate_elo.py \
    --model_folder "$MODEL_FOLDER" \
    --agent_config "$AGENT_CONFIG" \
    --env_mode NORMAL \
    --episodes_per_match "$EPISODES_PER_MATCH"

# Suppose evaluate_elo.py sorts final players and writes:
#   slurm/match_results.json
#   slurm/final_ratings.txt

# We can optionally do a quick summary. 
# Count how many .pth files found => plus 2 built-in opponents => total players => match count
NUM_MODELS=$(ls -1 "$MODEL_FOLDER"/*.pth 2>/dev/null | wc -l)
if [ "$NUM_MODELS" -gt 0 ]; then
    # total players = NUM_MODELS + 2
    PLAYERS=$((NUM_MODELS + 2))
    MATCHES=$((PLAYERS*(PLAYERS-1)/2))
    echo "We had $PLAYERS total players => $MATCHES matchups"
    echo "Each matchup was $EPISODES_PER_MATCH episodes => total episodes = $((MATCHES*EPISODES_PER_MATCH))"
fi

echo "Job completed at $(date)"
