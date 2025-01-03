#!/bin/bash

# Default values
ENV_NAME="Pendulum-v1"
SEED=42
LR=0.0001
MAX_EPISODES=2000
MAX_TIMESTEPS=2000
LOSS_TYPE="mse"
UPDATE_EVERY=1
USE_PER=false
USE_ERE=false
PER_ALPHA=0.6
PER_BETA=0.4
ERE_ETA0=0.996
ERE_MIN_SIZE=2500
SAVE_INTERVAL=500
LOG_INTERVAL=20
OUTPUT_DIR="./results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --max_episodes)
            MAX_EPISODES="$2"
            shift 2
            ;;
        --max_timesteps)
            MAX_TIMESTEPS="$2"
            shift 2
            ;;
        --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --update_every)
            UPDATE_EVERY="$2"
            shift 2
            ;;
        --use_per)
            USE_PER=true
            shift
            ;;
        --use_ere)
            USE_ERE=true
            shift
            ;;
        --per_alpha)
            PER_ALPHA="$2"
            shift 2
            ;;
        --per_beta)
            PER_BETA="$2"
            shift 2
            ;;
        --ere_eta0)
            ERE_ETA0="$2"
            shift 2
            ;;
        --ere_min_size)
            ERE_MIN_SIZE="$2"
            shift 2
            ;;
        --save_interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --log_interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter $1"
            exit 1
            ;;
    esac
done

# Construct the command with all parameters
CMD="python trainer.py \
    --env_name $ENV_NAME \
    --seed $SEED \
    --lr $LR \
    --max_episodes $MAX_EPISODES \
    --max_timesteps $MAX_TIMESTEPS \
    --loss_type $LOSS_TYPE \
    --update_every $UPDATE_EVERY \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --output_dir $OUTPUT_DIR"

# Add optional flags if enabled
if [ "$USE_PER" = true ]; then
    CMD="$CMD --use_per --per_alpha $PER_ALPHA --per_beta $PER_BETA"
fi

if [ "$USE_ERE" = true ]; then
    CMD="$CMD --use_ere --ere_eta0 $ERE_ETA0 --ere_min_size $ERE_MIN_SIZE"
fi

# Run the command and capture output
eval $CMD | tee >(while read line; do
    # Extract the run directory from the output (it will be printed at the start)
    if [[ $line == *"results/"* ]]; then
        run_dir=$(echo "$line" | grep -o "results/[^[:space:]]*")
        # Create training_log.txt in the run directory
        echo "$line" >> "${run_dir}/training_log.txt"
    else
        # If run_dir exists, append to its log file
        if [ ! -z "$run_dir" ]; then
            echo "$line" >> "${run_dir}/training_log.txt"
        fi
    fi
done) &

# Save the process ID
echo $! > training.pid

echo "Training started in background. PID saved in training.pid"
echo "Command executed: $CMD"
echo "Logs will be saved in the experiment's results directory"