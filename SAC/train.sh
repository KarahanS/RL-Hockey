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
# Noise parameters
NOISE_TYPE="normal"
NOISE_SIGMA=0.1
NOISE_THETA=0.15
NOISE_DT=0.01
NOISE_BETA=1.0
NOISE_SEQ_LEN=1000

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
        # Noise parameters
        --noise_type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        --noise_sigma)
            NOISE_SIGMA="$2"
            shift 2
            ;;
        --noise_theta)
            NOISE_THETA="$2"
            shift 2
            ;;
        --noise_dt)
            NOISE_DT="$2"
            shift 2
            ;;
        --noise_beta)
            NOISE_BETA="$2"
            shift 2
            ;;
        --noise_seq_len)
            NOISE_SEQ_LEN="$2"
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
    --output_dir $OUTPUT_DIR \
    --noise_type $NOISE_TYPE \
    --noise_sigma $NOISE_SIGMA \
    --noise_theta $NOISE_THETA \
    --noise_dt $NOISE_DT \
    --noise_beta $NOISE_BETA \
    --noise_seq_len $NOISE_SEQ_LEN"

# Add optional flags if enabled
if [ "$USE_PER" = true ]; then
    CMD="$CMD --use_per --per_alpha $PER_ALPHA --per_beta $PER_BETA"
fi

if [ "$USE_ERE" = true ]; then
    CMD="$CMD --use_ere --ere_eta0 $ERE_ETA0 --ere_min_size $ERE_MIN_SIZE"
fi

# Create error log file with PID
mkdir -p "${OUTPUT_DIR}"
ERROR_LOG="${OUTPUT_DIR}/error_pid_$$.log"

# Run the command in background and redirect stderr to error log
eval $CMD 2> "$ERROR_LOG" > /dev/null &

# Save the process ID
PID=$!
echo "Training started in background. PID: $PID"
echo "Command executed: $CMD"
echo "Logs will be saved in the experiment's results directory"
echo "Check ${ERROR_LOG} for any errors"
