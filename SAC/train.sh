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

# Hockey-specific defaults (only used if ENV_NAME contains "Hockey")
HOCKEY_MODE="NORMAL"  # NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE
OPPONENT_TYPE="none"  # none, basic, weak_basic, human

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        # Hockey-specific parameters (only used for hockey environments)
        --hockey_mode)
            HOCKEY_MODE="$2"
            shift 2
            ;;
        --opponent_type)
            OPPONENT_TYPE="$2"
            shift 2
            ;;
        # Standard parameters
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

# Handle Hockey-specific environment setup if needed
if [[ $ENV_NAME == *"Hockey"* ]]; then
    # Determine the environment name based on opponent type
    case $OPPONENT_TYPE in
        "none")
            ENV_NAME="Hockey-v0"
            ;;
        "basic" | "weak_basic")
            ENV_NAME="Hockey-One-v0"
            ;;
        "human")
            ENV_NAME="Hockey-v0"  # Human opponent uses base environment
            ;;
        *)
            echo "Invalid opponent type: $OPPONENT_TYPE"
            exit 1
            ;;
    esac

    # Convert hockey mode to numeric value and validate
    case $HOCKEY_MODE in
        "NORMAL")
            MODE_VALUE=0
            ;;
        "TRAIN_SHOOTING")
            MODE_VALUE=1
            ;;
        "TRAIN_DEFENSE")
            MODE_VALUE=2
            ;;
        *)
            echo "Invalid hockey mode: $HOCKEY_MODE"
            exit 1
            ;;
    esac
    
    CMD_EXTRA="--hockey_mode $HOCKEY_MODE --opponent_type $OPPONENT_TYPE"
else
    CMD_EXTRA=""
fi

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
    --noise_seq_len $NOISE_SEQ_LEN \
    $CMD_EXTRA"

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

# Print configuration
echo "Starting training with:"
echo "Environment: $ENV_NAME"
if [[ $ENV_NAME == *"Hockey"* ]]; then
    echo "Mode: $HOCKEY_MODE"
    echo "Opponent: $OPPONENT_TYPE"
fi
echo "Seed: $SEED"
echo "Learning Rate: $LR"

# Run the command in background and redirect stderr to error log
eval $CMD 2> "$ERROR_LOG" > /dev/null &

# Save the process ID
PID=$!
echo "Training started in background. PID: $PID"
echo "Command executed: $CMD"
echo "Logs will be saved in the experiment's results directory"
echo "Check ${ERROR_LOG} for any errors"