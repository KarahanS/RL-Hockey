import yaml
import os
import sys
import subprocess

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)

# Ensure correct usage
if len(sys.argv) < 2:
    print("Usage: python run_training.py <config.yaml> [jid]")
    sys.exit(1)

# Load config
config_file = sys.argv[1]
config = load_config(config_file)

# Get JID from command-line argument (default None)
JID = sys.argv[2] if len(sys.argv) > 2 else None
JID = JID if (JID and JID.isdigit()) else None

NAME = config.get("name", "SAC")

# Basic training / environment args
ENV_NAME = config.get("env_name", "Pendulum-v1")
SEED = config.get("seed", 42)
LR = config.get("lr", 0.0001)
MAX_EPISODES = config.get("max_episodes", 2000)
MAX_TIMESTEPS = config.get("max_timesteps", 2000)
UPDATE_EVERY = config.get("update_every", 1)
SAVE_INTERVAL = config.get("save_interval", 500)
LOG_INTERVAL = config.get("log_interval", 20)
OUTPUT_DIR = config.get("output_dir", "./results")
DISCOUNT = config.get("discount", 0.99)
BUFFER_SIZE = config.get("buffer_size", 1000000)

# Noise parameters
NOISE = config.get("noise", {})
NOISE_TYPE = NOISE.get("type", "normal")
NOISE_SIGMA = NOISE.get("sigma", 0.1)
NOISE_THETA = NOISE.get("theta", 0.15)
NOISE_DT = NOISE.get("dt", 0.01)
NOISE_BETA = NOISE.get("beta", 1.0)
NOISE_SEQ_LEN = NOISE.get("seq_len", 1000)

# Hockey-specific
HOCKEY = config.get("hockey", {})
HOCKEY_MODE = HOCKEY.get("mode", "NORMAL")
OPPONENT_TYPE = HOCKEY.get("opponent_type", "none")
KEEP_MODE = HOCKEY.get("keep_mode", False)

# Evaluation params
EVAL_INTERVAL = config.get("eval_interval", 0)
EVAL_EPISODES = config.get("eval_episodes", 1000)

# New advanced parameters from YAML
BATCH_SIZE = config.get("batch_size", 256)
HIDDEN_SIZES_ACTOR = config.get("hidden_sizes_actor", [256, 256])
HIDDEN_SIZES_CRITIC = config.get("hidden_sizes_critic", [256, 256])
TAU = config.get("tau", 0.005)
LEARN_ALPHA = config.get("learn_alpha", True)
ALPHA = config.get("alpha", 0.2)

USE_PER = config.get("use_per", False)
USE_ERE = config.get("use_ere", False)
PER_ALPHA = config.get("per_alpha", 0.6)
PER_BETA = config.get("per_beta", 0.4)
ERE_ETA0 = config.get("ere_eta0", 0.996)
ERE_MIN_SIZE = config.get("ere_min_size", 2500)

REWARD = config.get("reward", "basic")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

CMD_EXTRA = ["--hockey_mode", HOCKEY_MODE, "--opponent_type", OPPONENT_TYPE]
if KEEP_MODE:
    CMD_EXTRA.append("--keep_mode")
    
# Construct the command to call your trainer
CMD = [
    "python", "src/hockey_trainer.py",
    "--name", NAME,
    "--env_name", ENV_NAME,
    "--seed", str(SEED),
    "--lr", str(LR),
    "--max_episodes", str(MAX_EPISODES),
    "--max_timesteps", str(MAX_TIMESTEPS),
    "--update_every", str(UPDATE_EVERY),
    "--save_interval", str(SAVE_INTERVAL),
    "--log_interval", str(LOG_INTERVAL),
    "--output_dir", OUTPUT_DIR,
    "--noise_type", NOISE_TYPE,
    "--noise_sigma", str(NOISE_SIGMA),
    "--noise_theta", str(NOISE_THETA),
    "--noise_dt", str(NOISE_DT),
    "--noise_beta", str(NOISE_BETA),
    "--noise_seq_len", str(NOISE_SEQ_LEN),
    "--discount", str(DISCOUNT),
    "--buffer_size", str(BUFFER_SIZE),
    "--eval_interval", str(EVAL_INTERVAL),
    "--eval_episodes", str(EVAL_EPISODES),

    # Now pass the advanced parameters
    "--batch_size", str(BATCH_SIZE),
    # We'll convert lists to comma-separated strings
    "--hidden_sizes_actor", ",".join(map(str, HIDDEN_SIZES_ACTOR)),
    "--hidden_sizes_critic", ",".join(map(str, HIDDEN_SIZES_CRITIC)),
    "--tau", str(TAU),
    "--learn_alpha", str(LEARN_ALPHA),
    "--alpha", str(ALPHA),
    
    "--per_alpha", str(PER_ALPHA),
    "--per_beta", str(PER_BETA),
    "--ere_eta0", str(ERE_ETA0),
    "--ere_min_size", str(ERE_MIN_SIZE),
    "--reward", str(REWARD),
] + CMD_EXTRA

if USE_PER:
    CMD.append("--use_per")
if USE_ERE:
    CMD.append("--use_ere")
    
if JID:
    CMD += ["--id", JID]

process = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
actual_id = JID if JID else str(process.pid)

LOG_FILE = os.path.join(OUTPUT_DIR, f"{actual_id}_training-log.txt")
ERROR_LOG = os.path.join(OUTPUT_DIR, f"{actual_id}_error-log.txt")

with open(LOG_FILE, "w") as log, open(ERROR_LOG, "w") as err:
    log.write(f"==== Training started with ID: {actual_id} ====\n")
    err.write(f"==== Training started with ID: {actual_id} ====\n")

    print(f"✅ Training started (ID: {actual_id})")
    print(f"Logs: {LOG_FILE}")
    print(f"Errors: {ERROR_LOG}")

    for line in process.stdout:
        log.write(line)
        log.flush()

    for line in process.stderr:
        err.write(line)
        err.flush()

process.wait()

if process.returncode == 0:
    print(f"✅ Training completed successfully. Logs: {LOG_FILE}")
else:
    print(f"❌ Training failed (ID: {actual_id}). Check logs: {ERROR_LOG}")
