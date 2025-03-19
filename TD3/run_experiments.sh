#!/bin/bash
# run_noise_experiment.sh
# This script runs noise experiments on the hockey environment.
#
# It performs several groups of experiments (each for 5000 episodes against a strong opponent):
#
# Group 0: No noise enabled.
#
# Group 0.5: OU noise.
#
# Group 1: Pink noise experiments (exponent fixed to 1) with three different low-frequency cutoffs (fmin).
#          These experiments test the effect of different fmin values for pink noise.
#
# Group 2: Experiments with different exponents (thus different noise colors) while using a fixed fmin.
#          For example, an exponent of 0.5 (shallower than pink noise) and 1.5 (steeper, akin to brown noise)
#          are run.
#
# Both RND and layer normalization are disabled in all experiments.
#
# Make sure the script is executable:
#   chmod +x run_noise_experiment.sh

# -----------------------------
# Group 0: Gaussian noise
# -----------------------------
echo "Running experiment 1: No noise (expl_noise_type set to 'gaussian')"
python hockey-run.py \
    --mode vs_weak \
    --episodes 15000 \
    --seed 42 \
    --save_model \
    --expl_noise_type gaussian \
    --agent_class src.td3:TD3 \
    --no_rnd \
    --no_layer_norm

# -----------------------------
# Group 0.5: OU noise
# -----------------------------
echo "Running experiment 1.5: OU noise"
python hockey-run.py \
    --mode vs_weak \
    --episodes 15000 \
    --seed 43 \
    --save_model \
    --expl_noise_type ou \
    --agent_class src.td3:TD3 \
    --no_rnd \
    --no_layer_norm

# -----------------------------
# Group 1: Pink noise (exponent = 1) with varying fmin values
# -----------------------------
echo "Group 1: Pink noise (exponent=1) with different fmin values"

# Experiment 2: fmin = 0.0
echo "Running experiment 2: Pink noise with exponent=1 and pink_noise_fmin=0.0"
python hockey-run.py \
    --mode vs_weak \
    --episodes 15000 \
    --seed 44 \
    --save_model \
    --expl_noise_type pink \
    --expl_noise 0.1 \
    --pink_noise_exponent 1.0 \
    --pink_noise_fmin 0.0 \
    --agent_class src.td3:TD3 \
    --no_rnd \
    --no_layer_norm

# Experiment 3: fmin = 0.1
echo "Running experiment 3: Pink noise with exponent=1 and pink_noise_fmin=0.1"
python hockey-run.py \
    --mode vs_weak \
    --episodes 15000 \
    --seed 45 \
    --save_model \
    --expl_noise_type pink \
    --expl_noise 0.1 \
    --pink_noise_exponent 1.0 \
    --pink_noise_fmin 0.1 \
    --agent_class src.td3:TD3 \
    --no_rnd \
    --no_layer_norm

# Experiment 4: fmin = 0.2
echo "Running experiment 4: Pink noise with exponent=1 and pink_noise_fmin=0.2"
python hockey-run.py \
    --mode self_play \
    --episodes 15000 \
    --seed 46 \
    --save_model \
    --expl_noise_type pink \
    --expl_noise 0.1 \
    --pink_noise_exponent 1.0 \
    --pink_noise_fmin 0.2 \
    --agent_class src.td3:TD3 \
    --no_rnd \
    --no_layer_norm

# -----------------------------
# Group 2: Different exponents (different noise colors) with fixed fmin
# -----------------------------
echo "Group 2: Different exponents (noise colors) with fixed pink_noise_fmin=0.0"

# Experiment 5: Exponent = 0.5 (shallower than pink noise)
echo "Running experiment 5: Noise with exponent=0.5 and pink_noise_fmin=0.0"
python hockey-run.py \
    --mode vs_weak \
    --episodes 15000 \
    --seed 47 \
    --save_model \
    --expl_noise_type pink \
    --expl_noise 0.1 \
    --pink_noise_exponent 0.5 \
    --pink_noise_fmin 0.0 \
    --agent_class src.td3:TD3 \
    --no_rnd \
    --no_layer_norm

# Experiment 6: Exponent = 1.5 (steeper, akin to brown noise)
echo "Running experiment 6: Noise with exponent=1.5 and pink_noise_fmin=0.0"
python hockey-run.py \
    --mode vs_weak \
    --episodes 15000 \
    --seed 48 \
    --save_model \
    --expl_noise_type pink \
    --expl_noise 0.1 \
    --pink_noise_exponent 1.5 \
    --pink_noise_fmin 0.0 \
    --agent_class src.td3:TD3 \
    --no_rnd \
    --no_layer_norm

echo "All experiments completed."
