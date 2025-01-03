# hockey-env

This repository contains a hockey-like game environment for RL

## Installation

```bash
git clone https://github.com/KarahanS/RL-Hockey.git .               # clone the repository
cd RL-Hockey                                                        # cd into the repository
python -m venv env                                                  # create a virtual environment
source env/bin/activate                                             # activate the venv
pip install -r requirements.txt                                     # install the requirements
pip install torch torchvision torchaudio                            # install a suitable torch version
```
## HockeyEnv

![Screenshot](assets/hockeyenv1.png)

``hockey.hockey_env.HockeyEnv``

A two-player (one per team) hockey environment.
For our Reinforcment Learning Lecture @ Uni-Tuebingen.
See Hockey-Env.ipynb notebook on how to run the environment.

The environment can be generated directly as an object or via the gym registry:

``env = gym.envs.make("Hockey-v0")``

There is also a version against the basic opponent (with options)

``env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)``

