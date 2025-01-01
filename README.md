# hockey-env

This repository contains a hockey-like game environment for RL

## Install

``python3 -m pip install git+https://github.com/martius-lab/hockey-env.git``

or add the following line to your Pipfile

``hockey = {editable = true, git = "https://git@github.com/martius-lab/hockey-env.git"}``


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

