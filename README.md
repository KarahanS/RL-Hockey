#### :star: If you found this repository useful, please consider giving it a star to show your support. Thanks! ####

## Installation

```bash
git clone https://github.com/KarahanS/RL-Hockey.git .               # clone the repository
cd RL-Hockey                                                        # cd into the repository
python -m venv env                                                  # create a virtual environment
source env/bin/activate                                             # activate the venv
pip install -r requirements.txt                                     # install the requirements
pip install torch torchvision torchaudio                            # install a suitable torch version
```

## Advanced Reinforcement Learning Algorithms for Hockey Game Environment

This repository implements three state-of-the-art reinforcement learning algorithms for the hockey-like game environment from the University of Tübingen's Reinforcement Learning course (Winter 24/25):

1. **Soft Actor-Critic (SAC)** - Implemented by [Karahan Sarıtaş](https://github.com/KarahanS)
2. **Deep Q-Network (DQN)** - Implemented by [Kıvanç Tezören](https://github.com/kivanctezoren)
3. **Twin Delayed Deep Deterministic Policy Gradient (TD3)** - Implemented by [Oğuz Ata Çal](https://github.com/OguzAtaCal)

## Notable Achievement

Our SAC implementation (Muhteshember-SAC) ranked 5th among 140+ competing agents in the official [tournament leaderboard](https://comprl.cs.uni-tuebingen.de/leaderboard/). In a nutshell, we used automatic entropy tuning, pink noise, augmented custom reward, mirrored state-action pairs, and a prioritized opponent buffer with discounted upper confidence bound (Discounted-UCB) sampling for self-play training. We trained our model in stages, starting with matches against the strong opponent (bot). We then expanded to self-play training that included weak, strong, shooting and defense modes, as well as matches against our other trained models to provide diverse learning experiences.

## Documentation

For comprehensive documentation including:
- Detailed algorithm descriptions
- Extensive ablation and sensitivity studies on hyperparameters
- Implementation techniques crucial to our competitive performance

Please refer to our [technical report](https://github.com/KarahanS/RL-Hockey/blob/main/assets/RL_Course_2024_25_Final_Project_Report.pdf).
