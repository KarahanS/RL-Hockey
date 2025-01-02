import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle
from memory import ReplayMemory, PrioritizedExperienceReplay
from policies import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def select_loss_function(loss_type):
    """
    Select and return the appropriate loss function based on the input string.

    Args:
        loss_type (str): Type of loss function to use

    Returns:
        callable: Loss function to be used for critic training
    """
    loss_functions = {
        "mse": F.mse_loss,
        "huber": F.smooth_l1_loss,
        "mae": F.l1_loss,
        "mse_weighted": lambda input, target: torch.mean(
            F.mse_loss(input, target, reduction="none") * 1.0
        ),
    }

    return loss_functions.get(loss_type.lower(), F.mse_loss)


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible"""

    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class SACAgent:
    def __init__(
        self, observation_space, action_space, loss_fn=F.mse_loss, **userconfig
    ):
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Observation space {} incompatible ".format(observation_space)
            )
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace("Action space {} incompatible ".format(action_space))

        self._observation_space = observation_space
        self._action_space = action_space
        self._obs_dim = observation_space.shape[0]
        self._action_dim = action_space.shape[0]
        self.critic_loss_fn = loss_fn

        # Default configuration
        self._config = {
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "learning_rate_actor": 3e-4,
            "learning_rate_critic": 3e-4,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "tau": 0.005,  # Polyak update
            "alpha": 0.2,  # Temperature parameter
            "learn_alpha": True,  # Whether to learn temperature parameter
            "update_every": 1,
            "epsilon": 1e-6,
            # PER parameters
            "use_per": False,
            "per_alpha": 0.6,  # beta_1 in PER paper
            "per_beta": 0.4,  # beta_2 in PER paper
            "per_beta_increment": 0.001,
            # ERE parameters
            "use_ere": False,
            "ere_eta0": 0.996,  # Initial ERE decay rate
            "ere_etaT": 1.0,  # Final ERE decay rate
            "ere_c_k_min": 2500,  # Minimum ERE buffer size
        }
        self._config.update(userconfig)

        # Initialize networks
        self.actor = Actor(
            self._obs_dim,
            self._action_dim,
            self._config["hidden_sizes_actor"],
            self._config["learning_rate_actor"],
            action_space=action_space,
            device=device,
            epsilon=self._config["epsilon"],
        )

        # Two Q-functions to mitigate positive bias in policy improvement
        self.critic1 = Critic(
            self._obs_dim,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )
        self.critic2 = Critic(
            self._obs_dim,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )

        # Target networks
        self.critic1_target = Critic(
            self._obs_dim,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )
        self.critic2_target = Critic(
            self._obs_dim,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )

        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Initialize alpha (entropy temperature)
        if self._config["learn_alpha"]:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=self._config["learning_rate_actor"]
            )
        else:
            self.alpha = torch.tensor(self._config["alpha"])

        # Initialize replay buffer (PER or standard)
        if self._config["use_per"]:
            self.buffer = PrioritizedExperienceReplay(
                max_size=self._config["buffer_size"],
                beta_1=self._config["per_alpha"],
                beta_2=self._config["per_beta"],
                beta_increment=self._config["per_beta_increment"],
                epsilon=self._config["epsilon"],
                use_ere=self._config["use_ere"],
                eta_0=self._config["ere_eta0"],
                eta_T=self._config["ere_etaT"],
                c_k_min=self._config["ere_c_k_min"],
            )
        else:
            self.buffer = ReplayMemory(max_size=self._config["buffer_size"])

        self.train_iter = 0
        self.K = 0  # Track total steps for ERE

        # Move networks to device
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)

    def reset_noise(self):
        """Reset the noise process if it has a reset method"""
        if hasattr(self.actor.noise, "reset"):
            self.actor.noise.reset()

    def act(self, observation, eval_mode=False):
        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)

        with torch.no_grad():
            if eval_mode:
                action_mean, _ = self.actor(observation)
                action = (
                    torch.tanh(action_mean) * self.actor.action_scale
                    + self.actor.action_bias
                )
            else:
                action, _ = self.actor.sample(observation)

        return action.cpu().numpy()[0]

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self, iter_fit=32):
        losses = []

        for k in range(iter_fit):
            # Sample from replay buffer with PER/ERE if enabled
            if isinstance(self.buffer, PrioritizedExperienceReplay):
                experiences, indices, weights = self.buffer.sample(
                    batch=self._config["batch_size"], step=k, total_steps=self.K
                )
                state = torch.FloatTensor(np.stack(experiences[0])).to(device)
                action = torch.FloatTensor(np.stack(experiences[1])).to(device)
                reward = torch.FloatTensor(np.stack(experiences[2])[:, None]).to(device)
                next_state = torch.FloatTensor(np.stack(experiences[3])).to(device)
                done = torch.FloatTensor(np.stack(experiences[4])[:, None]).to(device)

                weights = torch.FloatTensor(weights).to(device)
            else:
                batch = self.buffer.sample(batch=self._config["batch_size"])
                state = torch.FloatTensor(np.stack(batch[:, 0])).to(device)
                action = torch.FloatTensor(np.stack(batch[:, 1])).to(device)
                reward = torch.FloatTensor(np.stack(batch[:, 2])[:, None]).to(device)
                next_state = torch.FloatTensor(np.stack(batch[:, 3])).to(device)
                done = torch.FloatTensor(np.stack(batch[:, 4])[:, None]).to(device)
                weights = torch.ones_like(reward).to(device)

            with torch.no_grad():
                # Sample next action and compute Q-target
                next_action, next_log_prob = self.actor.sample(next_state)

                q1_target = self.critic1_target(next_state, next_action)
                q2_target = self.critic2_target(next_state, next_action)
                q_target = torch.min(q1_target, q2_target)

                # Compute target with entropy
                value_target = q_target - self.alpha * next_log_prob
                q_backup = reward + (1 - done) * self._config["discount"] * value_target

            # Update critics
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)

            # Compute TD errors for priority updates
            td_error1 = q_backup.detach() - q1
            td_error2 = q_backup.detach() - q2

            # Compute critic losses with importance sampling weights
            critic1_loss = (td_error1.pow(2) * weights).mean()
            critic2_loss = (td_error2.pow(2) * weights).mean()

            self.critic1.optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1.optimizer.step()

            self.critic2.optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2.optimizer.step()

            # Update PER priorities if using PER
            if isinstance(self.buffer, PrioritizedExperienceReplay):
                new_priorities = abs(
                    ((td_error1 + td_error2) / 2.0 + self._config["epsilon"])
                    .detach()
                    .cpu()
                    .numpy()
                )
                self.buffer.update_priorities(indices, new_priorities)

            # Update actor (using importance sampling weights)
            action_new, log_prob = self.actor.sample(state)
            q1_new = self.critic1(state, action_new)
            q2_new = self.critic2(state, action_new)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (self.alpha * log_prob - q_new).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update alpha if necessary (using importance sampling weights)
            if self._config["learn_alpha"]:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            # Update target networks
            if self.train_iter % self._config["update_every"] == 0:
                for param, target_param in zip(
                    self.critic1.parameters(), self.critic1_target.parameters()
                ):
                    target_param.data.copy_(
                        self._config["tau"] * param.data
                        + (1 - self._config["tau"]) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic2.parameters(), self.critic2_target.parameters()
                ):
                    target_param.data.copy_(
                        self._config["tau"] * param.data
                        + (1 - self._config["tau"]) * target_param.data
                    )

            self.train_iter += 1
            losses.append((critic1_loss.item(), critic2_loss.item(), actor_loss.item()))

        return losses

    def state(self):
        return (
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic2.state_dict(),
        )

    def restore_state(self, state):
        self.actor.load_state_dict(state[0])
        self.critic1.load_state_dict(state[1])
        self.critic2.load_state_dict(state[2])
        self.critic1_target.load_state_dict(state[1])
        self.critic2_target.load_state_dict(state[2])


def main():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-e",
        "--env",
        action="store",
        type="string",
        dest="env_name",
        default="Pendulum-v1",
        help="Environment (default %default)",
    )
    optParser.add_option(
        "-n",
        "--eps",
        action="store",
        type="float",
        dest="epsilon",
        default=1e-6,
        help="Policy noise (default %default)",
    )
    optParser.add_option(
        "-t",
        "--train",
        action="store",
        type="int",
        dest="train",
        default=32,
        help="Number of training batches per episode (default %default)",
    )
    optParser.add_option(
        "-l",
        "--lr",
        action="store",
        type="float",
        dest="lr",
        default=0.0001,
        help="Learning rate (default %default)",
    )
    optParser.add_option(
        "-m",
        "--maxepisodes",
        action="store",
        type="int",
        dest="max_episodes",
        default=2000,
        help="Number of episodes (default %default)",
    )
    optParser.add_option(
        "-f",
        "--loss",
        action="store",
        type="string",
        dest="loss_type",
        default="mse",
        help="Loss function type (mse/huber/mae/mse_weighted, default %default)",
    )
    optParser.add_option(
        "-u",
        "--update",
        action="store",
        type="float",
        dest="update_every",
        default=1,
        help="Target network update frequency (default %default)",
    )
    optParser.add_option(
        "-s",
        "--seed",
        action="store",
        type="int",
        dest="seed",
        default=None,
        help="Random seed (default %default)",
    )

    optParser.add_option(
        "--use_per",
        action="store_true",
        dest="use_per",
        default=False,
        help="Use Prioritized Experience Replay",
    )
    optParser.add_option(
        "--use_ere",
        action="store_true",
        dest="use_ere",
        default=False,
        help="Use Emphasizing Recent Experience",
    )
    optParser.add_option(
        "--per_alpha",
        type="float",
        dest="per_alpha",
        default=0.6,
        help="PER alpha parameter (default: 0.6)",
    )
    optParser.add_option(
        "--per_beta",
        type="float",
        dest="per_beta",
        default=0.4,
        help="PER beta parameter (default: 0.4)",
    )
    optParser.add_option(
        "--ere_eta0",
        type="float",
        dest="ere_eta0",
        default=0.996,
        help="ERE initial eta parameter (default: 0.996)",
    )
    optParser.add_option(
        "--ere_min_size",
        type="int",
        dest="ere_min_size",
        default=2500,
        help="ERE minimum buffer size (default: 2500)",
    )
    opts, args = optParser.parse_args()

    env_name = opts.env_name
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous=True)
    else:
        env = gym.make(env_name)

    max_episodes = opts.max_episodes
    max_timesteps = 2000
    train_iter = opts.train
    log_interval = 20
    random_seed = opts.seed

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    critic_loss_fn = select_loss_function(opts.loss_type)

    sac = SACAgent(
        env.observation_space,
        env.action_space,
        critic_loss_fn,
        learning_rate_actor=opts.lr,
        learning_rate_critic=opts.lr,
        update_every=opts.update_every,
        use_per=opts.use_per,
        use_ere=opts.use_ere,
        per_alpha=opts.per_alpha,
        per_beta=opts.per_beta,
        ere_eta0=opts.ere_eta0,
        ere_c_k_min=opts.ere_min_size,
    )

    # Logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(
            f"./results/SAC_{env_name}-update{opts.update_every}-t{train_iter}-l{opts.lr}"
            f"-per{opts.use_per}-ere{opts.use_ere}-s{random_seed}-stat.pkl",
            "wb",
        ) as f:
            pickle.dump(
                {
                    "rewards": rewards,
                    "lengths": lengths,
                    "train": train_iter,
                    "lr": opts.lr,
                    "update_every": opts.update_every,
                    "losses": losses,
                    "per_enabled": opts.use_per,
                    "ere_enabled": opts.use_ere,
                    "buffer_stats": (
                        sac.buffer.get_statistics()
                        if hasattr(sac.buffer, "get_statistics")
                        else None
                    ),
                },
                f,
            )

    # Training loop
    for i_episode in range(1, max_episodes + 1):
        sac.K = 0
        sac.reset_noise()
        ob, _info = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            timestep += 1
            done = False

            a = sac.act(ob)
            ob_new, reward, done, trunc, _info = env.step(a)

            total_reward += reward
            sac.store_transition((ob, a, reward, ob_new, done))
            sac.K += 1

            ob = ob_new
            if done or trunc:
                break

        # number of updates is the same as episode length K
        # source: https://arxiv.org/pdf/1906.04009
        losses.extend(sac.train(sac.K))
        rewards.append(total_reward)
        lengths.append(t)

        # Save checkpoint
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(
                sac.state(),
                f"./results/SAC_{env_name}_{i_episode}-update{opts.update_every}-t{train_iter}-l{opts.lr}-s{random_seed}.pth",
            )
            save_statistics()

        # Logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print(
                "Episode {} \t avg length: {} \t reward: {}".format(
                    i_episode, avg_length, avg_reward
                )
            )

    save_statistics()


if __name__ == "__main__":
    main()
