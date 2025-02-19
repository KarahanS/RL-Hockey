import argparse
import copy
import os
import sys
from importlib import reload

import numpy as np
from matplotlib import pyplot as plt

# Adding the parent directory to the path to enable importing
root_dir = os.path.dirname(os.path.abspath("../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.dqn_action_space import CustomActionSpace
from DDQN.DQN import DQNAgent, TargetDQNAgent, DoubleDQNAgent
from DDQN.DDQN import DuelingDQNAgent, DoubleDuelingDQNAgent
from DDQN.dqn_trainer import Stats, Round, CustomHockeyMode, RandomWeaknessBasicOpponent, \
    train_ddqn_agent_torch, SACOpponent
    
import hockey.hockey_env as h_env


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def parse_rounds_arg(rounds_str: str) -> tuple[int, str, str]:
    """Parse the rounds argument string "len,oppname,modename" into a tuple of
    (round_len, opp_name, mode_name)
    """

    try:
        round_len, opp_name, mode_name = rounds_str.split(",")
        round_len = int(round_len)

        return round_len, opp_name, mode_name
    except ValueError:
        raise ValueError(f"Invalid rounds argument: {rounds_str}")
    except KeyError:
        raise ValueError(f"Invalid mode: {mode_name}")
    except Exception as e:
        raise ValueError(f"Error parsing rounds argument: {e}")


def parse_rounds_tuple(rounds_args: list | tuple, opps_dict: dict) -> list[Round]:
    """Parse the rounds argument tuple into a list of Round objects"""

    rounds = []
    has_cotraining = False

    for r in rounds_args:
        len, opp_name, mode_name = r

        if opp_name not in opps_dict:
            raise ValueError(f"Invalid opponent name: {opp_name}. Available opponents: {opps_dict.keys()}")
        
        if opp_name == "self_scratch":
            train_opp = True
            has_cotraining = True
        else:
            train_opp = False
        
        match mode_name:
            case "normal":
                mode = CustomHockeyMode.NORMAL
            case "sht":
                mode = CustomHockeyMode.SHOOTING
            case "def":
                mode = CustomHockeyMode.DEFENSE
            case "shtdef":
                mode = CustomHockeyMode.RANDOM_SHOOTING_DEFENSE
            case "rand":
                mode = CustomHockeyMode.RANDOM_ALL
            case _:
                raise ValueError(f"Invalid mode: {mode_name}")
        
        rounds.append(Round(len, opps_dict[opp_name], mode, train_opp=train_opp))
    
    return rounds, has_cotraining


def train(hparams, run_name, agent_type, action_space, model_dir="./models/", model_init_ckpt=None,
          skip_plot=False, plot_dir="./plots/", rounds_config=[(20_000, "strong", "normal")],
          eval_freq=500):
    # Load the environment
    env = h_env.HockeyEnv()

    # Define the agent
    match agent_type:
        case "dqn":
            agent_class = DQNAgent
        case "targ-dqn":
            agent_class = TargetDQNAgent
        case "doub-dqn":
            agent_class = DoubleDQNAgent
        case "duel-dqn":
            agent_class = DuelingDQNAgent
        case "doub-duel-dqn":
            agent_class = DoubleDuelingDQNAgent
        case _:
            raise ValueError(f"Invalid agent type: {agent_type}")
    
    # Define action space
    match action_space:
        case "default":
            action_space = env.discrete_action_space
        case "custom":
            action_space = CustomActionSpace()

    # TODO: Can we just explode the hparams dict here?
    agent_player = agent_class(
        env.observation_space,
        action_space,
        per=hparams["per"],
        hidden_sizes=hparams["hidden_sizes"],
        hidden_sizes_A=hparams["hidden_sizes_A"],
        hidden_sizes_V=hparams["hidden_sizes_V"],
        learning_rate=hparams["learning_rate"],
        discount=hparams["discount"],
        epsilon=hparams["epsilon"],
        epsilon_decay_rate=hparams["epsilon_decay_rate"],
        epsilon_min=hparams["epsilon_min"],
        update_target_freq=hparams["update_target_freq"],
        tau=hparams["tau"],
        use_torch=hparams["use_torch"]
    )

    # Load the model weights if continuing from a previous run
    if model_init_ckpt is not None:
        agent_player.load_state(model_init_ckpt)

    # Define the opponent(s)
    agent_opp_weak = h_env.BasicOpponent(weak=True)
    agent_opp_strong = h_env.BasicOpponent(weak=False)
    agent_opp_random = RandomWeaknessBasicOpponent(weakness_prob=hparams["weakness_prob"])
    agent_opp_self_scratch = copy.deepcopy(agent_player)  # Trained alongside the player
    agent_opp_self_copy = copy.deepcopy(agent_player)  # Copy of the player for evaluation - will be updated during training
    agent_opp_sac = SACOpponent(env=env)

    train_opps_dict = {  # Opponents to train against
        "weak": agent_opp_weak,
        "strong": agent_opp_strong,
        "randweak": agent_opp_random,
        "self_scratch": agent_opp_self_scratch,
        "sac": agent_opp_sac
    }

    # Define the rounds
    rounds, co_trained = parse_rounds_tuple(rounds_config, train_opps_dict)
    
    # Define the opponents to evaluate against
    eval_opps_dict = {  # Opponents to evaluate against
        "weak": agent_opp_weak,
        "strong": agent_opp_strong,
        #"randweak_p" + f"{agent_opp_random.weakness_prob}": agent_opp_random,
        "self_copy": agent_opp_self_copy,
        "sac": agent_opp_sac
    }
    if co_trained:
        # Evaluate against the co-trained agent as well
        eval_opps_dict["self_scratch"] = agent_opp_self_scratch

    # For visualization
    stats = Stats()
    wandb_hparams = hparams.copy()
    wandb_hparams["agent_type"] = agent_type
    wandb_hparams["run_name"] = run_name


    # Train the agent

    run_id = train_ddqn_agent_torch(
        agent_player,
        env,
        action_space,
        model_dir=model_dir,
        max_steps=hparams["max_steps"],
        rounds=rounds,
        stats=stats,
        eval_opps_dict=eval_opps_dict,
        ddqn_iter_fit=hparams["ddqn_iter_fit"],
        eval_freq=eval_freq,
        eval_num_matches=hparams["eval_num_matches"],
        tqdm=None,
        verbose=hparams["verbose"],
        wandb_hparams=wandb_hparams
    )

    # Save the agent model weights
    hs_a = ("_a" + str(hparams["hidden_sizes_A"])) if "duel" in agent_type else ""
    hs_v = ("_v" + str(hparams["hidden_sizes_V"])) if "duel" in agent_type else ""
    agent_player.save_state(os.path.join(model_dir, f"{run_id}_{run_name}_last_q{hparams["hidden_sizes"]}" + hs_a + hs_v + ".ckpt"))

    # Plot the statistics & save
    if not skip_plot:
        plot_stats(stats, dir=plot_dir)
    
    # Finalize
    env.close()


def plot_stats(stats: Stats, dir="./plots/"):
    if not os.path.exists(dir):
            os.makedirs(dir)

    returns_np = np.asarray(stats.returns)
    losses_np = np.asarray(stats.losses)

    plt.figure()
    plt.plot(returns_np[:, 1], label="Return")
    plt.plot(running_mean(returns_np[:, 1], 20), label="Smoothed return")
    for xc in stats.returns_training_stages:
        plt.axvline(x=xc, color='r', linestyle='--')
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("Return over episodes")
    plt.plot([0], color='r', linestyle='--', lw=2, label='Training stage')  # Dummy plot for legend
    plt.legend()
    plt.savefig(os.path.join(dir, "returns.png"))
    
    plt.figure()
    plt.plot(losses_np, label="Loss")
    plt.plot(running_mean(losses_np, 50), label="Smoothed Loss")
    for xc in stats.losses_training_stages:
        plt.axvline(x=xc, color='r', linestyle='--')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss over iterations")
    plt.plot([0], color='r', linestyle='--', lw=2, label='Training stage')  # Dummy plot for legend
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(dir, "losses.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDQN agent to play hockey")

    parser.add_argument("run_name", type=str, help="Name of the wandb run to log the training process")
    parser.add_argument("agent_type", type=str, help="Type of the agent to train",
                        choices=["dqn", "targ-dqn", "doub-dqn", "duel-dqn", "doub-duel-dqn"])
    parser.add_argument("--action-space", type=str, default="default", help="Type of the action space to use",
                        choices=["default", "custom"])

    # Agent hparam.s
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Path to the model weights to continue training from")
    parser.add_argument("--model-dir", type=str, default="./models/",
                        help="Directory to save the trained model weights")
    parser.add_argument("--plot-dir", type=str, default="./plots/",
                        help="Directory to save the plots")
    parser.add_argument("--per", action="store_true", help="Use Prioritized Experience Replay")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512],
                        help="Hidden layer sizes for the Q function")
    parser.add_argument("--hidden-sizes-A", type=int, nargs="+", default=[512, 512],
                        help="Hidden layer sizes for the advantage stream in Dueling DQN")
    parser.add_argument("--hidden-sizes-V", type=int, nargs="+", default=[512, 512],
                        help="Hidden layer sizes for the value stream in Dueling DQN")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the agent")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor for the agent")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Exploration rate for the agent")
    parser.add_argument("--epsilon-decay_rate", type=float, default=0.999, help="Decay rate of epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.2, help="Minimum value of epsilon")
    parser.add_argument("--update-target-freq", type=int, default=1000,
                        help="Frequency of updating the target network")
    parser.add_argument("--tau", type=float, default=1e-4, help="Soft update parameter for the target network")
    parser.add_argument("--use-numpy", action="store_true", help="Use NumPy functionalities for training")

    # Opponent hparam.s
    parser.add_argument("--weakness-prob", type=float, default=0.2, help="Probability of the opponent being weak")

    # Training hparam.s
    parser.add_argument("--ddqn-iter-fit", type=int, default=32, help="Number of iterations to train the DDQN agent"
                        " for each episode")
    parser.add_argument("--rounds", type=parse_rounds_arg, nargs="+", default=[(20_000, "strong", "normal")],
                        help="Rounds of training with different opponents and modes")
    parser.add_argument("--print-freq", type=int, default=25, help="Frequency of printing the training statistics")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting the training statistics")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Frequency of evaluating the agent")
    parser.add_argument("--eval-num-matches", type=int, default=1000,
                        help="Number of matches to play for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Verbosity of the training process")

    args = parser.parse_args()

    hparams = {
        # Agent hparam.s
        "per": args.per,
        "action_space": args.action_space,
        "hidden_sizes": args.hidden_sizes,
        "hidden_sizes_A": args.hidden_sizes_A,
        "hidden_sizes_V": args.hidden_sizes_V,
        "learning_rate": args.lr,
        "discount": args.discount,
        "epsilon": args.epsilon,
        "epsilon_decay_rate": args.epsilon_decay_rate,
        "epsilon_min": args.epsilon_min,
        "update_target_freq": args.update_target_freq,
        "tau": args.tau,
        "use_torch": not args.use_numpy,
        # Opponent hparam.s
        "weakness_prob": 0.5,
        # Training hparam.s
        "rounds_config": args.rounds,
        "max_steps": 100000,  # Overridden by environment
        "ddqn_iter_fit": args.ddqn_iter_fit,
        "print_freq": args.print_freq,
        "eval_num_matches": args.eval_num_matches,
        "verbose": args.verbose
    }

    # TODO: Support hparam search with appropriate run names
    train(hparams, args.run_name, args.agent_type, args.action_space, model_dir=args.model_dir,
          model_init_ckpt=args.continue_from, skip_plot=args.skip_plot, plot_dir=args.plot_dir,
          rounds_config=args.rounds, eval_freq=args.eval_freq)
