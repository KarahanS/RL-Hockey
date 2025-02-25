## How to use the models

You can use SAC model by importing `SACAgent` from our code and then calling the following function:

```python

def load_sac_agent(config_path, checkpoint_path, env):
    """
    Loads a SACAgent from a checkpoint (same as your original function).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("Loaded SAC config from checkpoint.")
    else:
        raise ValueError("No config found in checkpoint.")

    learn_alpha = config.get("learn_alpha", True)
    if isinstance(learn_alpha, str):
        learn_alpha = (learn_alpha.lower() == "true")

    agent = SACAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        discount=config.get("discount", 0.99),
        buffer_size=config.get("buffer_size", int(1e6)),
        learning_rate_actor=config.get("learning_rate_actor", 3e-4),
        learning_rate_critic=config.get("learning_rate_critic", 3e-4),
        update_every=config.get("update_every", 1),
        use_per=config.get("use_per", False),
        use_ere=config.get("use_ere", False),
        per_alpha=config.get("per_alpha", 0.6),
        per_beta=config.get("per_beta", 0.4),
        ere_eta0=config.get("ere_eta0", 0.996),
        ere_c_k_min=config.get("ere_c_k_min", 2500),
        noise=config.get(
            "noise",
            {
                "type": "normal",
                "sigma": 0.1,
                "theta": 0.15,
                "dt": 1e-2,
                "beta": 1.0,
                "seq_len": 1000,
            },
        ),
        batch_size=config.get("batch_size", 256),
        hidden_sizes_actor=config["hidden_sizes_actor"],
        hidden_sizes_critic=config["hidden_sizes_critic"],
        tau=config.get("tau", 0.005),
        learn_alpha=learn_alpha,
        alpha=config.get("alpha", 0.2),
        control_half=True,
    )
    agent.restore_full_state(checkpoint)
    return agent
```

Necessary config parameters are already saved in the checkpoint file. After loading the agent, you can use it to interact with the environment calling `.act()` or further train it using `.train()`. 

