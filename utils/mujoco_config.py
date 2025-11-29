"""
Configuration for training Mujoco Ant environment with PPO.
"""
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.typing import AlgorithmConfigDict
from utils.nan_debug_callback import NaNDebugCallback  # Uncomment to enable NaN debugging

def get_ppo_config(env_config, num_gpus, seed, num_rollout_workers=1, num_envs_per_worker=1):
    """
    PPO settings matched to common Ant baselines (SB3 RL-Zoo / CleanRL style):
      - n_steps = 2048 (rollout_fragment_length)
      - batch_size = n_steps * num_envs_total
      - n_epochs = 10
      - minibatch_size = 64
      - net: [64,64] with tanh
      - lr = 3e-4, gamma = 0.99, gae_lambda = 0.95
      - clip_param = 0.2, ent_coef = 0.0, vf_coef = 0.5
      - max_grad_norm = 0.5
      - KL penalty disabled (SB3-style; RLlib's kl_coeff=0.0)
      - Observation normalization enabled (MeanStdFilter) to mimic VecNormalize(obs)
    """
    # --- Core PPO hyperparams to match Ant baselines ---
    n_steps = 2048                      # per-env rollout length
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_param = 0.2
    entropy_coeff = 0.0
    vf_loss_coeff = 0.5
    max_grad_norm = 0.5                 # RLlib: grad_clip (standard default for PPO Ant-v4)
    num_sgd_iter = 10                   # "epochs"
    sgd_minibatch_size = 64

    # Total batch = per-env steps * total envs (workers * envs/worker).
    total_envs = max(1, num_rollout_workers) * max(1, num_envs_per_worker)
    train_batch_size = n_steps * total_envs

    # Model architecture: SB3 default MLP is [64, 64], tanh
    model_config = {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "tanh",
        "vf_share_layers": False,
        "free_log_std": False,
    }

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .environment(env="mujoco_env", env_config=env_config, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .training(
            lr=lr,
            gamma=gamma,
            lambda_=gae_lambda,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=vf_loss_coeff,
            # SB3 does NOT use an explicit KL penalty; disable RLlib's by setting coeff=0.0.
            kl_coeff=0.0,
            # Disable VF clipping (SB3 default) - observation normalization handles stability
            vf_clip_param=float('inf'),
            grad_clip=max_grad_norm,
            grad_clip_by="global_norm",  # Use global norm clipping for better stability
            num_sgd_iter=num_sgd_iter,
            sgd_minibatch_size=sgd_minibatch_size,
            train_batch_size=train_batch_size,
            # Use GAE (defaults to True in PPO, but set explicitly for clarity)
            use_gae=True,
        )
        .rollouts(
            num_rollout_workers=num_rollout_workers,
            num_envs_per_worker=num_envs_per_worker,
            rollout_fragment_length=n_steps,
            batch_mode="truncate_episodes",
        )
        .debugging(seed=seed, log_level="INFO")
        .rl_module(_enable_rl_module_api=False)
    )

    # Additional PPO/Model bits that mimic SB3/Ant habits
    updates: AlgorithmConfigDict = {
        "model": model_config,
        "normalize_actions": True,                 # SB3 uses tanh squashing; keep normalized actions
        "observation_filter": "MeanStdFilter",     # mimic VecNormalize(obs) during training - CRITICAL for stability
        "seed": seed,
        # Keep legacy APIs on for RLlib 2.7 style
        "_enable_rl_module_api": False,
        "_enable_learner_api": False,
        "enable_connectors": False,
        # Uncomment below to enable NaN debugging callback
        # "callbacks": NaNDebugCallback,
    }
    config.update_from_dict(updates)
    return config

def get_config():
    """
    Get environment configuration for Mujoco Ant.
    
    Returns:
        Dictionary with environment configuration
    """
    config = {
        "env_name": "Ant-v4",  # Gymnasium environment name
        "render_mode": None,  # Set to "human" for visualization
        "max_episode_steps": 1000,  # Maximum steps per episode
        
        # Ant-specific parameters (matching Gymnasium defaults)
        "healthy_reward": 1.0,
        "terminate_when_unhealthy": True,
        "healthy_z_range": [0.2, 1.0],
        "reset_noise_scale": 0.1,
        "exclude_current_positions_from_observation": True,
        
        # Frame skip (already built into Ant-v4)
        "frame_skip": 5,
    }
    
    return config

