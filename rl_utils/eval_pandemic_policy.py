import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import argparse
import numpy as np
import pickle
from rl_utils.env_setups import setup_pandemic_env, setup_pandemic_env_w_gt_rew_set, setup_glucose_env, setup_traffic_env
from utils.pandemic_config import get_ppo_config as get_pandemic_ppo_config
from utils.glucose_config import get_ppo_config as get_glucose_ppo_config
from utils.traffic_config import get_ppo_config as get_traffic_ppo_config
import os, warnings
from pathlib import Path
from ray.rllib.policy.policy import Policy

from pandemic.python.pandemic_simulator.environment.interfaces import (
    PandemicObservation,
    sorted_infection_summary,
    InfectionSummary,
)

os.environ["PYTHONWARNINGS"] = "ignore"        # inherited by all Ray workers
warnings.filterwarnings("ignore", category=UserWarning)
# from rl_utils.reward_wrapper import SumReward

def create_env(env_config, wrap_env=True):
    """Create environment based on the specified type."""
    env_type = env_config.get("env_type")
    reward_fun_type = env_config.get("reward_fun_type", "gt_rew_set")

    assert reward_fun_type in ["gt_rew_set", "learned_rew"]
   
    if env_type == "pandemic":
        if reward_fun_type == "gt_rew_set":
            return setup_pandemic_env_w_gt_rew_set(env_config, wrap_env)
        else:
            return setup_pandemic_env(env_config, wrap_env)

    if env_type == "glucose":
        return setup_glucose_env(env_config,wrap_env)
    if env_type == "traffic":
        return setup_traffic_env(env_config,wrap_env)
    raise ValueError(f"Unknown environment type: {env_type}")


def rollout_policy(algo, env, num_episodes=5):
    """Rollout the trained policy in the environment."""
    total_rewards = []

    crit_i = sorted_infection_summary.index(InfectionSummary.CRITICAL)
    dead_i = sorted_infection_summary.index(InfectionSummary.DEAD)
    rec_i = sorted_infection_summary.index(InfectionSummary.RECOVERED)
    inf_i = sorted_infection_summary.index(InfectionSummary.INFECTED)

    all_episode_stats =[]
    for episode in range(num_episodes):
        obs,obs_np, _ = env.reset_keep_obs_obj()
        episode_reward = 0
        done = False
        truncated = False

        episode_stats = {"critical":[], "dead":[], "recovered":[], "infected":[], "stage":[]}

        while not (done or truncated):
            # Get action from policy
            action = algo.compute_single_action(obs_np, explore=False)
            obs,obs_np, reward, done, truncated, _ = env.step_keep_obs_obj(action)
            episode_reward += reward
            
            curr_stage = int(obs.stage[-1][-1].item())
            curr_crit = float(obs.global_infection_summary[-1, -1, crit_i].item())
            curr_dead = float(obs.global_infection_summary[-1, -1, dead_i].item())
            curr_rec = float(obs.global_infection_summary[-1, -1, rec_i].item())
            curr_inf = float(obs.global_infection_summary[-1, -1, inf_i].item())

            episode_stats["critical"].append(curr_crit)
            episode_stats["dead"].append(curr_dead)
            episode_stats["recovered"].append(curr_rec)
            episode_stats["infected"].append(curr_inf)
            episode_stats["stage"].append(curr_stage)

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} return: {episode_reward}")
        all_episode_stats.append(episode_stats)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nMean return over {num_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward, all_episode_stats

def evaluate_policy_during_training(algo, env_config, iteration):
    """Evaluate the current policy during training."""
    print(f"\n=== Evaluating policy at iteration {iteration} ===")
    
    # Create environment for evaluation
    eval_env = create_env(env_config, wrap_env=True)
    
    # Rollout the policy
    mean_reward, std_reward, episode_stats = rollout_policy(algo, eval_env)
    
    # Cleanup
    eval_env.close()
    
    return mean_reward, std_reward, episode_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", type=str, default="pandemic", 
                      choices=["pandemic", "glucose", "traffic"],
                      help="Type of environment to train on")
    parser.add_argument("--num-workers", type=int, default=2,
                      help="Number of workers for parallel training")
    parser.add_argument("--num-gpus", type=int, default=0,
                      help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for training")

    parser.add_argument("--pol-ckpt", type=str, default=None,
                      help="Path to the policy checkpoint to load")

    #logs/data/pandemic/20250829_151547/model0

    args = parser.parse_args()

    pol_ckpt = args.pol_ckpt

    # Initialize Ray
    ray.init()

    # Register the environment
    
    print ("registering env")
    # Get environment-specific config
    if args.env_type == "pandemic":
        from utils.pandemic_config import get_config as get_env_config
        env_config = get_env_config()
        env_config["env_type"] = "pandemic"  # Add env_type to config
        
        ppo_config = get_pandemic_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("pandemic_env", create_env)
        
        # Add environment config to PPO config
        ppo_config = ppo_config.environment("pandemic_env", env_config=env_config)
        # pol_ckpt=

    elif args.env_type == "traffic":
        from utils.traffic_config import get_config as get_env_config
        from flow.utils.registry import make_create_env

        env_config = get_env_config()
       
        env_config["env_type"] = "traffic"  # Add env_type to config
       
        ppo_config = get_traffic_ppo_config( "traffic_env", env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("traffic_env", create_env)
        ppo_config = ppo_config.environment( "traffic_env", env_config=env_config)

    elif args.env_type == "glucose": # glucose
        from utils.glucose_config import get_config as get_env_config
        
        env_config = get_env_config()
        env_config["env_type"] = "glucose"  # Add env_type to config
        env_config["gt_reward_fn"] = "magni_rew"
        # env_config["reward_function2optimize"] = learned_reward

        ppo_config = get_glucose_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("glucose_env", create_env)
        
        # Add environment config to PPO config
        ppo_config = ppo_config.environment("glucose_env", env_config=env_config)

    # Create the algorithm - explicitly specify PPO for RLlib 2.7
    algo = ppo_config.build()

    # Path to the default policy inside the checkpoint.
    # pol_ckpt = (
    #     Path(base_checkpoint)
    #     / "policies"
    #     / "default_policy"          # change if your ID is different
    # )

    pretrained_policy = Policy.from_checkpoint(pol_ckpt)["default_policy"]  # env-free load  :contentReference[oaicite:0]{index=0}
    print(pretrained_policy)
    # algo.get_policy().set_state(pretrained_policy.get_state())
    algo.get_policy().set_weights(pretrained_policy.get_weights())  # ← weights only
    algo.workers.sync_weights()      # push to remote workers
    print("✔ warm-started policy from", pol_ckpt)
    
    
    print("\n=== Final Policy Evaluation ===")
    final_mean_reward, final_std_reward, episode_stats = evaluate_policy_during_training(algo, env_config, 0)

    #pickle dump episode_stats
    # os.makedirs(args.env_type + "_running_results", exist_ok=True)
    # os.makedirs(args.env_type + f"_running_results/", exist_ok=True)
    print (episode_stats)
    with open(f"{pol_ckpt}/episode_stats.pkl", "wb") as f:
        pickle.dump(episode_stats, f)

    # Cleanup
    algo.stop()

if __name__ == "__main__":
    main()
    

