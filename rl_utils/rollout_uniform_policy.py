import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import argparse
import numpy as np
import pickle
from rl_utils.env_setups import setup_pandemic_env, setup_glucose_env, setup_traffic_env
from utils.pandemic_config import get_ppo_config as get_pandemic_ppo_config
from utils.glucose_config import get_ppo_config as get_glucose_ppo_config
from utils.traffic_config import get_ppo_config as get_traffic_ppo_config
from rl_utils.random_action_policy_wrapper import RandomActionPolicy
import os, warnings
from pathlib import Path
from ray.rllib.policy.policy import Policy


os.environ["PYTHONWARNINGS"] = "ignore"        # inherited by all Ray workers
warnings.filterwarnings("ignore", category=UserWarning)
# from rl_utils.reward_wrapper import SumReward

def create_env(env_config, wrap_env=True):
    """Create environment based on the specified type."""
    env_type = env_config.get("env_type")
   
    if env_type == "pandemic":
        return setup_pandemic_env(env_config,wrap_env)
    if env_type == "glucose":
        return setup_glucose_env(env_config,wrap_env)
    if env_type == "traffic":
        return setup_traffic_env(env_config,wrap_env)
    raise ValueError(f"Unknown environment type: {env_type}")

def rollout_policy(algo, env, num_episodes=50):
    """Rollout the trained policy in the environment."""
    total_rewards = []
    total_true_rewards = []
    total_proxy_rewards = []
    action_space = algo.workers.local_worker().policy_map["default_policy"]
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        epsiode_true_return = 0
        episode_proxy_return = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action from policy
            # action = algo.compute_single_action(obs, explore=False)
            action = env.action_space.sample()
            obs, reward, done, truncated,info = env.step(action)
            episode_reward += reward
            if "true_reward" in info:
                epsiode_true_return += info["true_reward"]
                episode_proxy_return += info["proxy_reward"]
            else:
                epsiode_true_return += info["true_rew"]
                episode_proxy_return += info["proxy_rew"]
            # assert False
        total_rewards.append(episode_reward)
        total_true_rewards.append(epsiode_true_return)
        total_proxy_rewards.append(episode_proxy_return)
        print(f"Episode {episode + 1} return: {episode_reward}")
        print(f"Episode {episode + 1} true return: {epsiode_true_return}")
        print(f"Episode {episode + 1} proxy return: {episode_proxy_return}")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nMean return over {num_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"\nMean true return over {num_episodes} episodes: {np.mean(total_true_rewards):.2f} ± {np.std(total_true_rewards):.2f}")
    print(f"\nMean proxy return over {num_episodes} episodes: {np.mean(total_proxy_rewards):.2f} ± {np.std(total_proxy_rewards):.2f}")

    return mean_reward, std_reward, total_true_rewards

def evaluate_policy_during_training(algo, env_config, iteration):
    """Evaluate the current policy during training."""
    print(f"\n=== Evaluating policy at iteration {iteration} ===")
    
    # Create environment for evaluation
    eval_env = create_env(env_config, wrap_env=True)
    
    # Rollout the policy
    mean_reward, std_reward, all_rets = rollout_policy(algo, eval_env)
    
    # Cleanup
    eval_env.close()

    return mean_reward, std_reward, all_rets

def wrap_default_policy(worker, random_prob=1.0, pid="default_policy"):
    """Replace `pid` in *this* RolloutWorker with a RandomActionPolicy wrapper."""
    base = worker.policy_map[pid]                     # the pretrained policy obj
    worker.policy_map[pid] = RandomActionPolicy(base, random_prob=random_prob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", type=str, default="pandemic", 
                      choices=["pandemic", "glucose", "traffic"],
                      help="Type of environment to train on")
    parser.add_argument("--num-workers", type=int, default=2,
                      help="Number of workers for parallel training")
    parser.add_argument("--num-gpus", type=int, default=0,
                      help="Number of GPUs to use")
    parser.add_argument("--num-iterations", type=int, default=100,
                      help="Number of training iterations")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for training")
    # parser.add_argument(
    #     "--init-checkpoint",
    #     type=store_true,
    #     default=None,
    #     help="Path to an RLlib checkpoint whose policy weights will "
    #         "be used for initialization (training will *not* resume)."
    # )
    parser.add_argument('--init-checkpoint', action='store_true', help='Use RLlib checkpoint for policy initialization.')

    args = parser.parse_args()

    # Initialize Ray
    ray.init()

    # Register the environment
    
    print ("registering env")
    # Get environment-specific config
    if args.env_type == "pandemic":
        from utils.pandemic_config import get_config as get_env_config
        env_config = get_env_config()
        env_config["env_type"] = "pandemic"  # Add env_type to config
        
        # env_config["reward_function2optimize"] = learned_reward

        ppo_config = get_pandemic_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("pandemic_env", create_env)
        
        # Add environment config to PPO config
        ppo_config = ppo_config.environment("pandemic_env", env_config=env_config)
        base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/pandemic_base_policy/checkpoint_000100/"

    elif args.env_type == "traffic":
        from utils.traffic_config import get_config as get_env_config
        from flow.utils.registry import make_create_env

        env_config = get_env_config()
        # _, env_name = make_create_env(
        #     params=env_config["flow_params_default"],
        #     reward_specification=env_config["reward_specification"],
        #     reward_fun=env_config["reward_fun"],
        #     reward_scale=env_config["reward_scale"],
        # )
        env_config["env_type"] = "traffic"  # Add env_type to config
        ppo_config = get_traffic_ppo_config( "traffic_env", env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("traffic_env", create_env)
        ppo_config = ppo_config.environment( "traffic_env", env_config=env_config)
        base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/traffic_base_policy/checkpoint_000025"

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
        base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/glucose_base_policy/checkpoint_000300"
    # Create the algorithm - explicitly specify PPO for RLlib 2.7
    ppo_config = (
    ppo_config
    .evaluation(
        evaluation_num_workers=args.num_workers,          # new rollout-worker(s) just for eval
        evaluation_duration=args.num_workers,            # 10 episodes  (use "timesteps" if you prefer)
        evaluation_duration_unit="episodes",
        evaluation_config={"explore": False}   # exploit-only by default
        )
    )

    algo = ppo_config.build()

    if args.init_checkpoint:
        # Path to the default policy inside the checkpoint.
        pol_ckpt = (
            Path(base_checkpoint)
            / "policies"
            / "default_policy"          # change if your ID is different
        )

        pretrained_policy = Policy.from_checkpoint(pol_ckpt)  # env-free load  :contentReference[oaicite:0]{index=0}
        algo.get_policy().set_state(pretrained_policy.get_state())
        algo.workers.sync_weights()      # push to remote workers

        #Wrap the policy *everywhere* (local + remote workers)
        wrap_default_policy(algo.workers.local_worker(), random_prob=1.0)
        # … and all remote workers
        algo.workers.foreach_worker(
            lambda w: wrap_default_policy(w, random_prob=1.0)
        )
        print("✔ warm-started policy from", pol_ckpt)
    
    # Training loop with periodic evaluation
    # evaluation_results = algo.evaluate()                   # <- no training step happens
    # # mean_reward = result["evaluation"]["episode_reward_mean"]
    # # print(f"Average return over 10 eval episodes: {mean_reward:.2f}")
    # print (evaluation_results)
        
    # os.makedirs(args.env_type + "_running_results", exist_ok=True)

    # Final evaluation
    print("\n=== Final Policy Evaluation ===")
    final_mean_reward, final_std_reward, all_rets = evaluate_policy_during_training(algo, env_config, args.num_iterations)

    np.save(f"uniform_policy_returns/{args.env_type}_returns.npy", all_rets)
    # Print evaluation summary
    # print("\n=== Evaluation Summary ===")
    # for result in evaluation_results:
    #     print(f"Iteration {result['iteration']}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    # print(f"Final: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    
    # Cleanup
    algo.stop()

if __name__ == "__main__":
    main()
    

