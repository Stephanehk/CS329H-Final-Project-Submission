import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import argparse
import numpy as np
import pickle
from rl_utils.env_setups import setup_pandemic_env,setup_traffic_env_pan_reward_fn, setup_pandemic_env_w_gt_rew_set,setup_traffic_env_w_gt_rew_set, setup_glucose_env, setup_traffic_env, setup_mujoco_env, setup_mujoco_env_w_gt_rew_set, setup_mujoco_env_w_data_generation_mode
from utils.pandemic_config import get_ppo_config as get_pandemic_ppo_config
from utils.glucose_config import get_ppo_config as get_glucose_ppo_config
from utils.traffic_config import get_ppo_config as get_traffic_ppo_config
from utils.mujoco_config import get_ppo_config as get_mujoco_ppo_config
import os, warnings
from pathlib import Path
from ray.rllib.policy.policy import Policy
import datetime
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.algorithms.ppo import PPO

os.environ["PYTHONWARNINGS"] = "ignore"        # inherited by all Ray workers
warnings.filterwarnings("ignore", category=UserWarning)
# from rl_utils.reward_wrapper import SumReward

def create_env(env_config, wrap_env=True):
    """Create environment based on the specified type."""
    env_type = env_config.get("env_type")
    reward_fun_type = env_config.get("reward_fun_type")
   
    if env_type == "pandemic":
        if reward_fun_type == "gt_rew_set":
            return setup_pandemic_env_w_gt_rew_set(env_config, wrap_env)
        else:
            return setup_pandemic_env(env_config, wrap_env)
    if env_type == "glucose":
        return setup_glucose_env(env_config,wrap_env)
    if env_type == "traffic":
        if reward_fun_type == "gt_rew_set":
            return setup_traffic_env_w_gt_rew_set(env_config, wrap_env)
        elif reward_fun_type == "pan_reward_fn":
            return setup_traffic_env_pan_reward_fn(env_config, wrap_env)
        else:
            return setup_traffic_env(env_config,wrap_env)
    if env_type == "mujoco" or env_type == "mujoco_backflip":
        if reward_fun_type == "gt_rew_set":
            return setup_mujoco_env_w_gt_rew_set(env_config, wrap_env)
        elif reward_fun_type == "data_generation_mode":
            return setup_mujoco_env_w_data_generation_mode(env_config, wrap_env)
        else:
            return setup_mujoco_env(env_config, wrap_env)
    raise ValueError(f"Unknown environment type: {env_type}")


def rollout_policy(algo, env, num_episodes=5):
    """Rollout the trained policy in the environment."""
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action from policy
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} return: {episode_reward}")
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nMean return over {num_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward

def evaluate_policy_during_training(algo, env_config, iteration):
    """Evaluate the current policy during training."""
    print(f"\n=== Evaluating policy at iteration {iteration} ===")
    
    # Create environment for evaluation
    eval_env = create_env(env_config, wrap_env=True)
    
    # Rollout the policy
    mean_reward, std_reward = rollout_policy(algo, eval_env)
    
    # Cleanup
    eval_env.close()
    
    return mean_reward, std_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", type=str, default="pandemic", 
                      choices=["pandemic", "glucose", "traffic", "mujoco", "mujoco_backflip"],
                      help="Type of environment to train on")
    parser.add_argument("--num-workers", type=int, default=2,
                      help="Number of workers for parallel training")
    parser.add_argument("--num-gpus", type=int, default=0,
                      help="Number of GPUs to use")
    parser.add_argument("--num-iterations", type=int, default=100,
                      help="Number of training iterations")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for training")
    parser.add_argument('--init-checkpoint', action='store_true', help='Use RLlib checkpoint for policy initialization.')
    parser.add_argument("--reward-fun-type", type=str, default="gt_rew_set", 
                      choices=["gt_rew_set", "learned_rew", "data_generation_mode", "pan_reward_fn"],
                      help="Type of reward function to use")
    parser.add_argument("--gt-rew-i", type=int, default=0,
                      help="Index of g.t. reward function in set of g.t. reward functions, only used if (and must be set if) reward-fun-type==gt_rew_set")
    parser.add_argument("--flip-sign", action='store_true', help="Whether to flip the sign of the g.t. reward function, only used if reward-fun-type==gt_rew_set")
    parser.add_argument("--town-size", type=str, default="tiny")
    parser.add_argument("--use-shaping", action='store_true', help="Whether to use shaping for the reward function, only used if reward-fun-type==gt_rew_set")
    parser.add_argument("--no-convo-base-line", action='store_true', help="Whether to use no conversation baseline for the reward function, only used if reward-fun-type==learned_rew")
    #reward_fun_type
    args = parser.parse_args()

    # Initialize Ray
    ray.init()
    
    # Register the environment
    
    print ("registering env")

    print ("**gt_rew_i:", args.gt_rew_i)
    print ("**reward_fun_type:", args.reward_fun_type)
    print ("**flip_sign:", args.flip_sign)
    # Get environment-specific config
    if args.env_type == "pandemic":
        from utils.pandemic_config import get_config as get_env_config
        env_config = get_env_config(town_size=args.town_size)
        env_config["env_type"] = "pandemic"  # Add env_type to config
        env_config["reward_fun_type"] = args.reward_fun_type
        env_config["gt_rew_i"] = args.gt_rew_i
        env_config["flip_sign"] = args.flip_sign
        env_config["town_size"] = args.town_size
        env_config["no_convo_base_line"] = args.no_convo_base_line
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
        env_config["reward_fun_type"] = args.reward_fun_type
        env_config["gt_rew_i"] = args.gt_rew_i
        env_config["flip_sign"] = args.flip_sign

        shaping_checkpoint = "logs/data/traffic/20251124_153408/checkpoint_490"
        env_config["shaping_checkpoint"] = shaping_checkpoint
        env_config["use_shaping"] = args.use_shaping
        env_config["no_convo_base_line"] = args.no_convo_base_line

        ppo_config = get_traffic_ppo_config( "traffic_env", env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("traffic_env", create_env)
        ppo_config = ppo_config.environment( "traffic_env", env_config=env_config)
        base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/traffic_base_policy/checkpoint_000025"

        #this is the policy trained with a g.t. reward function from Pan et al. (slurm id: 13292952)

    elif args.env_type == "glucose": # glucose
        from utils.glucose_config import get_config as get_env_config
        
        env_config = get_env_config()
        env_config["env_type"] = "glucose"  # Add env_type to config
        env_config["gt_reward_fn"] = "magni_rew"
        env_config["reward_fun_type"] = args.reward_fun_type
        env_config["gt_rew_i"] = args.gt_rew_i
        env_config["flip_sign"] = args.flip_sign
        # env_config["reward_function2optimize"] = learned_reward

        ppo_config = get_glucose_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("glucose_env", create_env)
        
        # Add environment config to PPO config
        ppo_config = ppo_config.environment("glucose_env", env_config=env_config)
        base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/glucose_base_policy/checkpoint_000300"
    
    elif args.env_type == "mujoco" or args.env_type == "mujoco_backflip":
        from utils.mujoco_config import get_config as get_env_config
        
        env_config = get_env_config()
        env_config["env_type"] = args.env_type  # Add env_type to config
        env_config["reward_fun_type"] = args.reward_fun_type
        env_config["gt_rew_i"] = args.gt_rew_i
        env_config["flip_sign"] = args.flip_sign
        env_config["use_shaping"] = args.use_shaping

       
        register_env("mujoco_env", create_env)
        
        # Add environment config to PPO config
        ppo_config = get_mujoco_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
        ppo_config = ppo_config.environment("mujoco_env", env_config=env_config)

        #base_checkpoint="logs/data/mujoco/20251121_105505/checkpoint_2330"  # No base checkpoint for Mujoco initially
        base_checkpoint="logs/data/mujoco/20251124_090408/checkpoint_3960"

        #load config from successful run
        # config_checkpoint_path = "logs/data/mujoco/20251106_165904/checkpoint_9999"
        # state = Algorithm._checkpoint_info_to_algorithm_state(get_checkpoint_info(config_checkpoint_path))
        # # Extract the config and override GPU settings
        # config = state["config"].copy()
        # config["env_config"] = env_config
        # algo = PPO(config=config)
        # # ppo_config["env_config"] = env_config
        # # ppo_config = ppo_config.env_config("mujoco_env", env_config=env_config)
        # algo.restore(config_checkpoint_path)

    
    # Create the algorithm - explicitly specify PPO for RLlib 2.7
    algo = ppo_config.build()


    if args.init_checkpoint and base_checkpoint is not None:
        # Path to the default policy inside the checkpoint.
        pol_ckpt = (
            Path(base_checkpoint)
            / "policies"
            / "default_policy"          # change if your ID is different
        )

        pretrained_policy = Policy.from_checkpoint(pol_ckpt)  # env-free load  :contentReference[oaicite:0]{index=0}
        # algo.get_policy().set_state(pretrained_policy.get_state())
        algo.get_policy().set_weights(pretrained_policy.get_weights())  # ← weights only
        algo.workers.sync_weights()      # push to remote workers
        print("✔ warm-started policy from", pol_ckpt)
    elif args.init_checkpoint and base_checkpoint is None:
        print("⚠ Warning: --init-checkpoint specified but no base checkpoint available for this environment")
    # if args.init_checkpoint:
    #     pol_ckpt  = Path(base_checkpoint) / "policies" / "default_policy"
    #     chk_pol   = Policy.from_checkpoint(pol_ckpt)

    #     state = chk_pol.get_state()
    #     state.pop("policy_config", None)     # ← discard old config
    #     algo.get_policy().set_state(state)
    #     algo.workers.sync_weights()

    #     print("✔ warm-started policy state; kept new PPO config")
        
    # Training loop with periodic evaluation
    evaluation_results = []
    
    os.makedirs(args.env_type + "_running_results", exist_ok=True)
    os.makedirs(args.env_type + f"_running_results/{args.reward_fun_type}_{args.gt_rew_i}", exist_ok=True)
    save_freq=10

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = Path("logs") / "data" / args.env_type / timestamp
    save_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = None
    best_reward = -float('inf')
    best_ckpt_path = None
    for iteration in range(args.num_iterations):
        # Train for one iteration
        

        # result = algo.train()
        try:
            result = algo.train()
        except ValueError:
            print ("ValueError encountered, loading previous checkpoint and retrying training")

            print (f"Loading previous checkpoint from {best_ckpt_path}")
            pol_ckpt = (
                Path(best_ckpt_path)
                / "policies"
                / "default_policy"          # change if your ID is different
            )

            pretrained_policy = Policy.from_checkpoint(pol_ckpt)  # env-free load  :contentReference[oaicite:0]{index=0}
            # algo.get_policy().set_state(pretrained_policy.get_state())
            algo.get_policy().set_weights(pretrained_policy.get_weights())  # ← weights only
            algo.workers.sync_weights()      # push to remote workers
            args.num_iterations += 1
            continue
            # result = algo.train()

        if iteration % save_freq == 0:
            ckpt_path = save_root / f"checkpoint_{iteration}"
            checkpoint = algo.save(checkpoint_dir=str(ckpt_path))
            print (f"Saved checkpoint to {checkpoint}")
            if result["sampler_results"]["episode_reward_mean"] > best_reward:
                best_reward = result["sampler_results"]["episode_reward_mean"]
                best_ckpt_path = ckpt_path
        
        # print (result)
        # print (result.keys())
        # print ("\n")
        # Print training metrics
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print (result)
        # print ("episode reward mean:", result["sampler_results"]["episode_reward_mean"])

        with open(args.env_type + f"_running_results/{args.reward_fun_type}_{args.gt_rew_i}/iter_{iteration}.pkl", 'wb') as file:
            pickle.dump(result, file)

        # if "true_reward_mean" in result['custom_metrics']:
        #     print(f"Mean true return: {result['custom_metrics']['true_reward_mean']:.2f}")
        # if "proxy_reward_mean" in result['custom_metrics']:
        #     print(f"Mean proxy return: {result['custom_metrics']['proxy_reward_mean']:.2f}")
        
        # Evaluate every 10 iterations
        # if (iteration + 1) % 10 == 0:
        # mean_reward, std_reward = evaluate_policy_during_training(algo, env_config, iteration + 1)
        # evaluation_results.append({
        #     'iteration': iteration + 1,
        #     'mean_reward': mean_reward,
        #     'std_reward': std_reward
        # })

    print("Training completed!")
    ckpt_path = save_root / f"checkpoint_{iteration}"
    checkpoint = algo.save(checkpoint_dir=str(ckpt_path))
    print (f"Saved final checkpoint to {checkpoint}")
    
    # Final evaluation
    print("\n=== Final Policy Evaluation ===")
    final_mean_reward, final_std_reward = evaluate_policy_during_training(algo, env_config, args.num_iterations)
    
    # Print evaluation summary
    print("\n=== Evaluation Summary ===")
    for result in evaluation_results:
        print(f"Iteration {result['iteration']}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    print(f"Final: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    
    # Cleanup
    algo.stop()

if __name__ == "__main__":
    main()
    

