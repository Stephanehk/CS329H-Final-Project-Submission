import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import argparse
import numpy as np
import pickle
from rl_utils.env_setups import setup_pandemic_env, setup_pandemic_env_w_gt_rew_set,setup_traffic_env_w_gt_rew_set, setup_glucose_env, setup_traffic_env
from utils.pandemic_config import get_ppo_config as get_pandemic_ppo_config
from utils.glucose_config import get_ppo_config as get_glucose_ppo_config
from utils.traffic_config import get_ppo_config as get_traffic_ppo_config
import os, warnings
from pathlib import Path
from ray.rllib.policy.policy import Policy
import datetime

os.environ["PYTHONWARNINGS"] = "ignore"        # inherited by all Ray workers
warnings.filterwarnings("ignore", category=UserWarning)
# from rl_utils.reward_wrapper import SumReward

def create_env(env_config, wrap_env=True):
    """Create environment based on the specified type."""
    env_type = env_config.get("env_type")
    reward_fun_type = env_config.get("reward_fun_type")

    assert reward_fun_type in ["gt_rew_set", "learned_rew"]
   
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
        else:
            return setup_traffic_env(env_config,wrap_env)
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

    parser.add_argument("--reward-fun-type", type=str, default="gt_rew_set", 
                      choices=["gt_rew_set", "learned_rew"],
                      help="Type of reward function to use")
    parser.add_argument("--gt-rew-i", type=int, default=0,
                      help="Index of g.t. reward function in set of g.t. reward functions, only used if (and must be set if) reward-fun-type==gt_rew_set")

    parser.add_argument("--flip-sign", action='store_true', help="Whether to flip the sign of the g.t. reward function, only used if reward-fun-type==gt_rew_set")
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
        env_config = get_env_config()
        env_config["env_type"] = "pandemic"  # Add env_type to config
        env_config["reward_fun_type"] = args.reward_fun_type
        env_config["gt_rew_i"] = args.gt_rew_i
        env_config["flip_sign"] = args.flip_sign
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

        ppo_config = get_traffic_ppo_config( "traffic_env", env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("traffic_env", create_env)
        ppo_config = ppo_config.environment( "traffic_env", env_config=env_config)
        base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/traffic_base_policy/checkpoint_000025"

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
    # Create the algorithm - explicitly specify PPO for RLlib 2.7
    algo = ppo_config.build()

    if args.init_checkpoint:
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

    for iteration in range(args.num_iterations):
        # Train for one iteration
        
        if iteration % save_freq == 0:
            ckpt_path = save_root / f"checkpoint_{iteration}"
            checkpoint = algo.save(checkpoint_dir=str(ckpt_path))
            print (f"Saved checkpoint to {checkpoint}")

        result = algo.train()
        
        # print (result)
        # print (result.keys())
        # print ("\n")
        # Print training metrics
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print (result)
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
    

