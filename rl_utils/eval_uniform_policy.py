
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

class UniformRandomPolicy(Policy):
    def compute_actions(self, obs_batch, **kwargs):
        actions = [self.action_space.sample() for _ in obs_batch]
        return np.array(actions), [], {}

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
    # parser.add_argument("--base-checkpoint", type=str, required=True)
    #reward_fun_type
    args = parser.parse_args()

    
    # Initialize Ray
    ray.init()
    
    # Register the environment
    
    print ("registering env")
    # base_checkpoint = args.base_checkpoint
    gt_rew_fn2mean_return = {}
    for flip_sign in [False, True]:
        for gt_rew_i in range(25):
            for env_config_type in ["easy", "hard"]:

                if env_config_type == "easy":
                    #for pandemic 
                    town_size = "tiny"
                    #for traffic
                    exp_tag = "singleagent_merge_bus"
                else:
                    #for pandemic 
                    town_size = "medium"
                    #for traffic
                    exp_tag = "singleagent_merge_bus_bigger"

                # Get environment-specific config
                if args.env_type == "pandemic":
                    from utils.pandemic_config import get_config as get_env_config
                    env_config = get_env_config(town_size=town_size)
                    env_config["env_type"] = "pandemic"  # Add env_type to config
                    env_config["reward_fun_type"] = "gt_rew_set"
                    env_config["gt_rew_i"] = gt_rew_i
                    env_config["flip_sign"] = flip_sign
                    env_config["town_size"] = town_size

                    ppo_config = get_pandemic_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
                    register_env("pandemic_env", create_env)
                
                    # Add environment config to PPO config
                    ppo_config = ppo_config.environment("pandemic_env", env_config=env_config)
                    base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/pandemic_base_policy/checkpoint_000100/"
                elif args.env_type == "traffic":
                    from utils.traffic_config import get_config as get_env_config
                    from flow.utils.registry import make_create_env

                    env_config = get_env_config(exp_tag=exp_tag)
                    # _, env_name = make_create_env(
                    #     params=env_config["flow_params_default"],
                    #     reward_specification=env_config["reward_specification"],
                    #     reward_fun=env_config["reward_fun"],
                    #     reward_scale=env_config["reward_scale"],
                    # )
                    env_config["env_type"] = "traffic"  # Add env_type to config
                    env_config["reward_fun_type"] = "gt_rew_set"
                    env_config["gt_rew_i"] = gt_rew_i
                    env_config["flip_sign"] = flip_sign

                    ppo_config = get_traffic_ppo_config( "traffic_env", env_config, args.num_gpus, args.seed, args.num_workers)
                    register_env("traffic_env", create_env)
                    ppo_config = ppo_config.environment( "traffic_env", env_config=env_config)
                    base_checkpoint="/next/u/stephhk/orpo/data/base_policy_checkpoints/traffic_base_policy/checkpoint_000025"

                # Configure evaluation workers
                ppo_config = ppo_config.evaluation(
                    # evaluation_interval=None,  # We'll call evaluate() manually
                    # evaluation_duration=10,     # Number of episodes to evaluate
                    # evaluation_duration_unit="episodes",
                    evaluation_num_workers=1,   # Use 1 worker for evaluation
                )

            
                # Create the algorithm - explicitly specify PPO for RLlib 2.7
                algo = ppo_config.build()
                # Path to the default policy inside the checkpoint.
                # pol_ckpt = (
                #     Path(base_checkpoint)
                #     / "policies"
                #     / "default_policy"          # change if your ID is different
                # )

                # pretrained_policy = Policy.from_checkpoint(pol_ckpt)  # env-free load  :contentReference[oaicite:0]{index=0}
                # # algo.get_policy().set_state(pretrained_policy.get_state())
                # algo.get_policy().set_weights(pretrained_policy.get_weights())  # ← weights only
                # algo.workers.sync_weights()      # push to remote workers

                # zero out the weights of the policy
                # algo.get_policy().set_weights(np.zeros_like(algo.get_policy().get_weights()))
                # algo.workers.sync_weights()

                # Override the compute_actions method to return random actions
                def make_policy_random(worker):
                    """Modify a worker's policy to act uniformly randomly."""
                    policy = worker.get_policy()
                    
                    # Save the original method
                    original_compute_actions_from_input_dict = policy.compute_actions_from_input_dict
                    
                    # Create a random action function that preserves VF predictions
                    def random_compute_actions_from_input_dict(input_dict, **kwargs):
                        # Call the original method to get VF predictions and other info
                        original_actions, state_outs, extra_fetches = original_compute_actions_from_input_dict(input_dict, **kwargs)
                        
                        # Replace actions with random ones
                        obs_batch = input_dict["obs"]
                        batch_size = len(obs_batch)
                        random_actions = np.array([policy.action_space.sample() for _ in range(batch_size)])
                        
                        # Return random actions but keep the original state_outs and extra_fetches (including VF predictions)
                        return random_actions, state_outs, extra_fetches
                    
                    # Replace the method
                    policy.compute_actions_from_input_dict = random_compute_actions_from_input_dict
                
                # Apply the random policy to all evaluation workers
                algo.evaluation_workers.foreach_worker(make_policy_random)
                
                # Now evaluate with the random policy
                result = algo.evaluate()

                print (f"gt_rew_i: {gt_rew_i}, flip_sign: {flip_sign}, env_config_type: {env_config_type}")
                # print (result)
                episode_reward_mean = result["evaluation"]["sampler_results"]["episode_reward_mean"]
                print (f"episode_reward_mean: {episode_reward_mean}")

                key = f"{gt_rew_i}_{flip_sign}_{env_config_type}"
                gt_rew_fn2mean_return[key] = episode_reward_mean
                
                algo.stop()
                
    with open(f"data/learned_policy_evals/uniform_policy_{args.env_type}_gt_rew_fns2mean_return.pkl", "wb") as f:
        pickle.dump(gt_rew_fn2mean_return, f)

if __name__ == "__main__":
    main()
    

