import numpy as np
import pickle
import argparse



env_name = "traffic"

#load the learned policy returns
with open(f"data/learned_policy_evals/learned_policy_{env_name}_gt_rew_fns2mean_return.pkl", "rb") as f:
    learned_policy_returns = pickle.load(f)

#load the uniform policy returns
with open(f"data/learned_policy_evals/uniform_policy_{env_name}_gt_rew_fns2mean_return.pkl", "rb") as f:
    uniform_policy_returns = pickle.load(f)

#load the max rewards
with open(f"data/gt_rew_fn_data/{env_name}_gt_rew_fns2max_reward.pkl", "rb") as f:
    max_rewards = pickle.load(f)

#load checkpoint paths
with open(f"data/gt_rew_fn_data/{env_name}_gt_rew_fns2checkpoint_paths.pkl", "rb") as f:
    checkpoint_paths = pickle.load(f)

#these are the gt_i's that sucesfully trained policies
traffic_gt_i2keep = [0,1,22,23,24, 25, 25, 47, 48, 49]
# for env_config_type in ["easy", "hard"]:
env_config_type = "easy"
for flip_sign in [False, True]:
    for gt_rew_i in range(25):
        
        key = f"{gt_rew_i}_{flip_sign}_{env_config_type}"
        gt_i = gt_rew_i if not flip_sign else gt_rew_i + 25
        max_reward = max_rewards[gt_i]

        checkpoint_path = checkpoint_paths[gt_i]
        best_iter = int(checkpoint_path.split("/")[-1].split("_")[-1])

        if env_name == "traffic" and gt_i not in traffic_gt_i2keep:
            continue

        #exclude checkpoints that achieve max performance before 50 iterations; this indicats the policy is not learning
        # if best_iter < 100:
            # continue
        
        learned_policy_return = learned_policy_returns[key]
        uniform_policy_return = uniform_policy_returns[key]
        if ( max_reward - uniform_policy_return) == 0:
            scaled_ret = 0
        else:
            scaled_ret = (learned_policy_return - uniform_policy_return) /( max_reward - uniform_policy_return)
        print (f"gt_rew_i: {gt_rew_i}, flip_sign: {flip_sign}, env_config_type: {env_config_type}, scaled_return: {scaled_ret}")
        