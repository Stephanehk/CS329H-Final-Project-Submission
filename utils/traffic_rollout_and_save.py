import os
import pickle
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from typing import List
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import make_multi_agent
import torch
import numpy as np
from flow.utils.registry import make_create_env
from utils.traffic_config import get_config
from utils.trajectory_types import TrajectoryStep
from utils.traffic_gt_rew_fns import merge_true_reward_fn,commute_time, TrueTrafficRewardFunction, penalize_accel, penalize_headway

from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

def rollout_and_save(
    checkpoint_path: str,
    policy_name: str,
    save_dir: str,
    num_episodes: int = 100,
    exp_tag: str = "singleagent_merge_bus",
) -> None:
    """
    Roll out a policy and save trajectories to disk using pickle.
    
    Args:
        checkpoint_path: Path to the policy checkpoint
        save_dir: Directory to save trajectories
        num_episodes: Number of episodes to roll out
    """
    # Get config and create environment
    env_configs = get_config(exp_tag)
    
    create_env, env_name = make_create_env(
        params=env_configs["flow_params_default"],
        reward_specification=env_configs["reward_specification"],
        reward_fun=env_configs["reward_fun"],
        reward_scale=env_configs["reward_scale"],
    )

    print("env_name:", env_name)
    
    # # Register the environment with Ray
    # register_env(env_name, make_multi_agent(create_env))
    # register_env("MergePOEnv", make_multi_agent(create_env))

    #MergePOEnv
    env = create_env()
    
    # Extract policy name from checkpoint path
    if checkpoint_path == "traiffc-uniform-policy":
        policy_name = "traiffc-uniform-policy"
        checkpoint_info = get_checkpoint_info("rollout_data/2025-06-17_16-14-06/checkpoint_000100/")  # This returns a dict
        input_path = "rollout_data_pan_et_al_rew_fns/2025-06-17_16-14-06" #placeholders
        restore_checkpoint_path = "rollout_data/2025-06-17_16-14-06/checkpoint_000100/"
    else:
        # policy_name = checkpoint_path.split("/")[-3] + "/" + checkpoint_path.split("/")[-2]
        # checkpoint_info = get_checkpoint_info(checkpoint_path)  # This returns a dict
        # input_path = "rollout_data/"+policy_name
        checkpoint_info = get_checkpoint_info(checkpoint_path)  # This returns a dict
        input_path = checkpoint_path
        restore_checkpoint_path = checkpoint_path
    
    print("policy_name:", policy_name)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Load the policy
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    # Extract the config and override GPU settings
    config = state["config"].copy()
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    config["num_rollout_workers"] = 1
    config["evaluation_num_workers"] = 1
    config["input_"]=input_path
    config["env"] = make_multi_agent(create_env)

    # config["env_name"] = env_name
    # config["env_config"]["env_name"] = env_name
    # print(config.get("env_config"))
    # print("============")

    algo = PPO(config=config)
    # Load the checkpoint
    algo.restore(restore_checkpoint_path)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    gt_reward_set = TrueTrafficRewardFunction()
    gt_reward_set.set_specific_reward(int(policy_name.replace("policy_","")))
    
    # Roll out episodes
    for episode in range(0, num_episodes, 1):
        print(f"Rolling out episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs_np,last_info = env.reset()
        obs = TrafficObservation()
        obs.update_obs_with_sim_state(env, np.zeros(env.action_space.shape), {"crash":False})

        done = False
        steps = 0
        trajectory: List[TrajectoryStep] = []
        episode_reward = 0
        proxy_episode_reward = 0

        mod_episode_return = 0.0

        crash_total = 0
        mean_vel_total = 0.0
        min_vel_total = 0.0
        speed_squared_total = 0.0
        fuel_total = 0.0
        brake_total = 0.0
        headway_total = 0.0
        infos_did_crash = False
        while not done:
            # print (obs_np)
            # Get action from policy
            if checkpoint_path == "traiffc-uniform-policy":
                action = env.action_space.sample()
            elif "base_policy" in policy_name:
                action = algo.compute_single_action(torch.tensor(obs_np), policy_id="safe_policy0")
            else:
                action = algo.compute_single_action(torch.tensor(obs_np), policy_id="default_policy")
            
            # Step environment
            next_obs_np, reward, terminated, truncated, info = env.step(action)

            if not infos_did_crash and info["crash"]:
                infos_did_crash = True
            # next_obs_np = next_obs_np[0]
            # assert isinstance(next_obs_np, np.ndarray), "next_obs_np is not a NumPy array"

            next_obs = TrafficObservation()
            next_obs.update_obs_with_sim_state(env, action, info)
            

            # print (len(next_obs.rl_vehicles))
            # if len(next_obs.rl_vehicles) == 0:
            #     print(last_info)
            #     print (next_obs)
            #     print (info)
            #     print (next_obs_np)
            #     # If no RL vehicles, we can skip this step
            #     continue
            # assert len(next_obs.rl_vehicles) > 0, "No RL vehicles found in the next observation"

            done = terminated or truncated
            
            # Get true reward from info
            true_reward = info["true_reward"]
            proxy_reward = info["proxy_reward"]
            modified_reward = gt_reward_set.calculate_reward(obs, action, next_obs)

            mod_episode_return += modified_reward

            crash_total += gt_reward_set.cost1
            mean_vel_total += gt_reward_set.cost2
            min_vel_total += gt_reward_set.cost3
            speed_squared_total += gt_reward_set.cost4
            fuel_total += gt_reward_set.cost5
            brake_total += gt_reward_set.cost6
            headway_total += gt_reward_set.cost7

            # print ("true_reward:", true_reward)
            # print ("proxy_reward:", proxy_reward)
            # print ("conputed true reward:", merge_true_reward_fn(env, action))
            # print ("computed :", commute_time(env, action) + penalize_accel(env, action) + 0.1 * penalize_headway(env, action))

            # rf = TrueTrafficRewardFunction()
            # print ("obs.max_speed:", obs.max_speed)
            # print ("next_obs.max_speed:", next_obs.max_speed)
            # print ("obs.max_speed:", obs.max_length)
            # print ("next_obs.max_speed:", next_obs.max_length)


            # rf.sample_linear_reward(prev_obs=obs, action=action, obs=next_obs, weights=rf.weights[0])
            # pred_rew = 0
            # pred_rew = rf.calculate_reward(obs,action, next_obs)
            # print ("conputed true reward with custom obs:", pred_rew)
            # print ("\n")
           
            # Store step
            trajectory.append(TrajectoryStep(
                obs=obs,
                action=action,
                next_obs=next_obs,
                true_reward=true_reward,
                proxy_reward=proxy_reward,
                done=done
            ))

            episode_reward += true_reward
            proxy_episode_reward += proxy_reward
            
            obs = next_obs
            obs_np = next_obs_np
            last_info = info
            steps += 1
            
        print(f"Episode {episode} return: {episode_reward}")
        print(f"Episode {episode} proxy return: {proxy_episode_reward}")
        print (f"Episode {episode} modified return: {mod_episode_return}")

        print (f"Crash total: {crash_total}, Mean vel total: {mean_vel_total}, Min vel total: {min_vel_total}, Speed squared total: {speed_squared_total}, Fuel total: {fuel_total}, Brake total: {brake_total}, Headway total: {headway_total}")
        print ("infos_did_crash:", infos_did_crash)
        print ("---------------------------------------")
        # print (gt_reward_set.weights[int(policy_name.replace("policy_",""))])
        # assert False
        # assert False
        # Save trajectory using pickle with policy name in filename
        if exp_tag != "singleagent_merge_bus":
            save_path = os.path.join(save_dir, f"traffic_{policy_name}_{exp_tag}_trajectory_{episode}_full.pkl")
        else:
            save_path = os.path.join(save_dir, f"traffic_{policy_name}_trajectory_{episode}_full.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(trajectory, f)
    
    # Shutdown Ray
    ray.shutdown()

def load_trajectory(file_path: str) -> List[TrajectoryStep]:
    """
    Load a trajectory from disk.
    
    Args:
        file_path: Path to the trajectory file
        
    Returns:
        List of TrajectoryStep objects
    """
    with open(file_path, 'rb') as f:
        trajectory = pickle.load(f)
    return trajectory

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Rollout and save pandemic trajectories')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for paths (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None, help='End index for paths (default: None, process all)')
    args = parser.parse_args()

    with open("data/gt_rew_fn_data/traffic_gt_rew_fns2checkpoint_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    print ("Total paths to process:", len(paths))

    # Get all keys and slice based on start_idx and end_idx
    all_keys = list(paths.keys())[args.start_idx:args.end_idx]
    
    for rm_id in all_keys:
        checkpoint_path = paths[rm_id]
        # print ("checkpoint_path:", checkpoint_path)
        # print ("rm_id:", rm_id)
        # print (paths)
        # assert False
        save_dir = "rollout_data/trajectories/"
        
        # policy_name = checkpoint_path.split("/")[-3]
        # print ("checkpoint_path:", checkpoint_path)
        # print("policy_name:", policy_name)
        # continue
        print (f"=================={rm_id}========================")
        policy_name = f"policy_{rm_id}"
        rollout_and_save(
            checkpoint_path=checkpoint_path,
            policy_name = policy_name,
            save_dir=save_dir,
            num_episodes=20,
            exp_tag="singleagent_merge_bus_bigger"
        )






