import os
import pickle
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.policy.sample_batch import SampleBatch
from typing import List, Dict, Any
import torch
from ray.rllib.algorithms.ppo import PPO  # or whatever algorithm you use

# from pandemic_simulator.environment.interfaces import PandemicObservation
# from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation
from occupancy_measures.agents.orpo import ORPO
from utils.glucose_config import get_config

class TrajectoryStep:
    """Class to represent a single step in a trajectory."""
    def __init__(self, 
                 obs: GlucoseObservation,
                 action: int,
                 next_obs: GlucoseObservation,
                 true_reward: float,
                 done: bool):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.true_reward = true_reward
        self.done= done
    

def rollout_and_save(
    checkpoint_path: str,
    save_dir: str,
    num_episodes: int = 100,
    max_steps: int = 192,  # Default pandemic horizon
) -> None:
    """
    Roll out a policy and save trajectories to disk using pickle.
    
    
    Args:
        checkpoint_path: Path to the policy checkpoint
        save_dir: Directory to save trajectories
        num_episodes: Number of episodes to roll out
        max_steps: Maximum number of steps per episode
        obs_history_size: Number of timesteps of history to include
        num_days_in_obs: Number of days of data in each observation
    """
    # Extract policy name from checkpoint path
    if checkpoint_path == "glucose-uniform-policy":
        policy_name = "glucose-uniform-policy"
        checkpoint_name=""
        checkpoint_info = get_checkpoint_info("rollout_data/2025-07-09_16-56-49/checkpoint_000025/")  # This returns a dict
        input_path= "rollout_data/2025-07-09_16-56-49"
        restore_checkpoint_path = "rollout_data/2025-07-09_16-56-49/checkpoint_000025/"
    else:
        policy_name = checkpoint_path.split("/")[-3]
        checkpoint_name = checkpoint_path.split("/")[-2]
        checkpoint_info = get_checkpoint_info(checkpoint_path)  # This returns a dict
        input_path = "rollout_data/"+policy_name
        restore_checkpoint_path= checkpoint_path

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

    # print (vars(config))
    # assert False
    
    algo = ORPO(config=config)
    # Load the checkpoint
    algo.restore(restore_checkpoint_path)
    
    # Get config and non-essential business locations
    env_config = get_config()
    
    
    # Create environment with non-essential business location tracking
    env = SimglucoseEnv(
        config=env_config
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Roll out episodes
    for episode in range(0, num_episodes,1):
        print(f"Rolling out episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs_np, info = env.reset()
        obs = GlucoseObservation()
        obs.update_obs_with_sim_state(env)
        # obs = GlucoseObservation(obs_np, env)

        # print (obs)
        # assert False
       

        done = False
        steps = 0
        trajectory: List[TrajectoryStep] = []
        episode_reward = 0
        while not done:
            # Get action from policy
            if checkpoint_path == "glucose-uniform-policy":
                action = env.action_space.sample()
            elif "base_policy" in policy_name:
                action = algo.compute_single_action(obs_np, policy_id="safe_policy0")
            else:
                action = algo.compute_single_action(obs_np, policy_id="current")
            
            # Step environment
            next_obs_np, reward, terminated, truncated, info = env.step(action)
            # next_obs = GlucoseObservation(next_obs_np, env)
            next_obs = GlucoseObservation()
            next_obs.update_obs_with_sim_state(env)
            # print (next_obs.bg)
            #assert next_obs.bg is an np array:
            # assert isinstance(next_obs.bg, np.ndarray)
            done = terminated or truncated
            
            # Get true reward from info
            true_reward = info.get("true_reward", reward)
           
            
            # Store step
            trajectory.append(TrajectoryStep(
                obs=obs,
                action=action,
                next_obs=next_obs,
                true_reward=true_reward,
                done=done
            ))

            episode_reward += true_reward
            
            obs = next_obs
            obs_np = next_obs_np
            steps += 1
            
        print(f"Episode {episode} reward: {episode_reward}")
        # Save trajectory using pickle with policy name in filename

        save_path = os.path.join(save_dir, f"{policy_name}_{checkpoint_name}_trajectory_{episode}_full.pkl")
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
    #this is a safe policy
    checkpoint_path_hack = "rollout_data/glucose_base_policy/checkpoint_000300/"
    #this is a reward-hacking policy
    #/next/u/stephhk/orpo/data/logs/glucose/ORPO/proxy/model_256-256/seed_0/2025-05-12_14-12-46
    checkpoint_path_safe = "rollout_data/2025-05-12_14-12-46/checkpoint_000025/"

    save_dir = "rollout_data/trajectories/"
    
    checkpoint_opt = "rollout_data/2025-06-24_13-53-32/checkpoint_000500/"

    checkpoint_1 = "rollout_data/2025-07-09_16-56-49/checkpoint_000025/"
    checkpoint_2 = "rollout_data/2025-07-09_16-56-49/checkpoint_000050/"
    paths = ["glucose-uniform-policy"]
    for checkpoint_path in paths:
        rollout_and_save(
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            num_episodes=50,
        )
        
    # print(f"First step observation shape: {trajectory[0].obs.hourly_data.shape}") 
