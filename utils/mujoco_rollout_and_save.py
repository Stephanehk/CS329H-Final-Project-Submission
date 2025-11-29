import os
import pickle
import numpy as np
import ray
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.policy.sample_batch import SampleBatch
from typing import List, Dict, Any, Optional
import torch
from ray.rllib.algorithms.ppo import PPO

from utils.mujoco_config import get_config
from utils.trajectory_types import TrajectoryStep
from ray.tune.registry import register_env
from rl_utils.reward_wrapper import RewardWrapper
from utils.mujoco_gt_rew_fns import DataGenerationMujocoAntRewardFunction
from utils.mujoco_observation import MujocoObservation

# Configure MuJoCo for headless rendering (required for clusters without display)
# This must be set before importing mujoco or creating any environments
# os.environ['MUJOCO_GL'] = 'osmesa'  # Use OSMesa for offscreen rendering
# Alternative: os.environ['MUJOCO_GL'] = 'egl'  # Use EGL if osmesa is not available
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

def _record_all_episodes(episode_id: int) -> bool:
    """Episode trigger function for video recording. Records all episodes."""
    return True

# CPU software fallback (comment the two lines above and uncomment below if EGL isn't available):
# os.environ.setdefault("MUJOCO_GL", "osmesa")
# os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# Avoid stale X display in headless jobs:
os.environ.pop("DISPLAY", None)

def _reset_mujoco_renderer(env):
    # Peel wrappers to the raw mujoco env
    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    unwrapped = getattr(unwrapped, "unwrapped", unwrapped)

    # Close and null the renderer so it is lazily re-created next render()
    if hasattr(unwrapped, "mujoco_renderer") and unwrapped.mujoco_renderer is not None:
        try:
            unwrapped.mujoco_renderer.close()
        except Exception:
            pass
        try:
            unwrapped.mujoco_renderer = None
        except Exception:
            pass


def _mujoco_env_creator(cfg):
    """Create a Mujoco Ant environment with the specified config."""
    env_name = cfg.get("env_name", "Ant-v4")
    base_env = gym.make(env_name)
    
    # If reward function is specified in config, wrap with RewardWrapper
    if "gt_rew_i" in cfg:
        reward_fn = DataGenerationMujocoAntRewardFunction(data_generation_mode=cfg["gt_rew_i"])
        return RewardWrapper(base_env, env_name="mujoco", reward_function=reward_fn)
    
    return base_env

def rollout_and_save(
    checkpoint_path: str,
    policy_name: str,
    save_dir: str,
    num_episodes: int = 100,
    max_steps: int = 1000,  # Default Mujoco Ant horizon
    gt_rew_i: int = 0,
    save_video: bool = False,
    video_dir: Optional[str] = None,
) -> None:
    """
    Roll out a policy and save trajectories to disk using pickle.
    
    Args:
        checkpoint_path: Path to the policy checkpoint
        policy_name: Name identifier for the policy
        save_dir: Directory to save trajectories
        num_episodes: Number of episodes to roll out
        max_steps: Maximum number of steps per episode
        gt_rew_i: Ground truth reward index for data generation mode
        save_video: Whether to save video recordings of episodes
        video_dir: Directory to save videos (defaults to save_dir/videos)
    """
    # Load the checkpoint
    checkpoint_info = get_checkpoint_info(checkpoint_path)
    
    print("policy_name:", policy_name)
    print("checkpoint_path:", checkpoint_path)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    # Register environment
    register_env("mujoco_env", _mujoco_env_creator)
    
    # Extract the config and override GPU settings
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    config = state["config"].copy()
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    config["num_rollout_workers"] = 1
    config["evaluation_num_workers"] = 1

    # Create algorithm and restore checkpoint
    algo = PPO(config=config)
    algo.restore(checkpoint_path)
    
    # Get config
    env_config = get_config()
    
    # Create base environment (without reward wrapper for direct control)
    # Use render_mode="rgb_array" if saving videos
    render_mode = "rgb_array" if save_video else None
    
    # Wrap with video recorder if requested
    
    # Set up the reward function for this specific data generation mode
    reward_fn = DataGenerationMujocoAntRewardFunction(data_generation_mode=gt_rew_i)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Roll out episodes
    for episode in range(num_episodes):
        print(f"Rolling out episode {episode + 1}/{num_episodes}")
        
        base_env = gym.make(env_config.get("env_name", "Ant-v4"),terminate_when_unhealthy=False, render_mode=render_mode)

        if save_video:
            
            if video_dir is None:
                video_dir = os.path.join(save_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            
            # Create a subdirectory for this policy
            policy_video_dir = os.path.join(video_dir, policy_name)
            os.makedirs(policy_video_dir, exist_ok=True)
            
            # Wrap environment with RecordVideo
            base_env = RecordVideo(
                base_env,
                video_folder=policy_video_dir,
                episode_trigger=_record_all_episodes,  # Record all episodes
                name_prefix=f"{policy_name}_episode_{episode}"
            )
            print(f"Video recording enabled. Videos will be saved to: {policy_video_dir}")
    
        
        # Reset environment
        obs_np, info = base_env.reset()
        if save_video:
            base_env.render()
        obs = MujocoObservation(obs_np, base_env)

        done = False
        steps = 0
        trajectory: List[TrajectoryStep] = []
        episode_reward = 0
        
        while steps < max_steps:
            if steps % 100 == 0:
                print(f"  Step {steps}")
            
            # Get action from policy
            action = algo.compute_single_action(obs_np, policy_id="default_policy")
            
            # Step environment
            next_obs_np, original_reward, terminated, truncated, info = base_env.step(action)
            next_obs = MujocoObservation(next_obs_np, base_env)
            done = terminated or truncated
            
            # Calculate custom reward using the reward function
            custom_reward = reward_fn.calculate_reward(obs, action, next_obs)
            
            # The "true" reward is the original Ant reward
            # The "proxy" reward is what was used during training (custom reward)
            true_reward = original_reward
            proxy_reward = custom_reward
            
            # Remove env references to make observations picklable
            obs.env = None
            next_obs.env = None
            
            # Store step
            trajectory.append(TrajectoryStep(
                obs=obs,
                action=action,
                next_obs=next_obs,
                true_reward=custom_reward,
                proxy_reward=custom_reward,
                done=done
            ))

            episode_reward += custom_reward
            
            obs = next_obs
            obs_np = next_obs_np
            steps += 1
            
        print(f"Episode {episode} reward: {episode_reward:.2f}, steps: {steps}")
        if save_video:
            base_env.close_video_recorder()
            _reset_mujoco_renderer(base_env)

            base_env.close()
            del base_env
        else:
            base_env.close()
            del base_env
        # Save trajectory using pickle with policy name in filename
        save_path = os.path.join(save_dir, f"mujoco_{policy_name}_trajectory_{episode}_full.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(trajectory, f)
    
    # Close environment
    # base_env.close()
    
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
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Rollout and save Mujoco Ant trajectories')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for paths (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None, help='End index for paths (default: None, process all)')
    parser.add_argument('--num_episodes', type=int, default=20, help='Number of episodes per policy (default: 20)')
    parser.add_argument('--env_name', type=str, default="mujoco", help='Environment name (default: mujoco_ant)')
    parser.add_argument('--save_video', action='store_true', help='Save video recordings of episodes')
    parser.add_argument('--video_dir', type=str,  default="data/rollout_data/videos", help='Directory to save videos')
    args = parser.parse_args()

    #python3 -m utils.mujoco_rollout_and_save --start_idx 0  --end_idx 2 --num_episodes 20 --env_name mujoco
    
    # Load checkpoint paths
    checkpoint_paths_file = f"data/gt_rew_fn_data/{args.env_name}_gt_rew_fns2checkpoint_paths.pkl"
    with open(checkpoint_paths_file, "rb") as f:
        paths = pickle.load(f)
    print(f"Total paths to process: {len(paths)}")
    
    # Get all keys and slice based on start_idx and end_idx
    all_keys = list(paths.keys())[args.start_idx:args.end_idx]
    
    for gt_rew_i in all_keys:
        checkpoint_path = paths[gt_rew_i]
        save_dir = "rollout_data/trajectories/"

        
        print(f"=================={gt_rew_i}========================")
        policy_name = f"policy_{gt_rew_i}"
        if type(gt_rew_i) == str and "checkpoint" in gt_rew_i:
            gt_rew_i = int(gt_rew_i.split("_checkpoint_")[0])

        rollout_and_save(
            checkpoint_path=checkpoint_path,
            policy_name=policy_name,
            save_dir=save_dir,
            num_episodes=args.num_episodes,
            max_steps=1000,  # Standard Ant horizon
            gt_rew_i=gt_rew_i,
            save_video=args.save_video,
            video_dir=args.video_dir,
        )

