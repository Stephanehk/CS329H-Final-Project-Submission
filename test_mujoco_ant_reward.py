import inspect
import importlib.util
import sys
from pathlib import Path
import numpy as np
# from pandemic_simulator.environment.reward import SumReward
from utils.mujoco_gt_rew_fns import RewardFunction
from utils.mujoco_observation import MujocoObservation
from typing import Sequence

class SumReward:
    def __init__(self, reward_functions: Sequence[RewardFunction], weights: Sequence[float]):
        """Initialize a sum reward function.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: List of weights for each reward function
        """
        if len(reward_functions) != len(weights):
            raise ValueError("Number of reward functions must match number of weights")
        
        self._reward_fns = reward_functions
        self._weights = weights

    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        """Calculate the weighted sum of all reward functions.
        
        Args:
            prev_obs: Previous observation
            action: Action taken
            obs: Current observation
            
        Returns:
            float: Weighted sum of all reward values
        """
        total_reward = 0.0
        for reward_fn, weight in zip(self._reward_fns, self._weights):
            reward = reward_fn.calculate_reward(prev_obs, action, obs)
            total_reward += weight * reward

        total_reward = total_reward
        #*0.1 #scale down the reward by 10x to avoid overflow
        return total_reward

def load_reward_classes(env_type="mujoco",debug=False, no_convo_base_line=False):
    """Dynamically load reward classes from the generated objectives file."""
    # Get the path to the generated objectives file
    if no_convo_base_line:
        folder = "generated_objectives_no_convo_baseline"
    else:
        folder = "generated_objectives_debug" if debug else "generated_objectives"
    objectives_path = Path(folder) / f"{env_type}_generated_objectives.py"
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(f"{env_type}_generated_objectives", objectives_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{env_type}_generated_objectives"] = module
    spec.loader.exec_module(module)
    
    # Find all classes that inherit from RewardFunction
    reward_classes = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, module.RewardFunction) and 
            obj != module.RewardFunction):  # Exclude the base class itself
            reward_classes.append(obj)
    
    return reward_classes

def create_mujoco_ant_reward(env_type="mujoco",debug=False,weights = None, no_convo_base_line=False):
    # Get all reward classes
    reward_classes = load_reward_classes(env_type=env_type,debug=debug, no_convo_base_line=no_convo_base_line)
    
    # Create instances of all reward functions
    reward_functions = [cls() for cls in reward_classes]
    
    # Create weight vector of all 1's (equal weights)
    if weights is None:
        weights = [1.0] * len(reward_functions)
    # weights = [1.0] * len(reward_functions)
    # weights = np.load(f"active_learning_res/pandemic_gpt-4o-mini_prefs_feasible_weights.npy")

    
    # Create and return the SumReward instance
    return SumReward(reward_functions, weights)
