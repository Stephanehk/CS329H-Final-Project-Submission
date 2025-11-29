import inspect
import importlib.util
import sys
from pathlib import Path
from typing import List, Sequence
from generated_objectives.traffic_generated_objectives import RewardFunction
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation
import numpy as np



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
        self, prev_obs: TrafficObservation, action: int, obs: TrafficObservation
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
        return total_reward

def load_reward_classes(debug=False, no_convo_base_line=False):
    """Dynamically load reward classes from the generated objectives file."""
    # Get the path to the generated objectives file
    folder = "generated_objectives_debug" if debug else "generated_objectives_no_convo_baseline" if no_convo_base_line else "generated_objectives"
    objectives_path = Path(folder) / "traffic_generated_objectives.py"
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("traffic_generated_objectives", objectives_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["traffic_generated_objectives"] = module
    spec.loader.exec_module(module)
    
    # Find all classes that inherit from RewardFunction
    reward_classes = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, module.RewardFunction) and 
            obj != module.RewardFunction):  # Exclude the base class itself
            reward_classes.append(obj)
    
    return reward_classes


def create_traffic_reward(debug=False, weights=None, no_convo_base_line=False):
    # Get all reward classes
    reward_classes = load_reward_classes(debug=debug, no_convo_base_line=no_convo_base_line)
    
    # Create instances of all reward functions
    reward_functions = [cls() for cls in reward_classes]
    
    # Create weight vector of all 1's (equal weights)
    if weights is None:
        weights = [1.0] * len(reward_functions)
    # weights = [1.0] * len(reward_functions)
    # weights = np.load(f"active_learning_res/traffic_gpt-4o-mini_prefs_feasible_weights.npy")

    # Create and return the SumReward instance
    return SumReward(reward_functions, weights)

if __name__ == "__main__":
    # Create the reward function
    reward = create_traffic_reward()
    print("Successfully created traffic reward function with equal weights")
    print(f"Number of reward components: {len(reward._reward_fns)}")
    print("Reward components:")
    for i, rf in enumerate(reward._reward_fns):
        print(f"{i+1}. {rf.__class__.__name__}") 
