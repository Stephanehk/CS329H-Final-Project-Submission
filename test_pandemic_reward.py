import inspect
import importlib.util
import sys
from pathlib import Path
import numpy as np
from pandemic_simulator.environment.reward import SumReward

def load_reward_classes(debug=False, no_convo_base_line=False):
    """Dynamically load reward classes from the generated objectives file."""
    # Get the path to the generated objectives file
    if no_convo_base_line:
        folder = "generated_objectives_no_convo_baseline"
    else:
        folder = "generated_objectives_debug" if debug else "generated_objectives"
    objectives_path = Path(folder) / "pandemic_generated_objectives.py"
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("pandemic_generated_objectives", objectives_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pandemic_generated_objectives"] = module
    spec.loader.exec_module(module)
    
    # Find all classes that inherit from RewardFunction
    reward_classes = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, module.RewardFunction) and 
            obj != module.RewardFunction):  # Exclude the base class itself
            reward_classes.append(obj)
    
    return reward_classes

def create_pandemic_reward(debug=False,weights = None, no_convo_base_line=False):
    # Get all reward classes
    reward_classes = load_reward_classes(debug=debug, no_convo_base_line=no_convo_base_line)
    
    # Create instances of all reward functions
    reward_functions = [cls() for cls in reward_classes]
    
    # Create weight vector of all 1's (equal weights)
    if weights is None:
        weights = [1.0] * len(reward_functions)
    # weights = [1.0] * len(reward_functions)
    # weights = np.load(f"active_learning_res/pandemic_gpt-4o-mini_prefs_feasible_weights.npy")

    
    # Create and return the SumReward instance
    return SumReward(reward_functions, weights)

if __name__ == "__main__":
    # Create the reward function
    reward = create_pandemic_reward()
    print("Successfully created pandemic reward function with equal weights")
    print(f"Number of reward components: {len(reward._reward_fns)}")
    print("Reward components:")
    for i, rf in enumerate(reward._reward_fns):
        print(f"{i+1}. {rf.__class__.__name__}") 