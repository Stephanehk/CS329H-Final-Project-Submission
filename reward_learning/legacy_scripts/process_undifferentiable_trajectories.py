# load_saved_pairs.py
"""Utility for re-loading low-percentile (Δfeatures, label) pairs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Sequence, List, Dict, Any
import random
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from test_traffic_reward import create_traffic_reward
import numpy as np
import json


def traffic_sa2desc(trajectory, downsample_n: int = 30):
    """
    Convert a traffic trajectory into natural language descriptions for ChatGPT input.
    
    Args:
        trajectory: List of dictionaries containing 'obs' and 'action' keys
    """
    print("=== TRAJECTORY DATA ===")
    print()
    
    print("Trajectory Steps (sampled for analysis):")
    for step_idx, step in enumerate(trajectory):
        # Only print every 10th step (0, 10, 20, etc.)
        if step_idx % downsample_n != 0:
            continue
        print(f"\nStep {step_idx + 1}:")
        
        if 'obs' not in step:
            print(f"  ERROR: No observation found in step. Available keys: {list(step.keys())}")
            continue
            
        obs = step['obs']
        
        # If obs is a flat array, we need to parse it based on the structure
        # From the context, we know it's 5 components per vehicle: [ego_speed, lead_speed_diff, lead_headway, follow_speed_diff, follow_headway]
        num_vehicles = len(obs.rl_vehicles)
        print(f"  Vehicles controlled: {num_vehicles}")
        
        
        for vehicle_idx in range(num_vehicles):
            ego_speed = obs.ego_speeds[vehicle_idx]
            lead_speed_diff = obs.lead_speed_diffs[vehicle_idx]
            lead_headway = obs.lead_headways[vehicle_idx ]
            follow_speed_diff = obs.follow_speed_diffs[vehicle_idx]
            follow_headway = obs.follow_headways[vehicle_idx]
            
            print(f"    Vehicle {vehicle_idx + 1}:")
            print(f"      ego_speeds: {ego_speed:.4f}")
            print(f"      lead_speed_diffs: {lead_speed_diff:.4f}")
            print(f"      lead_headways: {lead_headway:.4f}")
            print(f"      follow_speed_diffs: {follow_speed_diff:.4f}")
            print(f"      follow_headways: {follow_headway:.4f}")
        
        # Print action
        if 'action' in step:
            action = step['action']
            if isinstance(action, (list, np.ndarray)):
                print(f"  Action (accelerations): {action}")
            else:
                print(f"  Action: {action}")
        else:
            print("  Action: Not found in step")
    
    print("\n" + "-" * 60)
    print("END OF TRAJECTORY DATA")
    print()

def traffic_x2desc(x: np.ndarray, reward_functions, objective_descriptions):
    """
    Convert a numpy array of delta features into natural language descriptions for LLM input.
    
    Args:
        x: np.ndarray, shape (d,) where d is the number of delta features
        reward_functions: List of reward functions corresponding to each feature
        objective_descriptions: Dict mapping reward function names to descriptions
    """
    for f_i in range(x.shape[0]):
        rf_name = reward_functions[f_i].__class__.__name__
        print(f"- {rf_name}: {x[f_i]:.6f}")
    print()
    print("-" * 60)


def _build_filename(
    env_name: str,
    gt_reward_key: str,
    policy_names: Sequence[str],
    percentile: int,
) -> str:
    """Keep the filename logic consistent with the saver."""
    return (
        f"{env_name}_{gt_reward_key}_pairs_p{percentile}_"
        f"{'_'.join(policy_names)}.pkl"
    )


def load_low_percentile_pairs(
    env_name: str,
    gt_reward_key: str,
    policy_names: Sequence[str],
    percentile: int = 10,
    data_dir: str | Path = "reward_learning_data",
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]]:
    """
    Reload the pickled (Δfeatures, label) pairs **and** their corresponding
    (trajectory1, trajectory2) tuples that fall within the lowest `percentile`
    of Mahalanobis distances.

    Returns
    -------
    X            : np.ndarray, shape (N, d)
    y            : np.ndarray, shape (N,)
    traj_pairs   : list of (traj1, traj2) for each saved pair
    """
    data_dir = Path(data_dir)
    fpath = data_dir / _build_filename(env_name, gt_reward_key, policy_names, percentile)

    if not fpath.exists():
        raise FileNotFoundError(
            f"Could not locate saved pair file:\n  {fpath}\n"
            f"Ensure `save_low_percentile_pairs` was called with matching arguments."
        )

    with open(fpath, "rb") as f:
        payload = pickle.load(f)

    # Backwards compatibility: older files may lack 'traj_pairs'
    if "traj_pairs" not in payload:
        raise KeyError(
            f"'traj_pairs' not found in {fpath}. "
            "Make sure the saving function version that stores trajectory pairs was used."
        )

    X          = np.asarray(payload["X"])
    y          = np.asarray(payload["y"])
    traj_pairs = payload["traj_pairs"]

    print(
        f"Loaded {len(X)} pairs (≤ {percentile}-th percentile) "
        f"+ {len(traj_pairs)} trajectory pairs from {fpath}"
    )
    return X, y, traj_pairs

env_name = "traffic"
policy_names = [
    "traffic_base_policy",
    "2025-06-17_16-14-06",
    "2025-06-24_13-51-42",
]

X_low, y_low, traj_pairs_low = load_low_percentile_pairs(
    env_name=env_name,
    gt_reward_key="true_reward",
    policy_names=policy_names,
    percentile=10,            # must match the saved cutoff
)


if env_name == "glucose":
    reward = create_glucose_reward()
elif env_name == "pandemic":
    reward = create_pandemic_reward()
elif env_name == "traffic":
    reward = create_traffic_reward()
else:
    raise ValueError(f"Unknown env_name: {env_name}")

with open(f"generated_objectives/{env_name}_objective_descriptions.json", 'r') as f:
    objective_descriptions = json.load(f)

reward_functions = reward._reward_fns  # noqa: SLF001 – preserve original API

print("X_low shape:", X_low.shape)
print("y_low shape:", y_low.shape)
print("Number of traj pairs:", len(traj_pairs_low))

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Sample 100 indices (or all if less than 100 available)
n_samples = min(10, len(X_low))
sample_indices = np.random.choice(len(X_low), size=n_samples, replace=False)

print(f"\nSampling {n_samples} elements...")
print("=" * 70)

# Create comprehensive LLM prompt
# Updated comprehensive LLM prompt
print("Task Description: Choosing the accelerations for each vehicle in a fleet of autonomous vehicles on an on-ramp attempting to merge into traffic on a highway.")
print()
print("You will review trajectory pairs from an autonomous vehicle traffic scenario. Each pair has one trajectory consistently preferred by a stakeholder, yet the current set of measurable objectives (features) you've previously identified does not adequately explain the stakeholder's preferences. The existing feature differences between trajectories are insufficient for capturing or distinguishing the stakeholder's preferred outcomes.")
print()
print("Your task is to propose additional measurable objectives derived from the traffic environment's observation and action spaces. These new objectives should clearly differentiate preferred trajectories from non-preferred ones, aligning closely with demonstrated stakeholder preferences. Aim for objectives that capture nuanced or subtle aspects missed by the existing features.")
print()
print("Guidelines for Identifying New Objectives:")
print("- Consider direct observations as well as combinations or derived metrics from observations and actions.")
print("- Explore temporal relationships, interactions between vehicles, and situational context.")
print("- Identify measurable behaviors or patterns that logically explain why the stakeholder prefers certain trajectories.")
print("- Avoid redundancy; ensure new objectives provide distinct explanatory power.")
print()
print("Trajectory Pair Presentation Format:")
print("- Stakeholder Preference (which trajectory is preferred)")
print("- Detailed sequential observations and actions for each trajectory") 
print("- Current feature differences (which fail to explain the stakeholder preference)")
print()
print("Observation Components (unchanged):")
print("- ego_speeds: Speed of the autonomous vehicle, normalized by maximum speed")
print("- lead_speed_diffs: Speed difference between autonomous vehicle and its leader, normalized by maximum speed")
print("- lead_headways: Following distance to the leading vehicle, normalized by maximum length")
print("- follow_speed_diffs: Speed difference between autonomous vehicle and its follower, normalized by maximum speed")
print("- follow_headways: Following distance to the trailing vehicle, normalized by maximum length")
print()
print("Action Components (unchanged):")
print("- action: Vector of bounded acceleration commands for each autonomous vehicle")
print()
print("Current Feature Components (insufficient for explaining preference):")
for rf in reward_functions:
    rf_name = rf.__class__.__name__
    if rf_name in objective_descriptions:
        print(f"- {rf_name}: {objective_descriptions[rf_name]}")
print()
print("=" * 80)

# Process each sampled trajectory pair
for i, sample_idx in enumerate(sample_indices):
    print(f"\nTrajectory Pair Analysis {i+1} of {n_samples}")
    print("=" * 60)
    
    # Determine preference
    preference = y_low[sample_idx]
    preferred = "Trajectory 1" if preference == 1 else "Trajectory 2"
    print(f"Stakeholder Preference: {preferred} is preferred")
    print()
    
    # Get the trajectory pair
    traj1, traj2 = traj_pairs_low[sample_idx]
    
    # Show trajectory 1
    print("Trajectory 1 Detailed Data:")
    traffic_sa2desc(traj1)
    
    # Show trajectory 2
    print("Trajectory 2 Detailed Data:")
    traffic_sa2desc(traj2)
    
    # Show feature differences
    print("Current Feature Differences (Trajectory 1 - Trajectory 2):")
    print("Note: These feature differences do NOT adequately capture the stakeholder's preference.")
    traffic_x2desc(X_low[sample_idx], reward_functions, objective_descriptions)
    
    print("=" * 80)


with open(f"env_context/traffic_context.txt", "r", encoding='utf-8') as file:
    env_context = file.read()

final_ins = "Below is the source code for the observation space of the environment that implements this task:\n\n"
final_ins += f"{env_context}\n\n"
final_ins +="Your task is to propose additional measurable objectives derived from the traffic environment's observation and action spaces. These new objectives should clearly differentiate preferred trajectories from non-preferred ones, aligning closely with demonstrated stakeholder preferences. Aim for objectives that capture nuanced or subtle aspects missed by the existing features."
print (final_ins)

print ("=" * 80)

with open(f"env_context/traffic_reward_signature.txt", "r", encoding='utf-8') as file:
    env_reward_signature = file.read()
implement_ins = "The RewardFunction interface is defined as:\n"
implement_ins += f"{env_reward_signature}"

implement_ins += "\n\nFor each additional measurable objective you proposed:\n"
implement_ins += "1. Create a new class that inherits from RewardFunction\n"
implement_ins += "2. Name the class based on the objective or pattern (e.g., 'SafeDrivingReward' for the safe driving objective or 'AvoidanceOfRapidSpeedChanges' for the undesirable pattern)\n"
implement_ins += "3. Implement the calculate_reward method using only the variables specified in the list\n"
implement_ins += "4. Use the exact variables and aggregation/penalty methods specified\n"
implement_ins += "5. Return a float value representing the reward (positive for objectives, negative for undesirable patterns)\n\n"
implement_ins += "Important considerations:\n"
implement_ins += "1. Each class you implement must be executable python code. Define all variables and functions you need within the class.\n"
implement_ins += "2. Only use variables that are available in the observation space of the environment\n"
implement_ins += "3. Assume that nothing outside of RewardFunction is available to you. You cannot use any other variables, functions, or libraries, even if they are available in the environment source code. \n"
implement_ins += "4. Implement the exact measurement/detection and aggregation/penalty methods as specified\n"
implement_ins += "5. Do not set arbitrary thresholds or values. Consider using the relative change between variables instead.\n"
implement_ins += "5. Add comments explaining the calculation logic\n"
implement_ins += "6. Handle edge cases appropriately\n"
implement_ins += "7. For objectives, return positive rewards for good behavior\n"
implement_ins += "8. For undesirable patterns, return negative rewards (penalties) when the pattern is detected\n"
implement_ins += "9. Avoid binary (0/1) or ternary (-1/0/1) rewards where possible. Instead, use continuous values that reflect the degree of achievement or violation.\n"
implement_ins += "10. Make sure you do not encounter divide by zero errors. If you are dividing by a variable, make sure it is never zero.\n"
implement_ins += "11. For objectives, if possible, use a continuous scale that reflects how well the objective is being achieved (e.g., instead of just checking if a condition is met, measure how far from optimal the current state is).\n"
implement_ins += "12. For undesirable patterns, if possible, make the penalty proportional to the severity of the violation (e.g., instead of just penalizing when a threshold is crossed, make the penalty proportional to how far the threshold is exceeded).\n"
implement_ins += "Please implement a class for each item in the consolidated list, following the format above and ensuring the rewards are as expressive as possible."

print (implement_ins)