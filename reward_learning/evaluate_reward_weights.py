import json
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict, Any, Tuple
from scipy.stats import kendalltau
from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from test_traffic_reward import create_traffic_reward
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from utils.pandemic_rollout_and_save import TrajectoryStep
from utils.glucose_config import get_config as get_glucose_config
from utils.pandemic_config import get_config as get_pandemic_config
from utils.traffic_config import get_config as get_traffic_config
from flow.utils.registry import make_create_env

import json
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from test_pandemic_reward import create_pandemic_reward
import ray
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.policy.sample_batch import SampleBatch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import scipy

from utils.pandemic_rollout_and_save import TrajectoryStep

# Add backward compatibility for pickle loading
# import sys
# sys.modules['utils.pandemic_rollout_and_save'] = type('', (), {'TrajectoryStep': TrajectoryStep})()

def load_rollout_data(env_name: str, rollout_dir: str, policy_names: List[str], n_trajectories_per_policy: int = 10) -> List[Dict[str, Any]]:
    """
    Load rollout data from a saved pickle file containing all trajectories.
    
    Args:
        env_name: Either 'glucose' or 'pandemic'
        rollout_dir: Base directory containing trajectory files
        policy_names: List of policy names to load trajectories from
        n_trajectories_per_policy: Number of trajectories to load per policy
        
    Returns:
        List of state transitions (obs, action, next_obs, reward)
    """
    # Create filename based on env_name and policy_names
    policy_names_str = "_".join(policy_names)
    data_filename = f"{env_name}_{policy_names_str}_rollout_data.pkl"
    data_path = Path(rollout_dir) / "rollout_data" / data_filename
    
    if not data_path.exists():
        print(f"Warning: Rollout data file {data_path} not found")
        return []
    
    # print(f"Loading rollout data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def extract_reward_features(env_name: str, obs, action, next_obs, reward_functions, env=None):
    """Extract features from each reward function for a given state transition."""
    features = {}
    for rf in reward_functions:
        # Calculate individual reward component
        component_reward = rf.calculate_reward(obs, action, next_obs)
        features[rf.__class__.__name__] = component_reward
    return features

def compute_trajectory_returns(env_name: str, rollout_data: List[Dict[str, Any]], reward_functions: List, feature_name2weight: Dict[str, float], env=None) -> Tuple[List[float], List[float]]:
    """
    Compute returns for each trajectory using both ground truth rewards and learned weights.
    
    Args:
        env_name: Either 'glucose' or 'pandemic'
        rollout_data: List of state transitions
        reward_functions: List of reward functions
        feature_name2weight: Dictionary mapping feature names to their weights
        env: Environment instance (needed for glucose)
        
    Returns:
        Tuple of (ground_truth_returns, predicted_returns)
    """
    # Group transitions by trajectory (assuming 192 transitions per trajectory)
    if env_name == "pandemic":
        transitions_per_trajectory = 192
    elif env_name == "glucose":
        transitions_per_trajectory = 20*12 * 24
    n_trajectories = len(rollout_data) // transitions_per_trajectory
    
    ground_truth_returns = []
    predicted_returns = []
    
    for traj_idx in range(n_trajectories):
        start_idx = traj_idx * transitions_per_trajectory
        end_idx = start_idx + transitions_per_trajectory
        
        # Get transitions for this trajectory
        traj_transitions = rollout_data[start_idx:end_idx]
        
        # Compute ground truth return
        ground_truth_return = sum(t['reward'] for t in traj_transitions)
        ground_truth_returns.append(ground_truth_return)
        
        # Compute predicted return using learned weights
        predicted_return = 0
        feature_sum = {name: 0.0 for name in feature_name2weight.keys()}

        
        for transition in traj_transitions:
            if env_name == "glucose":
                transition['obs'].bg  = np.array(transition['obs'].bg)
                transition['next_obs'].bg  = np.array(transition['next_obs'].bg)
        

            features = extract_reward_features(
                env_name,
                transition['obs'],
                transition['action'],
                transition['next_obs'],
                reward_functions,
                env
            )
            # Multiply each feature by its corresponding weight
            for feature_name, feature_value in features.items():
                predicted_return += feature_value * feature_name2weight[feature_name]
                feature_sum[feature_name] += feature_value

        # print (feature_sum)
        # print ("\n")
        predicted_returns.append(predicted_return)
    return ground_truth_returns, predicted_returns

def compute_trajectory_ranking_correlation(ground_truth_returns: List[float], predicted_returns: List[float]) -> Tuple[float, float]:
    """
    Compute Kendall-Tau correlation between ground truth and predicted trajectory rankings.
    
    Args:
        ground_truth_returns: List of ground truth returns for each trajectory
        predicted_returns: List of predicted returns for each trajectory
        
    Returns:
        Tuple of (kendall_tau, p_value)
    """
    # Get indices that would sort the returns in descending order
    # gt_indices = np.argsort(ground_truth_returns)[::-1]  # Descending order
    # pred_indices = np.argsort(predicted_returns)[::-1]  # Descending order

    gt_indices = scipy.stats.rankdata(ground_truth_returns)
    pred_indices = scipy.stats.rankdata(predicted_returns)
    
    # Compute Kendall-Tau correlation between the rankings
    tau, p_value = kendalltau(gt_indices, pred_indices)
    
    return tau, p_value

def compute_cross_entropy_loss_over_trajectories(env_name: str, rollout_data: List[Dict[str, Any]], reward_functions: List, feature_rewards: Dict[str, float], feasible_w: Dict[str, float], n_samples: int = 100) -> float:
    """
    Compute cross entropy loss over random trajectory pairs.
    
    Args:
        env_name: Either 'glucose' or 'pandemic'
        rollout_data: List of state transitions
        reward_functions: List of reward functions
        feature_rewards: Dictionary mapping feature names to their reward values
        feasible_w: Learned weights from active preference learning
        n_samples: Number of random trajectory pairs to sample
        
    Returns:
        Average cross entropy loss over valid trajectory pairs
    """
    # Group transitions by trajectory
    if env_name == "pandemic":
        transitions_per_trajectory = 192
    elif env_name == "glucose":
        transitions_per_trajectory = 20*12 * 24
    
    n_trajectories = len(rollout_data) // transitions_per_trajectory
    trajectories = []
    for traj_idx in range(n_trajectories):
        start_idx = traj_idx * transitions_per_trajectory
        end_idx = start_idx + transitions_per_trajectory
        trajectories.append(rollout_data[start_idx:end_idx])

    # Sample random trajectory pairs and compute cross entropy loss
    total_loss = 0
    valid_samples = 0

    for _ in range(n_samples):
        # Sample two random trajectories
        traj1_idx = np.random.randint(0, n_trajectories)
        traj2_idx = np.random.randint(0, n_trajectories)
        while traj2_idx == traj1_idx:  # Ensure different trajectories
            traj2_idx = np.random.randint(0, n_trajectories)
        
        traj1 = trajectories[traj1_idx]
        traj2 = trajectories[traj2_idx]

        # Compute feature sums for each trajectory
        feature_sum1 = {name: 0.0 for name in feature_rewards.keys()}
        feature_sum2 = {name: 0.0 for name in feature_rewards.keys()}

        for transition in traj1:
            features = extract_reward_features(
                env_name,
                transition['obs'],
                transition['action'],
                transition['next_obs'],
                reward_functions,
                None
            )
            for feature_name, feature_value in features.items():
                feature_sum1[feature_name] += feature_value

        for transition in traj2:
            features = extract_reward_features(
                env_name,
                transition['obs'],
                transition['action'],
                transition['next_obs'],
                reward_functions,
                None
            )
            for feature_name, feature_value in features.items():
                feature_sum2[feature_name] += feature_value

        # Compute rewards using both weight vectors
        reward1_feature = sum(feature_sum1[name] * weight for name, weight in feature_rewards.items())
        reward2_feature = sum(feature_sum2[name] * weight for name, weight in feature_rewards.items())
        reward1_learned = sum(feature_sum1[name] * weight for name, weight in feasible_w.items())
        reward2_learned = sum(feature_sum2[name] * weight for name, weight in feasible_w.items())

        # Skip if rewards are equal
        if reward1_feature == reward2_feature:
            continue

        # Get true preference (1 if traj1 preferred, 0 if traj2 preferred)
        true_pref = 1 if reward1_feature > reward2_feature else 0

        # Compute probability using learned weights
        learned_prob = 1.0 / (1.0 + np.exp(reward2_learned - reward1_learned))

        # Compute cross entropy loss
        epsilon = 1e-15  # Small constant to avoid log(0)
        learned_prob = np.clip(learned_prob, epsilon, 1 - epsilon)
        loss = -(true_pref * np.log(learned_prob) + (1 - true_pref) * np.log(1 - learned_prob))
        
        total_loss += loss
        valid_samples += 1

    avg_loss = total_loss / valid_samples if valid_samples > 0 else float('inf')
    return avg_loss, valid_samples



def evaluate_reward_weights(env_name: str, rollout_dir: str, policy_names: List[str], feature_rewards: Dict[str, float], feasible_w: Dict[str, float], n_trajectories_per_policy: int = 50):
    """
    Evaluate learned reward weights by computing Kendall-Tau correlation between rankings.
    """
    # Load rollout data
    rollout_data = load_rollout_data(env_name, rollout_dir, policy_names, n_trajectories_per_policy)
    
    # Create reward functions
    if env_name == 'glucose':
        env_config = get_glucose_config()
        env = SimglucoseEnv(config=env_config)
        reward = create_glucose_reward()
    if env_name == 'traffic':
        env_config = get_traffic_config()
        create_env, env_name = make_create_env(
            params=env_config["flow_params_default"],
            reward_specification=env_config["reward_specification"],
            reward_fun=env_config["reward_fun"],
            reward_scale=env_config["reward_scale"],
        )
        env = create_env()
        reward = create_traffic_reward()
        # env = SimglucoseEnv(config=env_config)
    else:  # pandemic
        env_config = get_pandemic_config()
        env = PandemicPolicyGymEnv(config=env_config, obs_history_size=3, num_days_in_obs=8)
        reward = create_pandemic_reward()
    
    reward_functions = reward._reward_fns
    
    # Compute returns for both ground truth and learned weights
    # gt_returns_feature, pred_returns_feature = compute_trajectory_returns(
    #     env_name, rollout_data, reward_functions, feature_rewards, env
    # )
    gt_returns_learned, pred_returns_learned = compute_trajectory_returns(
        env_name, rollout_data, reward_functions, feasible_w, env
    )
    
    # Compute Kendall-Tau correlation using trajectory rankings
    # tau_feature_vs_learned, p_value_feature_vs_learned = compute_trajectory_ranking_correlation(pred_returns_feature, pred_returns_learned)
    tau_gt_vs_learned, p_value_gt_vs_learned = compute_trajectory_ranking_correlation(gt_returns_learned, pred_returns_learned)

    print("\nResults:")
    # print(f"Input Feature vs. Learned weights Kendall-Tau correlation: {tau_feature_vs_learned:.4f} (p-value: {p_value_feature_vs_learned:.4f})")
    print(f"G.t Reward Fn vs. Learned weights Kendall-Tau correlation: {tau_gt_vs_learned:.4f} (p-value: {p_value_gt_vs_learned:.4f})")



    # Compute cross entropy loss over random trajectory pairs
    # avg_loss, valid_samples = compute_cross_entropy_loss_over_trajectories(
    #     env_name, rollout_data, reward_functions, feature_rewards, feasible_w
    # )
    # print(f"\nCross entropy loss over {valid_samples} trajectory pairs sampled from eval. rollout-data: {avg_loss:.4f}")

    # acc= compute_caccuracy_over_trajectories(env_name, rollout_data, reward_functions, None, feasible_w)
    # print(f"\nAccuracy trajectory pairs sampled from eval. rollout-data w.r.t G.t reward fn: {acc:.4f}")
    return tau_gt_vs_learned

def plot_feature_counts(env_name: str, rollout_dir,policy_names,feature_name2weight, plot_name, n_trajectories_per_policy: int = 50):
    """
    Plot histograms showing the frequency of each feature extracted from extract_reward_features.
    """
    import matplotlib.pyplot as plt

    rollout_data = load_rollout_data(env_name, rollout_dir, policy_names, n_trajectories_per_policy)

    if env_name == 'glucose':
        env_config = get_glucose_config()
        env = SimglucoseEnv(config=env_config)
        reward = create_glucose_reward()
        transitions_per_trajectory = 20*12 * 24
    elif env_name == "pandemic":  # pandemic
        env_config = get_pandemic_config()
        env = PandemicPolicyGymEnv(config=env_config, obs_history_size=3, num_days_in_obs=8)
        reward = create_pandemic_reward()
        transitions_per_trajectory = 192
    
    reward_functions = reward._reward_fns
    
    n_trajectories = len(rollout_data) // transitions_per_trajectory
    
    all_feature_vals = []
    for traj_idx in range(n_trajectories):
        start_idx = traj_idx * transitions_per_trajectory
        end_idx = start_idx + transitions_per_trajectory
        
        # Get transitions for this trajectory
        traj_transitions = rollout_data[start_idx:end_idx]
        
        # Compute predicted return using learned weights
        feature_sum = {name: 0.0 for name in feature_name2weight.keys()}
        
        for transition in traj_transitions:
            features = extract_reward_features(
                env_name,
                transition['obs'],
                transition['action'],
                transition['next_obs'],
                reward_functions,
                env
            )
            # Multiply each feature by its corresponding weight
            for feature_name, feature_value in features.items():
                feature_sum[feature_name] += feature_value
        
        all_feature_vals.append(feature_sum)
    
    # Create a figure with subplots for each feature
    n_features = len(feature_name2weight)
    n_cols = 2
    n_rows = (n_features + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    # Plot histogram for each feature
    for idx, feature_name in enumerate(feature_name2weight.keys()):
        # Extract values for this feature across all trajectories
        feature_values = [round(d[feature_name], 2) for d in all_feature_vals]
        
        # Create histogram
        ax = axes[idx]
        ax.hist(feature_values, bins=20, alpha=0.7)
        ax.set_title(f'Distribution of {feature_name}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Remove any empty subplots
    for idx in range(len(feature_name2weight), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(plot_name)

def plot_offline_pref_feature_counts(all_pairs, feature_names, plot_name):
    """
    Plot histograms showing the distribution of feature values in the offline preference learning pairs.
    
    Args:
        all_pairs: List of (f1, f2) pairs from offline preference learning
        feature_names: List of feature names corresponding to the indices in f1/f2
        plot_name: Name of the file to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Create a figure with subplots for each feature
    n_features = len(feature_names)
    n_cols = 2
    n_rows = (n_features + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    # Plot histogram for each feature
    for idx, feature_name in enumerate(feature_names):
        # Extract values for this feature from both f1 and f2
        feature_values = []
        for f1, f2 in all_pairs:
            feature_values.extend([round(f1[idx], 2), round(f2[idx], 2)])
        
        # Create histogram
        ax = axes[idx]
        ax.hist(feature_values, bins=20, alpha=0.7)
        ax.set_title(f'Distribution of {feature_name}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Remove any empty subplots
    for idx in range(len(feature_names), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(plot_name)

   
