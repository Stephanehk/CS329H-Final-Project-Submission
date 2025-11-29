import numpy as np
from scipy.optimize import linprog
import random
import json
import sys
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from reward_learning.active_learning_utils import find_full_weight_space
from reward_learning.evaluate_reward_weights import evaluate_reward_weights,plot_feature_counts,plot_offline_pref_feature_counts
from pathlib import Path
import itertools
from scipy.spatial import ConvexHull

def assign_synth_pref(feature_pair, feature_rewards, return_pref_prob=False):
    """
    Assigns a preference between two feature vectors using a Boltzmann distribution.
    The reward for each vector is calculated as the sum of (feature_count * feature_reward)
    for all features.
    
    Args:
        feature_pair: Tuple of (f1, f2) where each is a numpy array of feature values
        feature_rewards: Dictionary mapping feature names to their reward values
        
    Returns:
        1 if f1 is preferred, -1 if f2 is preferred
    """
    f1, f2 = feature_pair
    
    # Calculate rewards for each vector
    reward1 = sum(f1[i] * reward for i, reward in enumerate(feature_rewards.values()))
    reward2 = sum(f2[i] * reward for i, reward in enumerate(feature_rewards.values()))

    if return_pref_prob:
        prob = 1.0 / (1.0 + np.exp((reward1-reward2)))
        return prob

    if reward1 == reward2:
        return 0
    
    # # Calculate probability using Boltzmann distribution
    # # Using temperature parameter of 1.0 for now
    # temp = 1.0
    # prob = 1.0 / (1.0 + np.exp((reward2 - reward1) / temp))
    
    # Sample preference based on probability
    return 1 if reward2 > reward1 else -1

def load_reward_ranges(env_name,range_ceiling,horizon):
    """
    Load reward ranges and feature names for a given environment.
    Returns feature_ranges, binary_features, feature_names, and feature_rewards
    with binary features placed contiguously at the end of each list.
    """
    # Load reward ranges
    ranges_file = f"generated_objectives/{env_name}_reward_ranges.json"
    if not os.path.exists(ranges_file):
        raise ValueError(f"Reward ranges file not found: {ranges_file}")
    
    with open(ranges_file, 'r', encoding='utf-8') as f:
        reward_ranges = json.load(f)
    
    # Process ranges and determine binary features
    feature_ranges = []
    binary_features = []
    feature_names = list(reward_ranges.keys())
    
    # Initialize feature rewards dictionary with zeros
    feature_rewards = {name: 0.0 for name in feature_names}
    
    # First pass: collect all features and determine which are binary
    continuous_features = []
    binary_features_list = []
    continuous_ranges = []
    binary_ranges = []
    
    for feature_name in feature_names:
        range_str = reward_ranges[feature_name]
        # Check if range is discrete (denoted by curly braces)
        if '{' in range_str:
            # Extract values from {val1, val2} format
            values = range_str.split('{')[1].split('}')[0].split(',')
            min_val = float(values[0].strip())
            max_val = float(values[1].strip())
            binary_features_list.append(True)
            binary_ranges.append(list(set([min_val*i for i in range(horizon)]+[max_val*i for i in range(horizon)])))
            binary_features.append(feature_name)
        else:
            # Extract values from (min, max) or [min, max] format
            range_str = range_str.split('Range: ')[1].strip("'")
            if "(-inf" in range_str:
                min_val = -range_ceiling
            else:
                match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', range_str.split(',')[0])
                min_val = float(match.group())
            
            if "inf)" in range_str:
                max_val = range_ceiling
            else:
                match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', range_str.split(',')[1])
                max_val = float(match.group())
            
            continuous_features.append(feature_name)
            continuous_ranges.append((min_val*horizon, max_val*horizon))
            binary_features_list.append(False)
    
    # Combine continuous and binary features in order
    feature_names = continuous_features + binary_features
    # feature_ranges = continuous_ranges + binary_ranges
    binary_features = [False] * len(continuous_features) + [True] * len(binary_features)
    
    # Reorder feature_rewards to match the new order
    feature_rewards = {name: feature_rewards[name] for name in feature_names}
    
    return binary_ranges,continuous_ranges , binary_features, feature_names, feature_rewards

def sample_random_feature_pair(binary_feature_ranges,continious_feature_ranges, binary_features, cieling=500):
    """
    Randomly samples feature vectors f1 and f2 within their valid ranges.
    
    Args:
        feature_ranges: List of (min, max) tuples for each feature
        binary_features: List of booleans indicating if each feature is binary
        
    Returns:
        Tuple of (f1, f2) as numpy arrays
    """
    f1_vals, f2_vals = [], []
    for idx, (low, high) in enumerate(continious_feature_ranges):
        if low < -cieling:
            low = -cieling
        if high > cieling:
            high = cieling
        
        f1_vals.append(random.uniform(low, high))
        f2_vals.append(random.uniform(low, high))
    
    for idx, val in enumerate(binary_feature_ranges):
        f1_vals.append(random.choice(val))
        f2_vals.append(random.choice(val))

    
    return np.array(f1_vals, dtype=float), np.array(f2_vals, dtype=float)

def compute_cross_entropy_loss(true_weights, pred_weights):
    """
    Compute cross entropy loss between true and predicted weights.
    First converts weights to probabilities using softmax.
    """
    # Convert dictionaries to numpy arrays in same order
    true_vals = np.array(list(true_weights.values()))
    pred_vals = np.array(list(pred_weights.values()))
    
    # Apply softmax to convert to probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()
    
    true_probs = softmax(true_vals)
    pred_probs = softmax(pred_vals)
    
    # Compute cross entropy loss
    epsilon = 1e-15  # Small constant to avoid log(0)
    pred_probs = np.clip(pred_probs, epsilon, 1 - epsilon)
    loss = -np.sum(true_probs * np.log(pred_probs))
    
    return loss

class PreferenceNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PreferenceNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    def get_weights(self):
        return {name: weight.item() for name, weight in zip(self.feature_names, self.linear.weight[0])}

def main():
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python active_pref_learning.py <environment_name>")
        print("Example: python active_pref_learning.py pandemic")
        sys.exit(1)
    
    env_name = sys.argv[1]
    if env_name == "pandemic":
        horizon = 192
    
    # Load reward ranges and feature information
    binary_feature_ranges, continious_feature_ranges, binary_features, feature_names, feature_rewards = load_reward_ranges(env_name, range_ceiling=float('inf'), horizon=horizon)
    
    if env_name == "pandemic":
        # Load weights from saved JSON file
        policy_names_str = "pandemic_base_policy_2025-05-05_21-29-00"
        weights_path = Path("reward_learning_data") / f"pandemic_weights_{policy_names_str}.json"
        
        if not weights_path.exists():
            raise ValueError(f"Weights file not found: {weights_path}")
            
        with open(weights_path, 'r', encoding='utf-8') as f:
            weights_dict = json.load(f)
            
        # Remove non-weight entries from the dictionary
        for k in feature_rewards.keys():
            feature_rewards[k] = weights_dict[k]
        feature_rewards["PublicHealthOutcomesReward"]=0
        eval_policy_names = ["pandemic_base_policy","2025-05-05_21-29-00"]
    else:
        raise ValueError("Other Environments Are Not Implemented Yet")

    print("==================")
    print(feature_names)
    print(feature_rewards)
    print("==================")

    # Initialize neural network
    input_dim = len(feature_names)
    model = PreferenceNetwork(input_dim)
    model.feature_names = feature_names  # Store feature names for later use
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize data collection
    stopping_num = 1000
    all_pairs = []
    preferences = []

    # Collect preferences and train network
    for iteration in range(stopping_num):
        # Sample random feature pair
        f1, f2 = sample_random_feature_pair(binary_feature_ranges, continious_feature_ranges, binary_features)
        
        # Get true preference
        pref = assign_synth_pref((f1, f2), feature_rewards)
        if pref == 0:  # Skip if equal preference
            continue
        
        # Convert to tensors
        # f1_tensor = torch.tensor(f1)
        # f2_tensor = torch.tensor(f2)
        # pref_tensor = torch.FloatTensor([1.0 if pref == -1 else 0.0])  # Convert to binary classification
        
        # Store pair and preference
        all_pairs.append([f1, f2])
        preferences.append(1.0 if pref == -1 else 0.0)# Convert to binary classification TODO fix hacky formatting later

    # print (all_pairs)
    all_pairs = torch.tensor(all_pairs).float()
    preferences = torch.tensor(preferences).float()
    for train_iter in range(5000):
        pred_returns = model(all_pairs)
        pred_diff = pred_returns[:,0] - pred_returns[:,1]
        pred_diff = pred_diff.squeeze()
    
        loss = criterion(pred_diff, preferences)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if train_iter % 100 == 0:
            print(f"Iteration {train_iter}, Loss: {loss.item():.4f}")

    # Get learned weights
    learned_weights = model.get_weights()
    print("Learned weight vector:", learned_weights)
    print("True weight vector:", feature_rewards)
    
    # Evaluate learned weights
    evaluate_reward_weights(env_name, "rollout_data/", eval_policy_names, feature_rewards, learned_weights,n_trajectories_per_policy= 50)
    plot_feature_counts(env_name, "rollout_data/",eval_policy_names,feature_rewards, "eval_rollout_data_feature_counts.png", n_trajectories_per_policy= 50)
    plot_offline_pref_feature_counts(all_pairs.numpy().tolist(), feature_names, "generated_rollout_data_feature_counts.png")

    model.eval()  # Set to evaluation mode
    test_pairs = []
    test_preferences = []
    with torch.no_grad():
        for _ in range(10000):
            f1, f2 = sample_random_feature_pair(binary_feature_ranges, continious_feature_ranges, binary_features)
            true_pred = assign_synth_pref((f1, f2), feature_rewards)
            if true_pred == 0:
                continue

            test_pairs.append([f1, f2])
            test_preferences.append(1.0 if pref == -1 else 0.0)# Convert to binary classification TODO fix hacky formatting later



        test_pairs = torch.tensor(test_pairs).float()
        test_preferences = torch.tensor(test_preferences).float()
        pred_returns = model(all_pairs)
        pred_diff = pred_returns[:,0] - pred_returns[:,1]
        pred_diff = pred_diff.squeeze()
    
        loss = criterion(pred_diff, preferences)


        avg_loss = loss
        print(f"Avg. cross entropy loss over 10000 unf. generated samples: {avg_loss:.4f}")
        print("======================\n")

    return all_pairs, preferences, learned_weights

if __name__ == "__main__":
    main()
