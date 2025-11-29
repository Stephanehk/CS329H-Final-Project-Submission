import json
import numpy as np
import pickle
from pathlib import Path
import torch
from torch import nn
from typing import List, Dict, Tuple
from reward_learning.learn_glucose_reward_weights import PreferenceRewardNetwork
from test_glucose_reward import create_glucose_reward
import matplotlib.pyplot as plt

def plot_feature_difference_histograms(X: np.ndarray, y: np.ndarray, reward_functions: List, predicted_preferences: torch.Tensor):
    """
    Plot histograms of reward feature differences for all pairs and misclassified pairs.
    
    Args:
        X: Feature differences array
        y: True preference labels
        reward_functions: List of reward functions
        predicted_preferences: Predicted preferences from the model
    """
    # Convert to numpy for easier indexing
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    pred_np = predicted_preferences.numpy() if isinstance(predicted_preferences, torch.Tensor) else predicted_preferences
    
    # Get indices of misclassified pairs
    misclassified_indices = np.where(y_np != pred_np)[0]
    
    # Create subplots for each reward feature
    n_features = len(reward_functions)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, (ax, rf) in enumerate(zip(axes, reward_functions)):
        # Plot histogram for all pairs
        ax.hist(X[:, i], bins=50, alpha=0.5, label='All pairs', density=True)
        
        # Plot histogram for misclassified pairs
        ax.hist(X[misclassified_indices, i], bins=50, alpha=0.5, label='Misclassified pairs', density=True)
        
        ax.set_title(f'Distribution of {rf.__class__.__name__} Feature Differences')
        ax.set_xlabel('Feature Difference')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('feature_difference_histograms.png')
    plt.close()

def load_learned_weights_and_data(env_name: str, gt_reward_fn: str, policy_names: List[str]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Load learned weights and training data for a specific environment and ground truth reward function.
    
    Args:
        env_name: Name of the environment (e.g., 'glucose')
        gt_reward_fn: Name of the ground truth reward function
        policy_names: List of policy names used for training
        
    Returns:
        Tuple containing:
        - Dictionary of learned weights
        - Feature differences (X) used for training
        - Preference labels (y) used for training
    """
    save_dir = Path("reward_learning_data")
    policy_names_str = "_".join(policy_names)
    
    # Load weights
    weights_path = save_dir / f"{env_name}_{gt_reward_fn}_preference_weights_{policy_names_str}.json"
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights = json.load(f)
    
    # Load training data
    X_path = save_dir / f"{env_name}_preference_X_{policy_names_str}_{gt_reward_fn}.pkl"
    y_path = save_dir / f"{env_name}_preference_y_{policy_names_str}_{gt_reward_fn}.pkl"
    
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)
    
    return weights, X, y

def analyze_learned_weights(env_name: str, gt_reward_fn: str, policy_names: List[str]):
    """
    Analyze learned weights by evaluating their performance on preference pairs and printing detailed information
    about misclassified pairs.
    
    Args:
        env_name: Name of the environment (e.g., 'glucose')
        gt_reward_fn: Name of the ground truth reward function
        policy_names: List of policy names used for training
    """
    # Load weights and data
    weights, X, y = load_learned_weights_and_data(env_name, gt_reward_fn, policy_names)
    
    # Create reward network with learned weights
    reward = create_glucose_reward()
    reward_functions = reward._reward_fns
    input_size = len(reward_functions)
    reward_network = PreferenceRewardNetwork(input_size, use_linear_model=True)
    
    # Set the learned weights
    with torch.no_grad():
        reward_network.network[0].weight.data = torch.FloatTensor([list(weights.values())])
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Evaluate accuracy
    reward_network.eval()
    with torch.no_grad():
        predictions = reward_network(X_tensor)
        preference_probs = torch.sigmoid(predictions.squeeze())
        predicted_preferences = (preference_probs > 0.5).float()
        accuracy = (predicted_preferences == y_tensor).float().mean()
    
    print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Plot histograms of feature differences
    plot_feature_difference_histograms(X, y_tensor, reward_functions, predicted_preferences)
    
    # Find misclassified pairs
    misclassified_indices = torch.where(predicted_preferences != y_tensor)[0]
    
    print(f"\nFound {len(misclassified_indices)} misclassified pairs")
    print("\nDetailed analysis of misclassified pairs:")
    print("-" * 80)
    
    for idx in misclassified_indices:
        idx = idx.item()
        feature_diff = X[idx]
        true_preference = y[idx]
        predicted_prob = preference_probs[idx].item()
        
        print(f"\nPair {idx}:")
        print(f"True preference: {true_preference}")
        print(f"Predicted probability: {predicted_prob:.4f}")
        print("\nFeature differences:")
        for rf, diff in zip(reward_functions, feature_diff):
            print(f"{rf.__class__.__name__}: {diff:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    # Example usage
    env_name = "glucose"
    gt_reward_fn = "magni_rew"  # or "expected_cost_rew"
    policy_names = ["glucose_base_policy", "2025-05-12_14-12-46"]
    
    analyze_learned_weights(env_name, gt_reward_fn, policy_names)
