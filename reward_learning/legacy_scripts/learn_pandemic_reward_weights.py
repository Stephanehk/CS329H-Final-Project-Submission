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
import scipy
from scipy.stats import pearsonr
from scipy.stats import kendalltau

from utils.pandemic_rollout_and_save import TrajectoryStep
from rl_utils.reward_wrapper import SumReward

import warnings

# Turn all RuntimeWarnings into errors
warnings.simplefilter("error", RuntimeWarning)

def load_rollout_data(rollout_dir: str, policy_names: List[str], n_trajectories_per_policy: int = 10) -> List[Dict[str, Any]]:
    """
    Load rollout data from trajectory files for multiple policies and save processed data.
    
    Args:
        rollout_dir: Base directory containing trajectory files
        policy_names: List of policy names to load trajectories from
        n_trajectories_per_policy: Number of trajectories to load per policy
        
    Returns:
        List of state transitions (obs, action, next_obs, reward)
    """
    data = []
    trajectory_dir = Path(rollout_dir) / "trajectories"
    
    for policy_name in policy_names:
        for i in range(n_trajectories_per_policy):
            trajectory_path = trajectory_dir / f"{policy_name}_trajectory_{i}.pkl"
            if not trajectory_path.exists():
                print(f"Warning: Trajectory file {trajectory_path} not found")
                continue
                
            # print("trajectory_path:", trajectory_path)
            with open(trajectory_path, 'rb') as f:
                trajectory = pickle.load(f)
            
            # Convert trajectory steps to state transitions
            for step in trajectory:
                data.append({
                    'obs': step.obs,
                    'action': step.action,
                    'next_obs': step.next_obs,
                    'reward': step.true_reward
                })
    
    # Save processed data
    policy_names_str = "_".join(policy_names)
    data_filename = f"pandemic_{policy_names_str}_rollout_data.pkl"
    data_dir = Path(rollout_dir) / "rollout_data"
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / data_filename
    
    print(f"Saving processed rollout data to {data_path}")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    
    return data

def extract_reward_features(obs, action, next_obs, reward_functions):
    """Extract features from each reward function for a given state transition."""
    features = []
    for rf in reward_functions:
        # Calculate individual reward component
        component_reward = rf.calculate_reward(obs, action, next_obs)
        assert np.isscalar(component_reward) and component_reward != float("inf") and component_reward != float("-inf"), f"component_reward must be a scalar, got {type(component_reward)} with shape {getattr(component_reward, 'shape', 'no shape')} for {rf.__class__.__name__}"
        features.append(component_reward)
    return np.array(features)

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def learn_reward_weights(rollout_dir: str, policy_names: List[str], n_trajectories_per_policy: int = 10, debug: bool = False):
    """
    Learn weights for the reward functions using linear regression and neural network.
    
    Args:
        rollout_dir: Base directory containing trajectory files
        policy_names: List of policy names to load trajectories from
        n_trajectories_per_policy: Number of trajectories to load per policy
        debug: If True, load objectives from debug folder and prepend "debug" to output filename
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Load rollout data
        print("Loading rollout data...")
        rollout_data = load_rollout_data(rollout_dir, policy_names, n_trajectories_per_policy)
        print(f"Loaded {len(rollout_data)} state transitions")
        
        # Create reward functions
        print("Creating reward functions...")
        reward = create_pandemic_reward(debug=debug)
        reward_functions = reward._reward_fns
        
        # Prepare data for regression
        print("Preparing data for regression...")
        X = []  # Features (individual reward components)
        y = []  # Target (actual rewards)
        
        # Keep track of which trajectory each transition belongs to
        trajectory_ids = []
        current_traj_id = 0
        transitions_per_trajectory = []
        current_traj_transitions = 0
        
        for transition in rollout_data:
            features = extract_reward_features(
                transition['obs'],
                transition['action'],
                transition['next_obs'],
                reward_functions
            )
            X.append(features)
            y.append(transition['reward'])
            trajectory_ids.append(current_traj_id)
            current_traj_transitions += 1
            
            # If we've seen 100 transitions, assume it's a new trajectory
            # This is a heuristic - you might want to adjust this based on your actual trajectory lengths
            if current_traj_transitions >= 192:
                transitions_per_trajectory.append(current_traj_transitions)
                current_traj_id += 1
                current_traj_transitions = 0
        
        # Add the last trajectory if it has any transitions
        if current_traj_transitions > 0:
            transitions_per_trajectory.append(current_traj_transitions)
        
        X = np.array(X)
        y = np.array(y)

        # Filter out any data points where X contains NaN values
        valid_indices = ~np.isnan(X).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        trajectory_ids = np.array(trajectory_ids)[valid_indices]
        print(f"Removed {len(valid_indices) - np.sum(valid_indices)} data points containing NaN values")
        print(f"Remaining data points: {len(X)}")
        
        # Fit linear regression model
        print("\nFitting linear regression model...")
        model = LinearRegression(fit_intercept=False)  # No intercept since we want pure weighted sum
        model.fit(X, y)
        lr_predictions = model.predict(X)
        lr_mse = mean_squared_error(y, lr_predictions)
        print(f"Linear Regression MSE: {lr_mse:.4f}")

        # Calculate trajectory-level correlation
        unique_trajectories = np.unique(trajectory_ids)
        actual_returns = []
        predicted_returns = []
        
        for traj_id in unique_trajectories:
            traj_mask = trajectory_ids == traj_id
            actual_return = np.sum(y[traj_mask])
            predicted_return = np.sum(lr_predictions[traj_mask])
            actual_returns.append(actual_return)
            predicted_returns.append(predicted_return)
        
        correlation, p_value = pearsonr(actual_returns, predicted_returns)
        print(f"\nTrajectory-level correlation between predicted and actual returns: {correlation:.4f}")
        print(f"P-value: {p_value:.4f}")

        # Calculate Kendall-Tau correlation for linear regression
        gt_indices = np.argsort(actual_returns)[::-1]  # Descending order
        pred_indices = np.argsort(predicted_returns)[::-1]  # Descending order
        tau, tau_p_value = kendalltau(gt_indices, pred_indices)
        print(f"\nLinear Regression Kendall-Tau correlation: {tau:.4f}")
        print(f"P-value:",tau_p_value)

        # Get learned weights from linear regression
        weights = model.coef_
        
        # Print results
        print("\nLearned weights (Linear Regression):")
        for i, (rf, weight) in enumerate(zip(reward_functions, weights)):
            print(f"{i+1}. {rf.__class__.__name__}: {weight:.4f}")
        
        # Calculate R² score
        r2_score = model.score(X, y)
        print(f"\nR² score: {r2_score:.4f}")

        # Save learned weights
        weights_dict = {rf.__class__.__name__: float(weight) for rf, weight in zip(reward_functions, weights)}
        weights_dict['r2_score'] = float(r2_score)
        weights_dict['kendall_tau'] = float(tau)
        weights_dict['kendall_tau_p_value'] = float(tau_p_value)
        
        # Create directory if it doesn't exist
        save_dir = Path("reward_learning_data")
        save_dir.mkdir(exist_ok=True)
        
        # Save weights to JSON file
        policy_names_str = "_".join(policy_names)
        prefix = "debug_" if debug else ""
        save_path = save_dir / f"{prefix}pandemic_weights_{policy_names_str}.json"
        with open(save_path, 'w') as f:
            json.dump(weights_dict, f, indent=4)
        print(f"\nSaved learned weights to {save_path}")

        #print mean/var true reward labels 
        print ("Mean g.t. reward: ",np.mean(y))
        print ("Var of g.t. reward: : ",np.var(y))
        
        # Fit neural network model
        print("\nFitting neural network model...")
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model and training components
        nn_model = SimpleNN(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
        
        # Training loop
        n_epochs = 100
        for epoch in range(n_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = nn_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
        
        # Evaluate neural network
        nn_model.eval()
        with torch.no_grad():
            nn_predictions = nn_model(X_tensor).numpy()
        nn_mse = mean_squared_error(y, nn_predictions)
        print(f"Neural Network MSE: {nn_mse:.4f}")

        # Calculate trajectory-level correlation for neural network
        nn_predicted_returns = []
        for traj_id in unique_trajectories:
            traj_mask = trajectory_ids == traj_id
            predicted_return = np.sum(nn_predictions[traj_mask])
            nn_predicted_returns.append(predicted_return)
        
        nn_correlation, nn_p_value = pearsonr(actual_returns, nn_predicted_returns)
        print(f"\nNeural Network trajectory-level correlation: {nn_correlation:.4f}")
        print(f"P-value: {nn_p_value:.4f}")

        # Calculate Kendall-Tau correlation for neural network
        gt_indices = np.argsort(actual_returns)[::-1]  # Descending order
        nn_pred_indices = np.argsort(nn_predicted_returns)[::-1]  # Descending order
        nn_tau, nn_tau_p_value = kendalltau(gt_indices, nn_pred_indices)
        print(f"\nNeural Network Kendall-Tau correlation: {nn_tau:.4f}")
        print(f"P-value: {nn_tau_p_value:.4f}")

        return weights, model, nn_model
        
    finally:
        # Shutdown Ray
        ray.shutdown()

class PreferenceRewardNetwork(nn.Module):
    def __init__(self, input_size, use_linear_model=True):
        super(PreferenceRewardNetwork, self).__init__()
        if use_linear_model:
            self.network = nn.Sequential(
                nn.Linear(input_size, 1)  # Single layer network
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
    
    def forward(self, x):
        return self.network(x)

def get_traj_features(traj, transition_cache, reward_functions):
    features1 = []
    for step in traj:
        transition_tuple = tuple(np.concatenate((step['obs'].flatten(), step['action'].flatten(), step['next_obs'].flatten())))
        if transition_tuple in transition_cache:
            transition_reward_feats = transition_cache[transition_tuple]
        else:
            transition_reward_feats = extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions)
            transition_cache[transition_tuple] = transition_reward_feats
        features1.append(transition_reward_feats)
    features1 = np.sum(features1, axis=0)
    return features1, transition_cache


def learn_reward_weights_from_preferences(rollout_dir: str, policy_names: List[str], n_trajectories_per_policy: int = 10, n_preference_pairs: int = 5000, use_linear_model: bool = True, debug: bool = False):
    """
    Learn reward weights using preference learning over trajectory pairs.
    
    Args:
        rollout_dir: Base directory containing trajectory files
        policy_names: List of policy names to load trajectories from
        n_trajectories_per_policy: Number of trajectories to load per policy
        n_preference_pairs: Number of preference pairs to generate for training
        use_linear_model: Whether to use a linear model for the reward network
        debug: If True, load objectives from debug folder and prepend "debug" to output filename
    """
    # Load rollout data
    print("Loading rollout data...")
    rollout_data = load_rollout_data(rollout_dir, policy_names, n_trajectories_per_policy)
    print(f"Loaded {len(rollout_data)} state transitions")
    
    # Create reward functions
    print("Creating reward functions...")
    reward = create_pandemic_reward(debug=debug)
    reward_functions = reward._reward_fns
    
    # Group transitions into trajectories
    trajectories = []
    current_trajectory = []
    current_traj_transitions = 0
    seen_returns = []
    
    for transition in rollout_data:
        current_trajectory.append(transition)
        current_traj_transitions += 1
        
        if current_traj_transitions >= 193:  # Assuming 192 transitions per trajectory
            traj_ret = sum([trans["reward"] for trans in current_trajectory])
            #uncomment to prevent ties
            # if traj_ret not in seen_returns:
            #     seen_returns.append(traj_ret)
            #     trajectories.append(current_trajectory)
                
            trajectories.append(current_trajectory)
            current_trajectory = []
            current_traj_transitions = 0

    
    if current_trajectory:  # Add the last trajectory if it has any transitions
        trajectories.append(current_trajectory)
    
    print(f"Grouped data into {len(trajectories)} trajectories")
    
    # Generate preference pairs
    print("Generating preference pairs...")
    X = []  # Feature differences between trajectory pairs
    y = []  # Preferences (1 if first trajectory preferred, 0 if second preferred)
    reward_diffs = []
    transition_cache = {}
    for traj1_idx in range(len(trajectories)):
        for traj2_idx in range(traj1_idx + 1, len(trajectories)):
            traj1 = trajectories[traj1_idx]
            traj2 = trajectories[traj2_idx]
            
            # Calculate total reward for each trajectory
            reward1 = sum(step['reward'] for step in traj1)
            reward2 = sum(step['reward'] for step in traj2)
            
            features1, transition_cache= get_traj_features(traj1, transition_cache,reward_functions)
            features2, transition_cache= get_traj_features(traj2, transition_cache,reward_functions)

            # Extract features for each trajectory
            # features1 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions) 
            #                    for step in traj1], axis=0)
            # features2 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions) 
            #                    for step in traj2], axis=0)

            
            #-------Deterministic preferences---
            #deterministic preferences
            # assert reward1 != reward2
                # continue
            # print (features1, features2)
            X.append(features1 - features2)
            reward_diffs.append(reward1 - reward2)
            if reward1 > reward2:
                y.append(1)
            elif reward1 < reward2:
                y.append(0)
            else: 
                y.append(1)
                y.append(0)
                X.append(features1 - features2)
                reward_diffs.append(reward1 - reward2)

            #-------Stochastic preferences-----
            # Use feature differences as input
            # X.append(features1 - features2)
            
            # # Sample preference from sigmoid of reward difference
            # reward_diff = reward1 - reward2
            # # stochastic preferences
            # preference_prob = 1 / (1 + np.exp(-reward_diff))  # sigmoid
            # y.append(1 if np.random.random() < preference_prob else 0)
            
    # Save X and y arrays
    save_dir = Path("reward_learning_data")
    save_dir.mkdir(exist_ok=True)

    # X = np.load(save_dir / "pandemic_preference_X.npy")
    # y = np.load(save_dir / "pandemic_preference_y.npy")

    X = np.array(X)
    y = np.array(y)

   
    np.save(save_dir / "pandemic_preference_X.npy", X)
    np.save(save_dir / "pandemic_preference_y.npy", y)
    # print(f"Saved preference data to {save_dir}/preference_X.npy and {save_dir}/preference_y.npy")
    
    print(f"Generated {len(X)} preference pairs")
    
    # Create and train reward network
    print("Training reward network...")
    input_size = len(reward_functions)
    reward_network = PreferenceRewardNetwork(input_size, use_linear_model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(reward_network.parameters(), lr=0.0001)
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    n_epochs = 1000
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = reward_network(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    # Evaluate accuracy on the entire dataset
    reward_network.eval()
    with torch.no_grad():
        predictions = reward_network(X_tensor)
        # Apply sigmoid to get probabilities
        preference_probs = torch.sigmoid(predictions.squeeze())
        # Convert probabilities to binary predictions (1 if prob > 0.5, 0 otherwise)
        predicted_preferences = (preference_probs > 0.5).float()
        accuracy = (predicted_preferences == y_tensor).float().mean()
        print ("Raw Accuracy: ", accuracy)

        n_correct = 0
        for pref_prob, true_pref, r_diff in zip(preference_probs, y_tensor, reward_diffs):
            if r_diff == 0:
                if pref_prob < 0.6 and pref_prob > 0.4:
                    n_correct+=1
                continue
            pred_pref = 1 if pref_prob > 0.5 else 0
            if pred_pref == true_pref:
                n_correct += 1

        # accuracy = (predicted_preferences == y_tensor).float().mean()
        accuracy = n_correct / len(y_tensor)
        print(f"\nAccuracy of learned reward network: {accuracy:.4f}")
        
        print (f"# of tied traj pairs:", len([diff for diff in reward_diffs if diff ==0]))
    
    # Extract learned weights from the network
    with torch.no_grad():
        weights = reward_network.network[0].weight.data.numpy().flatten()
    
    # Print learned weights
    print("\nLearned weights (Preference Learning):")
    for i, (rf, weight) in enumerate(zip(reward_functions, weights)):
        print(f"{i+1}. {rf.__class__.__name__}: {weight:.4f}")
    
    # Save learned weights
    weights_dict = {rf.__class__.__name__: float(weight) for rf, weight in zip(reward_functions, weights)}
    
    # Create directory if it doesn't exist
    save_dir = Path("reward_learning_data")
    save_dir.mkdir(exist_ok=True)
    
    # Save weights to JSON file
    policy_names_str = "_".join(policy_names)
    prefix = "debug_" if debug else ""
    save_path = save_dir / f"{prefix}pandemic_preference_weights_{policy_names_str}.json"
    with open(save_path, 'w') as f:
        json.dump(weights_dict, f, indent=4)
    print(f"\nSaved learned weights to {save_path}")
    
    # Compute Kendall-Tau correlation between predicted and actual returns
    print("\nComputing Kendall-Tau correlation...")
    actual_returns = []
    predicted_returns = []
    
    for trajectory in trajectories:
        # Compute actual return
        actual_return = sum(step['reward'] for step in trajectory)
        actual_returns.append(actual_return)
        
        # Compute predicted return using learned weights
        features = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions) 
                          for step in trajectory], axis=0)
        with torch.no_grad():
            predicted_return = reward_network(torch.FloatTensor(features)).item()
        predicted_returns.append(predicted_return)
    
    # Sort indices by returns (descending)
    # gt_indices = np.argsort(actual_returns)[::-1]  # Descending order
    # pred_indices = np.argsort(predicted_returns)[::-1]  # Descending order
    gt_indices = scipy.stats.rankdata(actual_returns)
    pred_indices = scipy.stats.rankdata(predicted_returns)

    print ("ranking by g.treturn:")
    print (gt_indices)

    print ("ranking by predicted return:")
    print (pred_indices)
        
    # Compute Kendall-Tau correlation
    tau, p_value = kendalltau(gt_indices, pred_indices)
    print(f"Kendall-Tau correlation: {tau:.4f}")
    print(f"P-value:",p_value)
    
    
    return weights_dict, reward_network

if __name__ == "__main__":
    rollout_dir = "rollout_data/"
    policy_names = ["pandemic_base_policy","2025-05-05_21-29-00","2025-06-24_13-49-08"]#,pandemic_base_policy "2025-05-05_21-29-00","2025-06-24_13-49-08"
    debug = False  # Set to True to use debug objectives and prepend "debug" to output filename
    pref_weights, pref_model = learn_reward_weights_from_preferences(rollout_dir, policy_names, n_trajectories_per_policy=100, use_linear_model=True, debug=debug) 
    # weights, model, nn_model = learn_reward_weights(rollout_dir, policy_names, n_trajectories_per_policy=50, debug=debug)
