import json
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from test_glucose_reward import create_glucose_reward
import ray
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.policy.sample_batch import SampleBatch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import scipy
from scipy.stats import pearsonr, kendalltau
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
from utils.glucose_config import get_config
import copy
# from generated_objectives.glucose_generated_objectives import GroundTruthReward
from rl_utils.reward_wrapper import SumReward

from utils.glucose_rollout_and_save import TrajectoryStep

from utils.glucose_gt_rew_fns import MagniGroundTruthReward, ExpectedCostGroundTruthReward


def load_rollout_data(rollout_dir: str, policy_names: List[str], n_trajectories_per_policy: int = 10) -> List[Dict[str, Any]]:
    """
    Load rollout data from trajectory files for multiple policies and save processed data.
    Pads trajectories with zero transitions until they reach 2000 steps.
    Sets done=True on the 2000th step for all trajectories.
    
    Args:
        rollout_dir: Base directory containing trajectory files
        policy_names: List of policy names to load trajectories from
        n_trajectories_per_policy: Number of trajectories to load per policy
        
    Returns:
        List of state transitions (obs, action, next_obs, reward)
    """
    data = []
    trajectory_dir = Path(rollout_dir) / "trajectories"
    n_dones = 0
    for policy_name in policy_names:
        traj_length = 0
        for i in range(n_trajectories_per_policy):
            trajectory_path = trajectory_dir / f"{policy_name}_trajectory_{i}.pkl"
            if not trajectory_path.exists():
                print(f"Warning: Trajectory file {trajectory_path} not found")
                continue
                
            with open(trajectory_path, 'rb') as f:
                trajectory = pickle.load(f)
            
            # Convert trajectory steps to state transitions
            for step in trajectory:
                # Create GroundTruthReward instance
                ec_gt_reward = ExpectedCostGroundTruthReward()
                magni_gt_reward = MagniGroundTruthReward()
                # gt_reward_output = gt_reward.calculate_reward(step.obs, step.action, step.next_obs)
                # print(f"True reward: {step.true_reward}, GroundTruthReward output: {gt_reward_output}")
                # print (step.true_reward/gt_reward_output)
                # print (step.true_reward-gt_reward_output)
                # print ("\n")
                data.append({
                    'obs': step.obs,
                    'action': step.action,
                    'next_obs': step.next_obs,
                    # 'reward': step.true_reward,
                    "expected_cost_rew": ec_gt_reward.calculate_reward(step.obs, step.action, step.next_obs),
                    "magni_rew": magni_gt_reward.calculate_reward(step.obs, step.action, step.next_obs),
                    'done': False  # Set done=False for all steps except the last one
                })
                traj_length += 1
           
            # # Pad trajectory with zero transitions until it reaches 2000 steps
            # while traj_length < 2000:
            #     # Create zero transition
            

            #     zero_obs = GlucoseObservation()
            #     zero_obs.bg = np.zeros_like(trajectory[0].obs.bg)
            #     zero_obs.insulin = np.zeros_like(trajectory[0].obs.insulin)
            #     zero_obs.cho = np.zeros_like(trajectory[0].obs.cho)
            #     zero_obs.cost = 0.0

            #     zero_transition = {
            #         'obs': zero_obs,
            #         'action': np.zeros_like(trajectory[0].action),
            #         'next_obs':zero_obs,
            #         'reward': 0.0,
            #         'done': False
            #     }

            #     data.append(zero_transition)
            #     traj_length += 1

            # # Set done=True for the last step (2000th step)
            data[-1]['done'] = True
            n_dones += 1
            print("traj_length:", traj_length)
            traj_length = 0

    # Save processed data
    policy_names_str = "_".join(policy_names)
    data_filename = f"glucose_{policy_names_str}_rollout_data.pkl"
    data_dir = Path(rollout_dir) / "rollout_data"
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / data_filename
    
    print(f"Saving processed rollout data to {data_path}")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print("n_dones:", n_dones)
    return data

def extract_reward_features(obs, action, next_obs, reward_functions):
    """Extract features from each reward function for a given state transition."""
    
    obs.bg = np.array(obs.bg)
    next_obs.bg = np.array(next_obs.bg)
    # obs.insulin = np.array(insulin)
    features = []
    for rf in reward_functions:
        # Calculate individual reward component
        component_reward = rf.calculate_reward(obs, action, next_obs)
        features.append(component_reward)
    return np.array(features)

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


def learn_reward_weights_from_preferences(rollout_dir: str, policy_names: List[str],gt_reward_fn:str, n_trajectories_per_policy: int = 10, n_preference_pairs: int = 2000, use_linear_model: bool = True, resume: bool = False):
    """
    Learn reward weights using preference learning over trajectory pairs.
    
    Args:
        rollout_dir: Base directory containing trajectory files
        policy_names: List of policy names to load trajectories from
        n_trajectories_per_policy: Number of trajectories to load per policy
        n_preference_pairs: Number of preference pairs to generate for training
        use_linear_model: Whether to use a linear model for the reward network
        resume: If True, loads previously saved weights for the same policy_names and gt_reward_fn
    """
    print ("gt_reward_fn:",gt_reward_fn)
    # Load rollout data
    print("Loading rollout data...")
    rollout_data = load_rollout_data(rollout_dir, policy_names, n_trajectories_per_policy)
    print(f"Loaded {len(rollout_data)} state transitions")
    
    # Create reward functions
    print("Creating reward functions...")
    reward = create_glucose_reward()
    reward_functions = reward._reward_fns
    
    # Group transitions into trajectories
    trajectories = []
    current_trajectory = []
    current_traj_transitions = 0
    
    for transition in rollout_data:
        current_trajectory.append(transition)
        current_traj_transitions += 1
        
        if transition["done"]:
            trajectories.append(current_trajectory)
            current_trajectory = []
            current_traj_transitions = 0
    
    if current_trajectory:  # Add the last trajectory if it has any transitions
        trajectories.append(current_trajectory)
    
    print(f"Grouped data into {len(trajectories)} trajectories")
    
    #-----------------------------------------------------------------
    # Generate preference pairs
    print("Generating preference pairs...")
    X = []  # Feature differences between trajectory pairs
    y = []  # Preferences (1 if first trajectory preferred, 0 if second preferred)
    reward_diffs = []
    transition_cache={}
    for traj1_idx in range(len(trajectories)):
        for traj2_idx in range(traj1_idx + 1, len(trajectories)):
            traj1 = trajectories[traj1_idx]
            traj2 = trajectories[traj2_idx]
            
            # Calculate total reward for each trajectory
            reward1 = sum(step[gt_reward_fn] for step in traj1)
            reward2 = sum(step[gt_reward_fn] for step in traj2)

            features1, transition_cache= get_traj_features(traj1, transition_cache,reward_functions)
            features2, transition_cache= get_traj_features(traj2, transition_cache,reward_functions)

            # Extract features for each trajectory
            # features1 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions) 
            #                    for step in traj1], axis=0)
            # features2 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions) 
            #                    for step in traj2], axis=0)
            
            # Use feature differences as input
            X.append(features1 - features2)
            
            # Sample preference from sigmoid of reward difference
            reward_diff = reward1 - reward2
            # preference_prob = 1 / (1 + np.exp(-reward_diff))  # sigmoid
            # preference_prob = scipy.special.expit(reward_diff)

            # y.append(1 if np.random.random() < preference_prob else 0)
            # y.append(1 if reward1 > reward2 else 0)
            reward_diffs.append(reward_diff)
            if reward1 > reward2:
                y.append(1)
            elif reward1 < reward2:
                y.append(0)
            else: 
                y.append(1)
                y.append(0)
                X.append(features1 - features2)
                reward_diffs.append(reward1 - reward2)
    
    # Save X and y to pickle files
    save_dir = Path("reward_learning_data")
    save_dir.mkdir(exist_ok=True)
    policy_names_str = "_".join(policy_names)

    X = np.array(X)
    y = np.array(y)

    X_path = save_dir / f"glucose_preference_X_{policy_names_str}_{gt_reward_fn}.pkl"
    y_path = save_dir / f"glucose_preference_y_{policy_names_str}_{gt_reward_fn}.pkl"
    reward_diffs_path = save_dir / f"glucose_preference_reward_diffs_{policy_names_str}_{gt_reward_fn}.pkl"

   
    with open(X_path, 'wb') as f:
        pickle.dump(X, f)
    with open(y_path, 'wb') as f:
        pickle.dump(y, f)
    with open(reward_diffs_path, 'wb') as f:
        pickle.dump(reward_diffs, f)
    
    #-----------------------------------------------------------------
    # save_dir = Path("reward_learning_data")
    # save_dir.mkdir(exist_ok=True)

    # policy_names_str = "_".join(policy_names)
    # X_path = save_dir / f"glucose_preference_X_{policy_names_str}_{gt_reward_fn}.pkl"
    # y_path = save_dir / f"glucose_preference_y_{policy_names_str}_{gt_reward_fn}.pkl"
    
    # with open(X_path, 'rb') as f:
    #     X=pickle.load(f)
    # with open(y_path, 'rb') as f:
    #     y=pickle.load(f)

    # # # X = X[:4950]
    # # # y = y[:4950]
    #-----------------------------------------------------------------
    # print(f"Saved preference data to {X_path} and {y_path}")
    
    print(f"Generated {len(X)} preference pairs")
    
    # Create and train reward network
    print("Training reward network...")
    input_size = len(reward_functions)
    reward_network = PreferenceRewardNetwork(input_size, use_linear_model)

   
    # Load saved weights if resume is True
    if resume:
        weights_path = save_dir / f"glucose_{gt_reward_fn}_preference_weights_{policy_names_str}.json"
        if weights_path.exists():
            print(f"Loading saved weights from {weights_path}")
            with open(weights_path, 'r') as f:
                saved_weights = json.load(f)
            
            # Convert saved weights to numpy array in the same order as reward_functions
            weights_array = np.array([saved_weights[rf.__class__.__name__] for rf in reward_functions])
            
            # Set the network weights
            with torch.no_grad():
                reward_network.network[0].weight.data = torch.FloatTensor(weights_array).view(1, -1)
                print("Successfully loaded saved weights")
        else:
            print(f"No saved weights found at {weights_path}, starting from scratch")

    criterion = nn.BCEWithLogitsLoss()
    lr = 1e-6
    if resume:
        lr = 1e-8
    
    optimizer = torch.optim.Adam(reward_network.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=0)
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    n_epochs = 10000
    if resume:
        n_epochs = 10000

    best_loss = float('inf')
    best_model_state = None

    for epoch in range(n_epochs):
        total_loss = 0
        reward_network.train()
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = reward_network(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)

        # Save the model if it has the lowest loss so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(reward_network.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, LR: {lr:.6f}")

    # Load best model weights before evaluation
    reward_network.load_state_dict(best_model_state)
    reward_network.eval()

    with torch.no_grad():
        predictions = reward_network(X_tensor)
        # Apply sigmoid to get probabilities
        preference_probs = torch.sigmoid(predictions.squeeze())
        # Convert probabilities to binary predictions (1 if prob > 0.5, 0 otherwise)
        predicted_preferences = (preference_probs > 0.5).float()
        accuracy = (predicted_preferences == y_tensor).float().mean()
        print ("Raw Accuracy: ", accuracy)


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
    save_path = save_dir / f"glucose_{gt_reward_fn}_preference_weights_{policy_names_str}.json"
    with open(save_path, 'w') as f:
        json.dump(weights_dict, f, indent=4)
    print(f"\nSaved learned weights to {save_path}")
    
    # Compute Kendall-Tau correlation between predicted and actual returns
    print("\nComputing Kendall-Tau correlation...")
    actual_returns = []
    predicted_returns = []
    
    for trajectory in trajectories:
        # Compute actual return
        actual_return = sum(step[gt_reward_fn] for step in trajectory)
        actual_returns.append(actual_return)
        
        # Compute predicted return using learned weights
        features = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions) 
                          for step in trajectory], axis=0)
        with torch.no_grad():
            predicted_return = reward_network(torch.FloatTensor(features)).item()
        predicted_returns.append(predicted_return)
    
    # Sort indices by returns (descending)
    gt_indices = scipy.stats.rankdata(actual_returns)
    pred_indices = scipy.stats.rankdata(predicted_returns)
    
    print("ranking by g.t. return:")
    print(gt_indices)
    
    print("ranking by predicted return:")
    print(pred_indices)
    
    # Compute Kendall-Tau correlation
    tau, p_value = kendalltau(gt_indices, pred_indices)
    print(f"Kendall-Tau correlation: {tau:.4f}")
    print(f"P-value: {p_value}")
    

    return weights_dict, reward_network

if __name__ == "__main__":
    rollout_dir = "rollout_data/"
    policy_names = ["glucose_base_policy","2025-05-12_14-12-46","2025-06-24_13-53-32"] #"glucose_base_policy", "2025-05-12_14-12-46"
    pref_weights, pref_model = learn_reward_weights_from_preferences(rollout_dir, policy_names,gt_reward_fn="magni_rew", n_trajectories_per_policy=100, use_linear_model=True, resume=False)
    # pref_weights, pref_model = learn_reward_weights_from_preferences(rollout_dir, policy_names,gt_reward_fn="expected_cost_rew", n_trajectories_per_policy=100, use_linear_model=True, resume=False) 
