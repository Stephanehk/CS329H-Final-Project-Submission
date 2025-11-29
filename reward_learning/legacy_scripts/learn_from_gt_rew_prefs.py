import json
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kendalltau
from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from utils.glucose_config import get_config as get_glucose_config
from utils.pandemic_config import get_config as get_pandemic_config
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from utils.pandemic_rollout_and_save import TrajectoryStep
import argparse
import itertools

# from torch.nn.functional import sigmoid

def stable_sigmoid(x):
    """Numerically stable sigmoid function."""
    x = np.array(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=np.float64)

    # for positive values
    z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    # for negative values
    exp_x = np.exp(x[neg_mask])
    z[neg_mask] = exp_x / (1 + exp_x)

    return z

class PreferenceDataset:
    def __init__(self, trajectory_pairs: List[Tuple[List[Dict], List[Dict]]], preferences: List[int]):
        """
        Args:
            trajectory_pairs: List of (traj1, traj2) pairs
            preferences: List of 0 or 1 indicating preference (0 for first trajectory, 1 for second)
        """
        self.trajectory_pairs = trajectory_pairs
        self.preferences = preferences

class PreferenceRewardNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

def load_rollout_data(rollout_dir: str, policy_names: List[str], n_trajectories_per_policy: int = 10) -> List[Dict[str, Any]]:
    """Load rollout data from trajectory files for multiple policies."""
    data = []
    trajectory_dir = Path(rollout_dir) / "trajectories"
    
    for policy_name in policy_names:
        for i in range(n_trajectories_per_policy):
            trajectory_path = trajectory_dir / f"{policy_name}_trajectory_{i}.pkl"
            if not trajectory_path.exists():
                print(f"Warning: Trajectory file {trajectory_path} not found")
                continue
                
            with open(trajectory_path, 'rb') as f:
                trajectory = pickle.load(f)
            
            # Convert trajectory steps to state transitions
            for step in trajectory:
                data.append({
                    'obs': step.obs,
                    'action': step.action,
                    'next_obs': step.next_obs,
                    'reward': step.true_reward,
                    'done':step.done
                })
    
    return data

def extract_reward_features(obs, action, next_obs, reward_functions, env_name: str):
    """Extract features from each reward function for a given state transition."""
    features = []
    for rf in reward_functions:
        #fix formatting issue I have (not the LLMs fault!)
        if "glucose" in env_name:
            obs.bg = np.array(obs.bg)
            obs.cho = np.array(obs.cho)
            next_obs.bg = np.array(next_obs.bg)
            next_obs.cho = np.array(next_obs.cho)
        component_reward = rf.calculate_reward(obs, action, next_obs)
        features.append(component_reward)
    return np.array(features)

def create_trajectory_pairs(trajectories: List[List[Dict]], n_pairs: Union[int, str]) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Create pairs of trajectories.
    
    Args:
        trajectories: List of trajectories
        n_pairs: Number of pairs to create, or "all" for all possible combinations
    
    Returns:
        List of (traj1, traj2) pairs
    """
    n_trajectories = len(trajectories)
    pairs = []
    
    if n_pairs == "all":
        # Create all possible combinations using itertools
        pairs = list(itertools.combinations(trajectories, 2))
    else:
        # Random sampling with replacement
        seen_coords = set()
        for _ in range(n_pairs):
            i, j = np.random.choice(n_trajectories, 2, replace=False)
            while (i,j) in seen_coords or (j,i) in seen_coords or i == j:
                i, j = np.random.choice(n_trajectories, 2, replace=False)
            seen_coords.add((i,j))
            pairs.append((trajectories[i], trajectories[j]))
    
    return pairs

def sample_preference(traj1: List[Dict], traj2: List[Dict], temperature: float = 1.0) -> int:
    """Sample a preference using Boltzmann distribution based on sum of true rewards."""
    reward1 = sum(step['reward'] for step in traj1)
    reward2 = sum(step['reward'] for step in traj2)
    
    # Flip the probability to match the true reward ordering
    prob = stable_sigmoid(reward1-reward2)  # Note the order is swapped
    
    return np.random.choice([0, 1], p=[prob, 1-prob])

def compute_trajectory_reward(trajectory: List[Dict], reward_network: PreferenceRewardNetwork, reward_functions, env_name) -> float:
    """Compute total reward for a trajectory using the learned reward network."""
    total_reward = 0
    for step in trajectory:
        features = extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions,env_name=env_name)
        with torch.no_grad():
            reward = reward_network(torch.FloatTensor(features)).item()
        total_reward += reward
    return total_reward

def compute_kendall_tau(learned_ranking: List[float], true_ranking: List[float]) -> float:
    """Compute Kendall's tau correlation coefficient between two rankings."""
    tau, p = kendalltau(learned_ranking, true_ranking)
    return tau, p

def learn_from_preferences(env_name: str, rollout_dir: str, policy_names: List[str], n_trajectories_per_policy: int = 10, n_preference_pairs: int = 100, temperature: float = 1.0, load_data: bool = False, n_held_out =10):
    """
    Learn reward function from trajectory preferences.
    
    Args:
        env_name: Either 'glucose' or 'pandemic'
        rollout_dir: Directory containing trajectory files
        policy_names: List of policy names to load trajectories from
        n_trajectories_per_policy: Number of trajectories to load per policy
        n_preference_pairs: Number of preference pairs to generate
        temperature: Temperature parameter for Boltzmann distribution
        load_data: If True, load X and y from saved files instead of computing them
    """
    # Load environment and reward functions
    if env_name == 'glucose':
        env_config = get_glucose_config()
        env = SimglucoseEnv(config=env_config)
        reward = create_glucose_reward()
    else:  # pandemic
        from utils.pandemic_config import get_config
        env_config = get_config()
        env = PandemicPolicyGymEnv(config=env_config,obs_history_size=3,num_days_in_obs=8)
        reward = create_pandemic_reward()
    
    reward_functions = reward._reward_fns
    
    if load_data:
        print("Loading pre-computed training data...")
        X = np.load('temp_X.npy')
        y = np.load('temp_y.npy')
    else:
        # Load trajectories
        print("Loading trajectories...")
        all_transitions = load_rollout_data(rollout_dir, policy_names, n_trajectories_per_policy)
        
        # Group transitions into trajectories
        trajectories = []
        current_trajectory = []
        for transition in all_transitions:
            current_trajectory.append(transition)
            if transition.get('done', False):
                trajectories.append(current_trajectory)
                current_trajectory = []
        if current_trajectory:
            trajectories.append(current_trajectory)
        
        # Shuffle trajectories
        np.random.shuffle(trajectories)
        
        # Split into training and held-out sets
        held_out_trajectories = trajectories[:n_held_out]
        training_trajectories = trajectories[n_held_out:]
        
        print(f"Number of training trajectories: {len(training_trajectories)}")
        print(f"Number of held-out trajectories: {len(held_out_trajectories)}")
        
        # Create preference dataset from training trajectories
        print("Creating preference dataset...")
        trajectory_pairs = create_trajectory_pairs(training_trajectories, n_preference_pairs)
        preferences = [sample_preference(traj1, traj2, temperature) for traj1, traj2 in trajectory_pairs]
        
        # Prepare training data
        X = []  # Feature differences between trajectory pairs
        y = []  # Preferences
        print("Featurizing trajectory segments...")
        for (traj1, traj2), pref in zip(trajectory_pairs, preferences):
            # Extract features for each trajectory
            features1 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions, env_name) 
                               for step in traj1], axis=0)
            features2 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions, env_name) 
                               for step in traj2], axis=0)
            
            # Use feature differences as input
            X.append(features1 - features2)
            y.append(pref)
        
        X = np.array(X)
        y = np.array(y)
        
        # Save the data
        print("Saving training data...")
        np.save('temp_X.npy', X)
        np.save('temp_y.npy', y)

    # Create and train reward network
    print("Training reward network...")
    input_size = len(reward_functions)
    reward_network = PreferenceRewardNetwork(input_size)
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
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    # Evaluate on held-out trajectories
    print("\nEvaluating learned reward function...")
    print(f"Number of held-out trajectories: {len(held_out_trajectories)}")
    
    # Compute rankings
    learned_rewards = [compute_trajectory_reward(traj, reward_network, reward_functions, env_name=env_name) 
                      for traj in held_out_trajectories]
    true_rewards = [sum(step['reward'] for step in traj) for traj in held_out_trajectories]
    
    traj_is_learned = list(range(len(held_out_trajectories)))
    traj_is_learned.sort(key=lambda x: learned_rewards[x])
    
    traj_is_true = list(range(len(held_out_trajectories)))
    traj_is_true.sort(key=lambda x: true_rewards[x])
    
    print("Learned ranking:", traj_is_learned)
    print("True ranking:", traj_is_true)
    tau, p = compute_kendall_tau(traj_is_learned, traj_is_true)
    print(f"Kendall's tau correlation: {tau:.4f}")
    print(f"P value: {p:.6f}")
    
    # Compute test cross entropy loss
    print("\nComputing test cross entropy loss...")
    test_pairs = create_trajectory_pairs(held_out_trajectories, n_preference_pairs)
    test_preferences = [sample_preference(traj1, traj2, temperature) for traj1, traj2 in test_pairs]
    
    X_test = []
    y_test = []
    
    for (traj1, traj2), pref in zip(test_pairs, test_preferences):
        features1 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions, env_name) 
                           for step in traj1], axis=0)
        features2 = np.sum([extract_reward_features(step['obs'], step['action'], step['next_obs'], reward_functions, env_name) 
                           for step in traj2], axis=0)
        X_test.append(features1 - features2)
        y_test.append(pref)
    
    X_test = torch.FloatTensor(np.array(X_test))
    y_test = torch.FloatTensor(np.array(y_test))
    
    reward_network.eval()
    with torch.no_grad():
        outputs = reward_network(X_test)
        test_loss = criterion(outputs.squeeze(), y_test).item()
    
    print(f"Test cross entropy loss: {test_loss:.4f}")
    
    # Print learned weights
    weights = reward_network.linear.weight.data.numpy()[0]
    print("\nLearned reward weights:")
    for i, (rf, weight) in enumerate(zip(reward_functions, weights)):
        print(f"{i+1}. {rf.__class__.__name__}: {weight:.4f}")
    
    return reward_network, tau

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn reward function from preferences')
    parser.add_argument('--env', type=str, default='glucose', choices=['glucose', 'pandemic'],
                      help='Environment to use (glucose or pandemic)')
    parser.add_argument('--load-data', action='store_true',
                      help='Load pre-computed training data instead of computing it')
    args = parser.parse_args()

    # Example usage
    env_name = args.env
    rollout_dir = "rollout_data/"
    if env_name == "glucose":
        policy_names = ["glucose_base_policy"]
    elif env_name == "pandemic":
        # policy_names = ["pandemic_base_policy","2025-05-05_21-29-00"]
        policy_names = ["pandemic_base_policy"]
    
    reward_network, tau = learn_from_preferences(
        env_name=env_name,
        rollout_dir=rollout_dir,
        policy_names=policy_names,
        n_trajectories_per_policy=50,
        n_held_out=20,
        n_preference_pairs="all",
        temperature=1.0,
        load_data=args.load_data
    ) 