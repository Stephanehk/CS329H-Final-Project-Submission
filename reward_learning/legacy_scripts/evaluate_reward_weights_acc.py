import numpy as np
import scipy
from scipy.stats import kendalltau

from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from test_traffic_reward import create_traffic_reward
from reward_learning.active_pref_learning import load_reward_ranges
from utils.pandemic_rollout_and_save import TrajectoryStep

def extract_reward_features(obs, action, next_obs, reward_functions):
    """Extract features from each reward function for a given state transition."""
    features = {}
    for rf in reward_functions:
        # Calculate individual reward component
        component_reward = rf.calculate_reward(obs, action, next_obs)
        assert np.isscalar(component_reward) and component_reward != float("inf") and component_reward != float("-inf"), f"component_reward must be a scalar, got {type(component_reward)} with shape {getattr(component_reward, 'shape', 'no shape')} for {rf.__class__.__name__}"
        features[rf.__class__.__name__] = component_reward
    return features

env_name = "pandemic"

rollout_dir = "rollout_data/"
if env_name == "pandemic":
    horizon = 192
    n_trajectories_per_policy=100
    policy_names = ["pandemic_base_policy", "2025-05-05_21-29-00"]
    reward_fn = "reward"
    from reward_learning.learn_pandemic_reward_weights import load_rollout_data
    reward = create_pandemic_reward()
elif env_name == "glucose":
    horizon = 2000
    n_trajectories_per_policy=100
    policy_names = ["glucose_base_policy", "2025-05-12_14-12-46"]
    reward_fn ="magni_rew" # "expected_cost_rew"
    from reward_learning.learn_glucose_reward_weights import load_rollout_data
    reward = create_glucose_reward()
elif env_name == "traffic":
    horizon = 4000
    n_trajectories_per_policy=100
    policy_names = ["2025-06-17_16-14-06", "traffic_base_policy"]
    reward_fn = "reward"
    from reward_learning.learn_traffic_reward_weights import load_rollout_data
    reward = create_traffic_reward()

reward_functions = reward._reward_fns

binary_feature_ranges,continious_feature_ranges, binary_features, feature_names, feature_rewards = load_reward_ranges(env_name, range_ceiling=float('inf'),horizon=horizon)

weights = np.load(f"active_learning_res/{env_name}_feasible_weights.npy")
feasible_w_dict = {name: weight for name, weight in zip(feature_names, weights)}

rollout_data = load_rollout_data(rollout_dir, policy_names, n_trajectories_per_policy)
# Group transitions into trajectories
trajectories = []
current_trajectory = []
current_traj_transitions = 0

for transition in rollout_data:
    current_trajectory.append(transition)
    current_traj_transitions += 1
    #the pandemic env has fixed trajectory lengths, while the other envs do not
    if env_name == "pandemic" and current_traj_transitions >= 193:
        trajectories.append(current_trajectory)
        current_trajectory = []
        current_traj_transitions = 0
    elif env_name != "pandemic" and transition["done"]:
        trajectories.append(current_trajectory)
        current_trajectory = []
        current_traj_transitions = 0

if current_trajectory:  # Add the last trajectory if it has any transitions
    trajectories.append(current_trajectory)

print(f"Grouped data into {len(trajectories)} trajectories")

gt_returns = []
predicted_returns = []
for trajectory in trajectories:
    #get return of trajectory under ground-truth reward function
    gt_return = sum(transition[reward_fn] for transition in trajectory)
    predicted_return=0
    for transition in trajectory:
        if env_name == "glucose":
            transition['obs'].bg  = np.array(transition['obs'].bg)
            transition['next_obs'].bg  = np.array(transition['next_obs'].bg)

        features = extract_reward_features(
            transition['obs'],
            transition['action'],
            transition['next_obs'],
            reward_functions)
        # Multiply each feature by its corresponding weight
        for feature_name, feature_value in features.items():
            predicted_return += feature_value * feasible_w_dict[feature_name]
    gt_returns.append(gt_return)
    predicted_returns.append(predicted_return)

n_correct = 0
n_tied = 0
n_total = 0
for traj1_idx in range(len(trajectories)):
    for traj2_idx in range(traj1_idx + 1, len(trajectories)):
        n_total +=1
        if gt_returns[traj1_idx] == gt_returns[traj2_idx]:
            n_tied += 1
            if predicted_returns[traj1_idx] == predicted_returns[traj2_idx]:
                n_correct += 1
            continue

        if np.argmax([gt_returns[traj1_idx], gt_returns[traj2_idx]]) == np.argmax([predicted_returns[traj1_idx], predicted_returns[traj2_idx]]):
            n_correct += 1
print (f"pref. accuracy: {n_correct}/{n_total} = {n_correct/n_total}")
print (f"# of trajectory pairs with the same g.t. return: {n_tied}")

gt_indices = scipy.stats.rankdata(gt_returns)
pred_indices = scipy.stats.rankdata(predicted_returns)
tau, p_value = kendalltau(gt_indices, pred_indices)
print(f"Kendall-Tau correlation: {tau:.4f}")
print(f"P-value:",p_value)