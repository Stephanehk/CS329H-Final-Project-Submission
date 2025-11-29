import numpy as np
from scipy.optimize import linprog
import random
import json
import sys
import os
import re
import torch
from reward_learning.evaluate_reward_weights import evaluate_reward_weights
from pathlib import Path
import itertools

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

def main():
    np.random.seed(1)
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python active_pref_learning.py <environment_name>")
        print("Example: python active_pref_learning.py pandemic")
        sys.exit(1)
    
    env_name = sys.argv[1]
    if env_name == "pandemic":
        horizon = 192
        n_trajectories_per_policy=50
    elif env_name == "glucose":
        horizon = 20*12*24
        n_trajectories_per_policy=10
    
    # Load reward ranges and feature information
    #TODO: need to actually figure out what range_ceiling is 
    binary_feature_ranges,continious_feature_ranges, binary_features, feature_names, feature_rewards = load_reward_ranges(env_name, range_ceiling=float('inf'),horizon=horizon)
    
    # assert False
    if env_name == "pandemic":
        # Load weights from saved JSON file
        policy_names_str = "pandemic_base_policy_2025-05-05_21-29-00"  # This should match the policy names used in learn_pandemic_reward_weights.py
        weights_path = Path("reward_learning_data") / f"pandemic_weights_{policy_names_str}.json"
        
        if not weights_path.exists():
            raise ValueError(f"Weights file not found: {weights_path}")
            
        with open(weights_path, 'r') as f:
            weights_dict = json.load(f)
            
        # for k in feature_rewards.keys():
        #     feature_rewards[k] = weights_dict[k]
        for k in feature_rewards.keys():
            feature_rewards[k] = np.random.uniform(-10, 10)
        # feature_rewards = {k: v for k, v in weights_dict.items() if k not in ['r2_score', 'kendall_tau', 'kendall_tau_p_value']}
        eval_policy_names = ["pandemic_base_policy","2025-05-05_21-29-00"]
    elif env_name == "glucose":
        for k in feature_rewards.keys():
            feature_rewards[k] = np.random.uniform(-10, 10)
        eval_policy_names = ["glucose_base_policy","2025-05-12_14-12-46"]
        #glucose_glucose_base_policy_2025-05-12_14-12-46_rollout_data.pkl
    else:
        raise ValueError("Other Environments Are Not Implemented Yet")
    reward_dim = len(feature_names)

    print ("==================")
    print (feature_names)
    print (feature_rewards)
    print ("==================")

    # Initialize other variables
    stopping_num = 10000
    n_pairs2sampler_per_iter = 1000
    n_init_pairs = 0
    

    preferences = []
    all_pairs = []

    print ("***n_init_pairs:", n_init_pairs)
    for _ in range(n_init_pairs):
        f1, f2 = sample_random_feature_pair(binary_feature_ranges,continious_feature_ranges, binary_features, cieling=500)
        pref = assign_synth_pref((f1, f2), feature_rewards)
        if pref == 0:
            continue
        all_pairs.append((f1, f2))
        preferences.append(pref)

    A_ub, b_ub, A_eq, b_eq, bounds = generate_inequalities(all_pairs, preferences, dim=reward_dim)
    for iteration in range(stopping_num):
        # construct the current feasible weight-space polyhedron
        # poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim=reward_dim)
        

        best_pair = None
        highest_uncertainty = float("-inf")
        for _ in range(n_pairs2sampler_per_iter):
            f1, f2 = sample_random_feature_pair(binary_feature_ranges,continious_feature_ranges, binary_features, cieling=500)
            
            
            min_val, max_val = compute_min_and_max_dot( A_ub, b_ub, A_eq, b_eq,bounds, f2-f1)
            uncertainty_in_direction = max_val - min_val
           
            # uncertainty_in_direction = max_val - min_val
            # assert uncertainty_in_direction >= 0
            if uncertainty_in_direction > highest_uncertainty and np.sign(max_val) != np.sign(min_val):
                highest_uncertainty = uncertainty_in_direction
                best_pair = (f1, f2)
        if best_pair is not None:
            if highest_uncertainty == 0:
                raise ValueError("Couldn't find a point where w might disagree on the preference")
            f1, f2 = best_pair
            pref = assign_synth_pref((f1, f2), feature_rewards)
            if pref == 0:
                break
            all_pairs.append((f1, f2))
            preferences.append(pref)
        else:
            print ("Couldn't find a splitting pair, increasing n_pairs2sampler_per_iter")
            n_pairs2sampler_per_iter = int (n_pairs2sampler_per_iter*1.1)

        # inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
        A_ub, b_ub, A_eq, b_eq,bounds = generate_inequalities(all_pairs, preferences, dim=reward_dim)

       
        # A_ub = np.array(inequalities, dtype=float) #Negative sign coverts this to the form where we wish to finx w that satisfies A'w < b  instead of Aw>b
        # b_ub = np.array(b, dtype=float)

        dim_w = A_ub.shape[1]
        #TODO: we assume assume no conflicts for now
        # all_pairs, preferences = mod_check_and_remove_conflicts(all_pairs, preferences, task_description, feature_names, binary_features)
        result = linprog(c=[1]*dim_w, A_ub=A_ub, b_ub=b_ub,A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        print ("A_ub.shape:",A_ub.shape)
        print ("highest uncertainty:",highest_uncertainty)

        #------sanity check-------
        # min_val_after, max_val_after = compute_min_and_max_dot(inequalities, b, f2-f1)
        # assert np.sign(max_val_after) == np.sign(min_val_after)
        # uncertainty_in_direction = max_val - min_val
        # uncertainty_in_direction = -max_val - min_val
        # assert uncertainty_in_direction == 0
        # print("Before:", min_val, max_val)
        # print("After :", min_val_after, max_val_after)
        #------------------------

        print ("\n")
       
        
        if result.success and iteration % 10 == 0:
            feasible_w = result.x[:reward_dim]
            # l1 = np.sum(np.abs(feasible_w))
            # if l1 == 0:                         # degenerate, all-zero solution
            #     raise RuntimeError("LP returned w = 0; check constraints.")
            # feasible_w /= l1      

            # Verify that feasible_w satisfies all previous preferences
            all_satisfied = True
            for (f1, f2), pref in zip(all_pairs, preferences):
                dot1 = np.dot(feasible_w, f1)
                dot2 = np.dot(feasible_w, f2)
                if pref == 1 and dot1 >= dot2:
                    print(f"Warning: Preference not satisfied for pair {f1}, {f2}")
                    print(f"Expected f1 < f2 but got {dot1} >= {dot2}")
                    all_satisfied = False
                elif pref == -1 and dot1 <= dot2:
                    print(f"Warning: Preference not satisfied for pair {f1}, {f2}")
                    print(f"Expected f1 > f2 but got {dot1} <= {dot2}")
                    all_satisfied = False
            
            if not all_satisfied:
                raise ValueError("Warning: Found weights do not satisfy all preferences!")
            
            # Construct dictionary mapping feature names to feasible weights
            feasible_w_dict = {name: weight for name, weight in zip(feature_names, feasible_w)}
            print ("feasible weight vector:", feasible_w_dict)
            print ("true weight vector:", feature_rewards)
            evaluate_reward_weights(env_name, "rollout_data/", eval_policy_names, feature_rewards, feasible_w_dict, n_trajectories_per_policy=n_trajectories_per_policy)
            
            # Compute cross entropy loss over 1000 random samples
            n_correct = 0
            for _ in range(1000):
                f1, f2 = sample_random_feature_pair(binary_feature_ranges, continious_feature_ranges, binary_features)
                # Get predictions using both weight vectors
                true_pred = assign_synth_pref((f1, f2), feature_rewards)
                if true_pred == 0:
                    continue
                # pred_prob = assign_synth_pref((f1, f2), feasible_w_dict, return_pref_prob=True)
                # pred_prob = max(pred_prob, 1e-10)
                # if pred_prob == 1:
                #     loss = -np.log(pred_prob)*(1 if true_pred == -1 else 0)
                # else:
                #     loss = -(np.log(pred_prob)*(1 if true_pred == -1 else 0) + np.log(1-pred_prob)*(1 if true_pred == 1 else 0))
                # total_loss += loss
                
                n_correct += assign_synth_pref((f1, f2), feasible_w_dict) == true_pred
            
            print(f"Accuracy: {(n_correct/1000):.4f}")
            print ("======================\n")

        elif not result.success:
            raise ValueError("  uh oh! No feasible solution found..")

        
def compute_min_and_max_dot( A_ub, b_ub, A_eq, b_eq, bounds,direction):
    """
    returns (min_val, max_val) of w^T direction subject to w in 'poly' by
    solving two linear programs using the polyhedron's H-representation (inequalities)
    """
    # H = poly.get_inequalities()
    # A_ub = []
    # b_ub = []
    # for row in H:
    #     b_i = float(row[0])
    #     A_i = -np.array([float(x) for x in row[1:]], dtype=float)
    #     A_ub.append(A_i)
    #     b_ub.append(b_i)

    # A_ub = np.array(inequalities, dtype=float) #Not sure why we need the negative sign
    # b_ub = np.array(b, dtype=float)
    reward_dim =len(direction)
    c1 = np.array(direction, dtype=float)
    c2 = np.ones(reward_dim)
    c = np.concatenate((c1,c2))

    neg_c = np.concatenate((-c1,c2))

    # solve for max (c^T w) => min (-(c^T) w).
    res_max = linprog(c=neg_c, A_ub=A_ub, b_ub=b_ub,A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    # print (res_max)
    max_val = float(c1.dot(res_max.x[:reward_dim])) if res_max.success else float('-inf')     

    # solve for min (c^T w)
    res_min = linprog(c=c, A_ub=A_ub, b_ub=b_ub,A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    min_val = float(c1.dot(res_min.x[:reward_dim])) if res_min.success else float('-inf')

    return min_val, max_val



def maximize_dot_margin_lp(A_ub, b_ub, A_eq, b_eq, direction, M=1e3):
    """
    Solve a single LP to maximize max_val - min_val, where:
        min_val <= w^T direction
        max_val >= w^T direction
        |max_val - min_val| <= M
    
    Args:
        A_ub, b_ub, A_eq, b_eq: LP constraints for [w, u]
        direction: vector to define dot product
        M: upper bound on |max_val - min_val|
    
    Returns:
        LP result from scipy.optimize.linprog
    """
    dim = len(direction)
    D = 2 * dim  # w + u
    total_vars = D + 2  # + min_val, max_val
    idx_min_val = D
    idx_max_val = D + 1

    # Pad existing constraints
    A_ub_padded = [np.pad(row, (0, 2)) for row in A_ub]
    A_eq_padded = [np.pad(row, (0, 2)) for row in A_eq]

    # min_val <= w^T direction → -w^T direction + min_val ≥ 0
    c1 = np.zeros(total_vars)
    c1[:dim] = -direction
    c1[idx_min_val] = 1
    A_ub_padded.append(c1)
    b_ub.append(0.0)

    # max_val >= w^T direction → w^T direction - max_val ≥ 0
    c2 = np.zeros(total_vars)
    c2[:dim] = direction
    c2[idx_max_val] = -1
    A_ub_padded.append(c2)
    b_ub.append(0.0)

    # |max_val - min_val| <= M → two inequalities:
    # max_val - min_val <= M
    c3 = np.zeros(total_vars)
    c3[idx_max_val] = 1
    c3[idx_min_val] = -1
    A_ub_padded.append(c3)
    b_ub.append(M)

    # min_val - max_val <= M
    c4 = np.zeros(total_vars)
    c4[idx_min_val] = 1
    c4[idx_max_val] = -1
    A_ub_padded.append(c4)
    b_ub.append(M)

    # Objective: maximize (max_val - min_val) = c^T x
    c = np.zeros(total_vars)
    c[idx_max_val] = 1
    c[idx_min_val] = -1

    bounds = [(-np.inf, np.inf)] * D + [(None, None), (None, None)]

    result = linprog(
        c=-c,  # maximize → minimize negative
        A_ub=np.array(A_ub_padded),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq_padded),
        b_eq=np.array(b_eq),
        bounds=bounds
    )

    return result

# def compute_min_and_max_dot( A_ub, b_ub, A_eq, b_eq, direction):
#     res = maximize_dot_margin_lp(A_ub.tolist(), b_ub.tolist(), A_eq.tolist(), b_eq.tolist(), direction=direction, M=100000.0)

#     dim = len(direction)
#     if res.success:
#         w = res.x[:dim]
#         min_val = res.x[2 * dim]
#         max_val = res.x[2 * dim + 1]
#         print("Success: max_val - min_val =", max_val - min_val)
#     else:
#         print("LP infeasible or unbounded")
#     return min_val, max_val

def generate_inequalities(pairs, preferences, dim, w_range=5.0, force_abs_w0=True):
    """
    Generate LP constraints from preference data with:
    (1) Preference satisfaction constraints
    (2) Element-wise bounds on w ∈ [-w_range, w_range]
    (3) |w_0| = 1 if force_abs_w0 is True (handled via w_0 = ±1)

    Returns:
        A_ub, b_ub: inequality constraints (A_ub @ x <= b_ub)
        A_eq, b_eq: equality constraints (A_eq @ x == b_eq)
        bounds: variable bounds for linprog
    """
    D = 2 * dim  # Variables: [w_0, ..., w_{d-1}, u_0, ..., u_{d-1}]
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    epsilon = 1e-5

    # Preference constraints
    for (f0, f1), pref in zip(pairs, preferences):
        delta_f = f1 - f0
        constraint = np.zeros(D)
        if pref == -1:
            constraint[:dim] = delta_f
            A_ub.append(constraint)
            b_ub.append(-epsilon)
        elif pref == 1:
            constraint[:dim] = -delta_f
            A_ub.append(constraint)
            b_ub.append(-epsilon)
        elif pref == 0:
            constraint1 = np.zeros(D)
            constraint2 = np.zeros(D)
            constraint1[:dim] = delta_f
            constraint2[:dim] = -delta_f
            A_ub.extend([constraint1, constraint2])
            b_ub.extend([0.0, 0.0])

    # Box constraints: -w_range <= w_i <= w_range
    for i in range(dim):
        c1 = np.zeros(D)
        c1[i] = 1
        A_ub.append(c1)
        b_ub.append(w_range)

        c2 = np.zeros(D)
        c2[i] = -1
        A_ub.append(c2)
        b_ub.append(w_range)

    # Add |w_i| <= u_i constraints
    for i in range(dim):
        c1 = np.zeros(D)
        c1[i] = 1     # w_i
        c1[dim + i] = -1  # -u_i
        A_ub.append(c1)
        b_ub.append(0)

        c2 = np.zeros(D)
        c2[i] = -1    # -w_i
        c2[dim + i] = -1  # -u_i
        A_ub.append(c2)
        b_ub.append(0)

    # Add constraint to force |w_0| = 1
    if force_abs_w0:
        # Option 1: you can run this LP twice, once with w_0 = 1 and once with w_0 = -1
        c_eq = np.zeros(D)
        c_eq[0] = 1  # w_0 = ±1
        A_eq.append(c_eq)
        b_eq.append(1.0)
        # You can change b_eq[-1] to -1.0 for the other LP

    # Variable bounds: w_i ∈ [-w_range, w_range], u_i ∈ [0, w_range]
    bounds = [(-w_range, w_range)] * dim + [(0, w_range)] * dim

    return np.array(A_ub), np.array(b_ub), np.array(A_eq), np.array(b_eq), bounds



if __name__ == "__main__":
    main()
