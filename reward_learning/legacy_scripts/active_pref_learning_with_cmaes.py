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
import cma

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

def get_uncertainty_for_feature_pair(x, inequalities, b, binary_feature_ranges, continuous_feature_ranges, binary_features):
    """
    Compute uncertainty for a feature pair, where x is the flattened concatenation of f1 and f2.
    Returns negative uncertainty since CMA-ES minimizes.
    """
    try:
        dim = len(continuous_feature_ranges) + len(binary_feature_ranges)
        f1 = x[:dim].copy()  # Make copies to avoid modifying input
        f2 = x[dim:].copy()
        
        # Project continuous features to valid ranges
        for i, (low, high) in enumerate(continuous_feature_ranges):
            f1[i] = np.clip(f1[i], low, high)
            f2[i] = np.clip(f2[i], low, high)
        
        # Project binary features to nearest valid value
        for i, valid_vals in enumerate(binary_feature_ranges):
            idx = i + len(continuous_feature_ranges)
            f1[idx] = min(valid_vals, key=lambda x: abs(x - f1[idx]))
            f2[idx] = min(valid_vals, key=lambda x: abs(x - f2[idx]))
        
        min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
        
        # Handle edge cases
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return 1e10  # Return a large positive value for invalid solutions
        
        uncertainty = max_val - min_val
        
        # Return negative uncertainty since CMA-ES minimizes
        # Also return a large positive value if signs are the same (no uncertainty)
        if np.sign(max_val) == np.sign(min_val):
            return 1e10
        return -uncertainty
    except Exception as e:
        print(f"Error in get_uncertainty_for_feature_pair: {e}")
        return 1e10  # Return a large positive value for any errors

def find_best_feature_pair_cmaes(inequalities, b, binary_feature_ranges, continuous_feature_ranges, binary_features, n_pairs2sampler_per_iter=1000):
    """
    Use CMA-ES to find the feature pair with highest uncertainty.
    """
    dim = len(continuous_feature_ranges) + len(binary_feature_ranges)
    
    # Initialize CMA-ES with better parameters
    x0 = np.zeros(2 * dim)  # Initial point (flattened f1 and f2)
    sigma0 = 0.1  # Smaller initial step size for better stability
    
    # Create bounds for continuous features
    lower_bounds = []
    upper_bounds = []
    
    # Add bounds for continuous features (for both f1 and f2)
    for low, high in continuous_feature_ranges:
        lower_bounds.extend([low, low])
        upper_bounds.extend([high, high])
    
    # Add bounds for binary features (for both f1 and f2)
    for valid_vals in binary_feature_ranges:
        min_val = min(valid_vals)
        max_val = max(valid_vals)
        lower_bounds.extend([min_val, min_val])
        upper_bounds.extend([max_val, max_val])
    
    bounds = [lower_bounds, upper_bounds]
    
    # Initialize CMA-ES with better options
    opts = {
        'bounds': bounds,
        'CMA_diagonal': True,  # Use diagonal covariance matrix for better stability
        'CMA_elitist': True,   # Keep best solution
        'maxiter': n_pairs2sampler_per_iter,
        'popsize': 10,         # Smaller population size for better stability
        'seed': np.random.randint(0, 10000),  # Random seed
        'CMA_rankmu': True,    # Use rank-mu update
        'CMA_rankone': True,   # Use rank-one update
        'CMA_mirrors': 0,      # No mirrored sampling
        'CMA_mirrormethod': 0, # No mirroring
        'CMA_active': True,    # Use active CMA
        'CMA_cmean': 1,        # Use weighted mean
        'tolfun': 1e-8,        # Function value tolerance
        'tolx': 1e-8,          # Solution tolerance
        'minstd': 1e-8,        # Minimum standard deviation
        'maxstd': 1e8,         # Maximum standard deviation
        'scaling_of_variables': None  # No scaling
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    # Run optimization
    best_x = None
    best_uncertainty = float('-inf')
    generation = 0
    
    while generation < n_pairs2sampler_per_iter:
        try:
            solutions = es.ask()
            
            # Ensure solutions are within bounds
            for i, sol in enumerate(solutions):
                for j, (low, high) in enumerate(zip(lower_bounds, upper_bounds)):
                    if sol[j] < low:
                        sol[j] = low
                    elif sol[j] > high:
                        sol[j] = high
            
            fitness_values = [get_uncertainty_for_feature_pair(x, inequalities, b, 
                                                             binary_feature_ranges, 
                                                             continuous_feature_ranges, 
                                                             binary_features) 
                             for x in solutions]
            
            # Filter out invalid solutions
            valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
            if valid_indices:
                valid_solutions = [solutions[i] for i in valid_indices]
                valid_fitness = [fitness_values[i] for i in valid_indices]
                es.tell(valid_solutions, valid_fitness)
            
            # Track best solution
            if valid_indices:
                best_idx = np.argmin(valid_fitness)  # CMA-ES minimizes
                if -valid_fitness[best_idx] > best_uncertainty:
                    best_uncertainty = -valid_fitness[best_idx]
                    best_x = valid_solutions[best_idx]
            
            generation += 1
            
        except Exception as e:
            print(f"Error in CMA-ES iteration {generation}: {e}")
            break
    
    if best_x is None:
        return None, 0
    
    # Convert best solution back to feature pairs
    f1 = best_x[:dim]
    f2 = best_x[dim:]
    
    # Project binary features to valid values
    for i, valid_vals in enumerate(binary_feature_ranges):
        idx = i + len(continuous_feature_ranges)
        f1[idx] = min(valid_vals, key=lambda x: abs(x - f1[idx]))
        f2[idx] = min(valid_vals, key=lambda x: abs(x - f2[idx]))
    
    return (f1, f2), best_uncertainty

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
            
        # Remove non-weight entries from the dictionary
        for k in feature_rewards.keys():
            feature_rewards[k] = weights_dict[k]
        # feature_rewards = {k: v for k, v in weights_dict.items() if k not in ['r2_score', 'kendall_tau', 'kendall_tau_p_value']}
        eval_policy_names = ["pandemic_base_policy","2025-05-05_21-29-00"]
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
    n_random_fails = 0
    fallback2CMAES = 10

    preferences = []
    all_pairs = []

    inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
    for iteration in range(stopping_num):
        # inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
        if n_random_fails >= fallback2CMAES:
            # Replace random sampling with CMA-ES
            best_pair, highest_uncertainty = find_best_feature_pair_cmaes(
                inequalities, b, binary_feature_ranges, continious_feature_ranges, 
                binary_features, n_pairs2sampler_per_iter
            )
            print ("falling back to CMAES to find f1/f2:")
            print ("best_pair:", best_pair)
        else:
            best_pair = None
            highest_uncertainty = float("-inf")
            for _ in range(n_pairs2sampler_per_iter):
                f1, f2 = sample_random_feature_pair(binary_feature_ranges,continious_feature_ranges, binary_features, cieling=500)
                min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
                uncertainty_in_direction = max_val - min_val
            
                # uncertainty_in_direction = max_val - min_val
                # assert uncertainty_in_direction >= 0
                if uncertainty_in_direction > highest_uncertainty and np.sign(max_val) != np.sign(min_val):
                    highest_uncertainty = uncertainty_in_direction
                    best_pair = (f1, f2)
            
            if best_pair is None:
                n_random_fails +=1
    
        
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
            print("Couldn't find a splitting pair, increasing n_pairs2sampler_per_iter")
            n_pairs2sampler_per_iter *= 2

        inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
        A_ub = np.array(inequalities, dtype=float) #Negative sign coverts this to the form where we wish to finx w that satisfies A'w < b  instead of Aw>b
        b_ub = np.array(b, dtype=float)

        dim_w = A_ub.shape[1]
        #TODO: we assume assume no conflicts for now
        # all_pairs, preferences = mod_check_and_remove_conflicts(all_pairs, preferences, task_description, feature_names, binary_features)
        result = linprog(c=[0]*dim_w, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))

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
            feasible_w = result.x
            l1 = np.sum(np.abs(feasible_w))
            if l1 == 0:                         # degenerate, all-zero solution
                raise RuntimeError("LP returned w = 0; check constraints.")
            feasible_w /= l1      

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
            evaluate_reward_weights(env_name, "rollout_data/", eval_policy_names, feature_rewards, feasible_w_dict, n_trajectories_per_policy=50)
            
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

        
def compute_min_and_max_dot(inequalities, b, direction):
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

    A_ub = np.array(inequalities, dtype=float) #Not sure why we need the negative sign
    b_ub = np.array(b, dtype=float)
    c = np.array(direction, dtype=float)

    # solve for max (c^T w) => min (-(c^T) w).
    res_max = linprog(c=-c, A_ub=A_ub, b_ub=b_ub)
    max_val = float(c.dot(res_max.x)) if res_max.success else float('inf')     

    # solve for min (c^T w)
    res_min = linprog(c=c, A_ub=A_ub, b_ub=b_ub)
    min_val = float(c.dot(res_min.x)) if res_min.success else float('-inf')

    return min_val, max_val

def generate_inequalities(pairs, preferences, dim,scale=1.0):
    weight_bounds = [(-scale, scale) for _ in range(dim)]

    pref_matrix = []
    b = []
    epsilon = 1e-5
    
    if(len(pairs) != 0):
        assert dim == len(pairs[0][0])
        for (f0, f1), pref in zip(pairs, preferences):
            delta_f = f1 - f0
            if pref == -1:
                pref_matrix.append(list(delta_f))
                b.append(-epsilon)
            elif pref == 1:
                pref_matrix.append(list(-delta_f))
                b.append(-epsilon)
            elif pref == 0:
                pref_matrix.extend([list(delta_f), list(-delta_f)])
                b.extend([0,0])

    if len(weight_bounds) != dim:
        raise ValueError("weight_bounds length must match number of features")
    for i, (L, U) in enumerate(weight_bounds):
        row_lb = [1 if j == i else 0 for j in range(dim)]
        row_ub = [-1 if j == i else 0 for j in range(dim)]
        pref_matrix.extend([row_lb, row_ub])
        b.extend([-L, U])

    #pref_matrix = A, such that Aw > b
    return pref_matrix, b





if __name__ == "__main__":
    main()
