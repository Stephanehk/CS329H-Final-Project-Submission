import numpy as np
from scipy.optimize import linprog
import random
import json
import sys
import os
import re
import torch
from reward_learning.active_learning_utils import find_full_weight_space
from reward_learning.evaluate_reward_weights import evaluate_reward_weights
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

def optimize_feature_pair(poly, feature_ranges, binary_features, num_steps=1000, learning_rate=0.01):
    """
    Uses gradient descent to find feature vectors f1 and f2 that maximize max_val-min_val
    where direction = f1-f2 and max_val/min_val are computed using compute_min_and_max_dot.
    Handles discrete variables by trying all possible combinations while optimizing continuous variables.
    Enforces range constraints for continuous variables.
    
    Args:
        poly: The current feasible polyhedron
        feature_ranges: List of (min, max) tuples for each feature
        binary_features: List of booleans indicating if each feature is binary
        num_steps: Number of gradient descent steps
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Tuple of (f1, f2) as numpy arrays
    """
    # Get the polyhedron constraints
    H = poly.get_inequalities()
    A_ub = []
    b_ub = []
    for row in H:
        b_i = float(row[0])
        A_i = -np.array([float(x) for x in row[1:]], dtype=float)
        A_ub.append(A_i)
        b_ub.append(b_i)
    A_ub = torch.tensor(np.array(A_ub, dtype=float), dtype=torch.float32)
    b_ub = torch.tensor(np.array(b_ub, dtype=float), dtype=torch.float32)

    print ("A_ub:",A_ub.shape)
    # print (A_ub.shape)
    print ("b_ub:",b_ub.shape)
    # print (b_ub.shape)
    
    # Get indices of binary and continuous features
    binary_indices = [i for i, is_binary in enumerate(binary_features) if is_binary]
    if len(binary_indices) > 10:
        raise ValueError("Too many binary features to optimize")
    continuous_indices = [i for i, is_binary in enumerate(binary_features) if not is_binary]
    
    # Get all possible values for binary features
    binary_values = []
    for idx in binary_indices:
        low, high = feature_ranges[idx]
        binary_values.append([low, high])

  
    
    # Generate all possible combinations of binary values
    binary_combinations = list(itertools.product(*binary_values))

   
    
    best_loss = float('inf')
    all_losses = []
    best_f1 = None
    best_f2 = None
    
    # For each combination of binary values, optimize continuous variables
    for binary_combo in binary_combinations:
        # Initialize values for all features
        f1_vals = []
        f2_vals = []
        
        # Set values for all features
        for i, (low, high) in enumerate(feature_ranges):
            if i in binary_indices:
                # Use binary value from combination
                idx = binary_indices.index(i)
                f1_vals.append(binary_combo[idx])
                f2_vals.append(binary_combo[idx])
            else:
                f1_vals.append(1)
                f2_vals.append(-1)

        # Create tensors with requires_grad=True
        f1 = torch.tensor(f1_vals, requires_grad=True, dtype=torch.float32)
        f2 = torch.tensor(f2_vals, requires_grad=True, dtype=torch.float32)

        # Create a mask to zero out gradients for binary indices
        grad_mask = torch.ones_like(f1)
        for idx in binary_indices:
            grad_mask[idx] = 0.0

        optimizer = torch.optim.Adam([f1, f2], lr=learning_rate)
        
        best_step_loss = float('inf')
        best_step_f1 = None
        best_step_f2 = None
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Compute direction
            direction = (f1 - f2)
            
            # Compute max_val using differentiable optimization
            w = torch.zeros_like(direction, requires_grad=True)
            lambda_ = torch.ones_like(b_ub, requires_grad=True)
            
            # Compute the Lagrangian
            objective = direction @ w
            constraints = A_ub @ w - b_ub
            lagrangian = objective - lambda_ @ constraints            
            
            # Compute gradients
            grad_w = torch.autograd.grad(lagrangian, w, create_graph=True)[0]
            grad_lambda = torch.autograd.grad(lagrangian, lambda_, create_graph=True)[0]
            
            # Update w and lambda
            w = w + learning_rate * grad_w
            lambda_ = torch.clamp(lambda_ - learning_rate * grad_lambda, min=0)
            
            # Compute min_val similarly but with negative direction
            w_min = torch.zeros_like(direction, requires_grad=True)
            lambda_min = torch.ones_like(b_ub, requires_grad=True)
            
            objective_min = -direction @ w_min
            constraints_min = A_ub @ w_min - b_ub
            lagrangian_min = objective_min - lambda_min @ constraints_min
            
            grad_w_min = torch.autograd.grad(lagrangian_min, w_min, create_graph=True)[0]
            grad_lambda_min = torch.autograd.grad(lagrangian_min, lambda_min, create_graph=True)[0]
            
            w_min = w_min + learning_rate * grad_w_min
            lambda_min = torch.clamp(lambda_min - learning_rate * grad_lambda_min, min=0)
            
            # Compute max_val and min_val
            max_val = direction @ w
            min_val = direction @ w_min
            
            # Add range constraint penalties for continuous variables
            range_penalty = 0.0
            for idx in continuous_indices:
                low, high = feature_ranges[idx]
                # Only add penalties if the range is finite
                if not np.isinf(low):
                    range_penalty += torch.relu(low - f1[idx])**2 + torch.relu(low - f2[idx])**2
                if not np.isinf(high):
                    range_penalty += torch.relu(f1[idx] - high)**2 + torch.relu(f2[idx] - high)**2
            
            # Objective: maximize max_val - min_val while satisfying range constraints
            loss = -(max_val - min_val) + 100.0 * range_penalty  # Large penalty coefficient to enforce constraints
            
            # Track best loss and values
            if loss.item() < best_step_loss:
                best_step_loss = loss.item()
                best_step_f1 = f1.detach().clone()
                best_step_f2 = f2.detach().clone()
            
            # Backpropagate
            loss.backward()
            
            # Zero out gradients for binary indices
            f1.grad = f1.grad * grad_mask
            f2.grad = f2.grad * grad_mask
            
            # Update parameters
            optimizer.step()
            
            # Project continuous variables back to valid ranges
            # with torch.no_grad():
            #     for idx in continuous_indices:
            #         low, high = feature_ranges[idx]
            #         if not np.isinf(low) and not np.isinf(high):
            #             f1[idx] = torch.clamp(f1[idx], low, high)
            #             f2[idx] = torch.clamp(f2[idx], low, high)
        # print ("====================\n")
        # assert False
        # Check if this combination gave better results
        if best_step_loss < best_loss:
            best_loss = best_step_loss
            best_f1 = best_step_f1
            best_f2 = best_step_f2
        all_losses.append(best_step_loss)
    print ("best_loss:",best_loss)
    print ("all_losses: ", all_losses)
    
    return best_f1.numpy(), best_f2.numpy()

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

def calculate_polyhedron_volume(poly):
    """
    Calculate the volume of a polyhedron using its vertices.
    
    Args:
        poly: A cdd.Polyhedron object
        
    Returns:
        float: The volume of the polyhedron
    """
    # Get vertices from the polyhedron
    generators = poly.get_generators()
    vertices = []
    for row in generators:
        if int(row[0]) == 1:  # Only consider vertices (t=1), not rays (t=0)
            vertices.append(row[1:])
    
    if not vertices:
        return 0.0
    
    vertices = np.array(vertices)
    
    # If we have fewer vertices than dimensions + 1, volume is 0
    if len(vertices) <= vertices.shape[1]:
        return 0.0
    
    # print (vertices.shape)
    
    # assert False
    try:
        # Compute convex hull and its volume
        hull = ConvexHull(vertices)
        return hull.volume
    except Exception as e:
        print(f"Error computing volume: {e}")
        return 0.0

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
    default_scale = 2
    found_reward = False
    reward_func = []
    preferences = []
    all_pairs = []

    #TODO: we assume the range of all binary features is (0,1), which is not enforced for the LLM-generated features (some are binary features in the range (0,10))
    # feature_ranges = [(0, 1) for _ in range(reward_dim )]
    found_reward = False
    reward_func = []
    attempts = 10000
    #TODO: I always asssume signs is none; still not 100% sure what that var does
    for iteration in range(stopping_num):
        # construct the current feasible weight-space polyhedron
        poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim=reward_dim)
        # print ("...constructed the current feasible weight-space polyhedron")
        # check if polyhedron is effectively a single point (or empty)
        generators = poly.get_generators()
        tvals = [int(row[0]) for row in generators]  # t=1 => vertex, t=0 => ray
        n_vertices = sum(1 for t in tvals if t == 1)
        n_rays = sum(1 for t in tvals if t == 0)

        if (n_vertices == 1 and n_rays == 0):
            print("feasible region is now a single point; no further query can reduce it")
            break

        # find a pair that reduces the feasible region
        # found_splitting_pair = False
        # range_expansions = 0
        # max_expansions = 20
        # current_ranges = list(feature_ranges)
        min_val=max_val=0

        print ("===========iteration ",iteration)

        while not (min_val < 0.0 < max_val):
       
            # if iteration < 100:
            f1, f2 = sample_random_feature_pair(binary_feature_ranges,continious_feature_ranges, binary_features, cieling=500)
            # else:
            #     f1, f2 = optimize_feature_pair(poly, current_ranges, binary_features)
            print ("generated pair:")
            print (f1,f2)
            # if len(all_pairs) == 0:
            #     all_pairs.append((f1, f2))
            #     pref = assign_synth_pref((f1, f2), feature_rewards)
            #     preferences.append(pref)
            #     break
            # Try both optimization and random sampling
            # try:
            #     # First try gradient descent optimization
            #     f1, f2 = optimize_feature_pair(poly, current_ranges, binary_features)
            # except Exception as e:
            #     print(f"Optimization failed, falling back to random sampling: {e}")
            #     # Fall back to random sampling if optimization fails
            #     f1, f2 = sample_random_feature_pair(current_ranges, binary_features)
            
            direction = f1 - f2
            min_val, max_val = compute_min_and_max_dot(poly, direction)
            uncertainty_in_direction = max_val - min_val

            print (min_val,max_val)
            
            # Rest of the code remains the same
            if min_val < 0.0 < max_val:
                # Get current feasible region volume before adding new constraint
                # current_poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim=reward_dim)
                # current_volume = calculate_polyhedron_volume(current_poly)
                
                # Add new preference
                pref = assign_synth_pref((f1, f2), feature_rewards)
                if pref == 0:
                    break
                
                # Check if new constraint reduces feasible region
                # test_pairs = all_pairs + [(f1, f2)]
                # test_prefs = preferences + [pref]
                # test_poly = get_feasible_poly_with_expansion(test_pairs, test_prefs, dim=reward_dim)
                # test_volume = calculate_polyhedron_volume(test_poly)
                
                # print(f"Volume before: {current_volume:.6f}, Volume after: {test_volume:.6f}")
                # print(f"Volume reduction: {((current_volume - test_volume) / current_volume * 100):.2f}%")
                
                all_pairs.append((f1, f2))
                preferences.append(pref)

                # # Only add the preference if it reduces the feasible region
                # if len(test_vertices) < len(current_vertices):
                #     all_pairs.append((f1, f2))
                #     preferences.append(pref)
                #     print(f"\n[Iteration {iteration}] New Query Found - Reduced feasible region from {len(current_vertices)} to {len(test_vertices)} vertices")
                # else:
                #     print(f"\n[Iteration {iteration}] Skipped query - Did not reduce feasible region")

                final_poly = get_feasible_poly_with_expansion(all_pairs, preferences, reward_dim)
                
                # see if there's a feasible solution and print it out
                H = final_poly.get_inequalities()
                A_ub, b_ub = [], []
                for row in H:
                    b_i = float(row[0])
                    A_i = -np.array([float(x) for x in row[1:]], dtype=float)
                    A_ub.append(A_i)
                    b_ub.append(b_i)
                # print (len(H))
                # assert False
                # if A_ub:
                #     A_ub = np.array(A_ub, dtype=float)
                #     b_ub = np.array(b_ub, dtype=float)
                #     lp_res = linprog(c=[0] * dim, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                #     if lp_res.success:
                #         # print("  Feasible w:", lp_res.x)
                #         found_reward = True
                #         reward = lp_res.x
                #     # else:
                #         # print("no feasible w!")
                A_ub = np.array(A_ub, dtype=float)
                # print (A_ub)
                # print (A_ub.shape)
                # print ("=======")
                # print (b_ub)
                # print ("========================\n")
                b_ub = np.array(b_ub, dtype=float)
                dim_w = A_ub.shape[1]
                #TODO: we assume assume no conflicts for now
                # all_pairs, preferences = mod_check_and_remove_conflicts(all_pairs, preferences, task_description, feature_names, binary_features)
                result = linprog(c=[0]*dim_w, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))

                print ("A_ub.shape:",A_ub.shape)
                # print ("len(all_pairs):",len(all_pairs))
                if result.success:
                    feasible_w = result.x
                    # print("  A feasible weight vector for the updated region is:", feasible_w)
                    found_reward = True
                    # reward_func = result.x
                    
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
                    total_loss = 0
                    for _ in range(1000):
                        f1, f2 = sample_random_feature_pair(binary_feature_ranges, continious_feature_ranges, binary_features)
                        # Get predictions using both weight vectors
                        true_pred = assign_synth_pref((f1, f2), feature_rewards)
                        if true_pred == 0:
                            continue
                        pred_prob = assign_synth_pref((f1, f2), feasible_w_dict, return_pref_prob=True)
                        pred_prob = max(pred_prob, 1e-10)
                        if pred_prob == 1:
                            loss = -np.log(pred_prob)*(1 if true_pred == -1 else 0)
                        else:
                            loss = -(np.log(pred_prob)*(1 if true_pred == -1 else 0) + np.log(1-pred_prob)*(1 if true_pred == 1 else 0))
                        total_loss += loss
                    
                    avg_loss = total_loss / 1000
                    print(f"Avg. cross entropy loss over 1000 unf. generated samples: {avg_loss:.4f}")
                    print ("======================\n")

                else:
                    raise ValueError("  uh oh! No feasible solution found..")

            #TODO: if we decide not to use random sampling of f1/f2 we do not need 
            # if not found_splitting_pair:
            #     # expand continuous feature ranges by factor of 10
            #     range_expansions += 1
            #     new_ranges = []
            #     for (low, high), is_bin in zip(current_ranges, binary_features):
            #         if is_bin:
            #             new_ranges.append((0.0, 1.0))  # binary feature
            #         else:
            #             center = (low + high) / 2
            #             half_range = (high - low) / 2 * default_scale
            #             new_low = center - half_range
            #             new_high = center + half_range
            #             new_ranges.append((new_low, new_high))
            #     current_ranges = new_ranges
            #     attempts*=2
            #     print(f"No splitting pair found. Increasing distance_scale:",current_ranges)

        # if not found_splitting_pair:
        #     print("No additional query can further cleave the current region. Stopping.")
        #     break
        
    return all_pairs, preferences, reward_func

def compute_min_and_max_dot(poly, direction):
    """
    returns (min_val, max_val) of w^T direction subject to w in 'poly' by
    solving two linear programs using the polyhedron's H-representation (inequalities)
    """
    H = poly.get_inequalities()
    A_ub = []
    b_ub = []
    for row in H:
        b_i = float(row[0])
        A_i = -np.array([float(x) for x in row[1:]], dtype=float)
        A_ub.append(A_i)
        b_ub.append(b_i)

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)
    c = np.array(direction, dtype=float)

    # solve for max (c^T w) => min (-(c^T) w).
    res_max = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    max_val = float(c.dot(res_max.x)) if res_max.success else float('inf')     

    # solve for min (c^T w)
    res_min = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    min_val = float(c.dot(res_min.x)) if res_min.success else float('-inf')

    return min_val, max_val

def generate_inequalities(pairs, preferences, dim,scale=1.0):
    weight_bounds = [(-scale, scale) for _ in range(dim)]

    pref_matrix = []
    epsilon = 1e-5
    
    if(len(pairs) != 0):
        assert dim == len(pairs[0][0])
        for (f0, f1), pref in zip(pairs, preferences):
            delta_f = f1 - f0
            if pref == -1:
                pref_matrix.append([-epsilon] + list(-delta_f))
            elif pref == 1:
                pref_matrix.append([-epsilon] + list(delta_f))
            elif pref == 0:
                pref_matrix.extend([[0] + list(delta_f), [0] + list(-delta_f)])

    if len(weight_bounds) != dim:
        raise ValueError("weight_bounds length must match number of features")
    for i, (L, U) in enumerate(weight_bounds):
        row_lb = [-L] + [1 if j == i else 0 for j in range(dim)]
        row_ub = [U] + [-1 if j == i else 0 for j in range(dim)]
        pref_matrix.extend([row_lb, row_ub])

    #pref_matrix = A, such that Aw > 0
    return pref_matrix

def get_feasible_poly_with_expansion(all_pairs, preferences, dim=2, signs=None):
    """
    creates feasible polyhedron for the weight space by:
      1) Starting with a small bounding box [-1, 1]^dim
      2) Building a polyhedron with those bounds + existing pairwise constraints
      3) Checking if it's unbounded. If unbounded, multiply bounds by 10 and repeat.
    returns a feasible bounded cdd polyhedron or empty.
    """
    #-------COMMENTED OUT CUS IDK WHAT THIS DOES OR WHY------------
    scale = 1.0
    while True:
        weight_bounds = []
        if signs is None:
            # bounding box in each dimension as [-scale, scale]
            weight_bounds = [(-scale, scale) for _ in range(dim)]
        else:
            #TODO: not sure wha this does
            for j, s in enumerate(signs):
                if j == dim - 1:                
                    weight_bounds.append((s, s))
                else:                           
                    if s > 0:
                        weight_bounds.append((0, scale))
                    else:
                        weight_bounds.append((-scale, 0))

        # print(weight_bounds)
        # print(dim)
        assert False
        
        poly = find_full_weight_space(all_pairs, preferences, basic_bounds=weight_bounds, num_features=dim)

        # check if poly is unbounded 
        generators = poly.get_generators()
        n_rays = sum(1 for row in generators if int(row[0]) == 0)
        if n_rays > 0:
            # unbounded, so expand further
            scale *= 10.0
        else:
            # bounded (or possibly empty) -> return
            return poly
    #------------------------------------------
    
    # poly = find_full_weight_space(all_pairs, preferences, basic_bounds=None, num_features=dim)
    # return poly

if __name__ == "__main__":
    main()
