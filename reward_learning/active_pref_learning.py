import numpy as np
from scipy.optimize import linprog
import random
import json
import sys
import os
import re
import torch
from pathlib import Path
import itertools
import openai
from openai import OpenAI
import pickle
from secret_keys import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

import os, json, re
from typing import List, Tuple, Dict, Any

# from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from test_traffic_reward import create_traffic_reward
from test_mujoco_ant_reward import create_mujoco_ant_reward
from reward_learning.evaluate_reward_weights import evaluate_reward_weights
from reward_learning.elicit_llm_prefs import elicit_direction_preferences, elicit_LLM_pref, elicit_guiding_principles, get_human_pref#assign_LLM_pref,
from reward_learning.learn_reward_weights import load_rollout_data, group_trajectories, get_traj_features
from reward_learning.elicit_priors import find_dominating_features, add_dom_feat_prefs, sample_random_feature_pair, get_informative_single_feat_pair, llm_sample_feature_pair, get_informative_tradeoff_feat_pair
from reward_learning.active_learning_utils import compute_min_and_max_dot
from utils.stakeholder_type_prompts import personaility_prompts


def save_learning_results(
    env_name: str,
    model_name: str,
    direct_llm_preference: bool,
    extra_details: str,
    feasible_w: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    preferences: list,
    all_pairs: list,
    should_elicit_guiding_principles: bool = False,
    guiding_principles: str = None,
    direction_pref_guidance: str = None,
    direction_prefs: dict = None,
    pref_justifications: list = None,
) -> None:
    """
    Save all learning results to files.
    
    Args:
        env_name: Name of the environment
        model_name: Name of the model used
        direct_llm_preference: Whether direct LLM preferences were used
        extra_details: Additional details to append to filenames
        feasible_w: Feasible weight vector
        A_ub: Upper bound constraint matrix
        b_ub: Upper bound constraint vector
        preferences: List of preference labels
        all_pairs: List of feature pairs
        should_elicit_guiding_principles: Whether guiding principles were elicited
        guiding_principles: Text of guiding principles (required if should_elicit_guiding_principles is True)
        direction_pref_guidance: Text of direction preference guidance (required if should_elicit_guiding_principles is True)
        direction_prefs: Dictionary of direction preferences/dominating features (required if should_elicit_guiding_principles is True)
        pref_justifications: List of preference justifications
    """
    base_path = f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}"
    
    # Save numpy arrays
    np.save(f"{base_path}_prefs_feasible_weights.npy", feasible_w)
    np.save(f"{base_path}_prefs_A_ub.npy", A_ub)
    np.save(f"{base_path}_prefs_b_ub.npy", b_ub)
    
    # Save pickle files
    with open(f"{base_path}_preferences.pkl", 'wb') as f:
        pickle.dump(preferences, f)
    with open(f"{base_path}_pairs.pkl", 'wb') as f:
        pickle.dump(all_pairs, f)
    with open(f"{base_path}_pref_justifications.pkl", 'wb') as f:
        pickle.dump(pref_justifications, f)
    
    # Save guiding principles if applicable
    if should_elicit_guiding_principles:
        if guiding_principles is None or direction_pref_guidance is None or direction_prefs is None:
            raise ValueError("guiding_principles, direction_pref_guidance, and direction_prefs must be provided when should_elicit_guiding_principles is True")
        
        with open(f"{base_path}_guiding_principles.txt", 'w') as f:
            f.write(guiding_principles)
        with open(f"{base_path}_direction_pref_guidance.txt", 'w') as f:
            f.write(direction_pref_guidance)
        with open(f"{base_path}_dominating_features.pkl", 'wb') as f:
            pickle.dump(direction_prefs, f)


def load_json_to_feature_lists(env_name, feature_names,is_binary_feature, offset, use_no_convo_baseline=False):
    """
    Load a JSON file and collect lists of feature values for each key.

    Args:
        feature_names (list): List of feature names (not directly used here, 
                              but kept as an argument for flexibility).
        use_no_convo_baseline (bool): If True, load from generated_objectives_no_convo_baseline/

    Returns:
        list: A list of lists, where each sublist contains all values for a given key.
    """
    base_dir = "generated_objectives_no_convo_baseline" if use_no_convo_baseline else "generated_objectives"
    with open(f"{base_dir}/{env_name}_categorical_usage.json", "r") as f:
        data = json.load(f)

    all_key_lists = []
    categorical_feature_names = []
    for key, values in data.items():
        print (key, values)
        value_is = []
        for v in values:
            if v not in feature_names:
                raise ValueError(f"Value '{v}' from JSON not found in feature_names.")
            if not is_binary_feature[feature_names.index(v)]:
                continue
            value_is.append(feature_names.index(v)-offset)
            categorical_feature_names.append(v)
        if len(value_is)>0:
            all_key_lists.append(value_is)

    # print ("Categorical feature indices:", all_key_lists)
    return all_key_lists, categorical_feature_names


def load_reward_ranges(env_name: str,
                       range_ceiling: float,
                       horizon: int,
                       use_no_convo_baseline: bool = False
) -> Tuple[
        List[List[float]],                # binary_ranges
        List[Tuple[float, float]],        # continuous_ranges
        List[bool],                       # binary_flags
        List[str],                        # feature_names
        Dict[str, float]                  # feature_rewards
]:
    """
    Read generated_objectives/{env_name}_reward_ranges.json or 
    generated_objectives_no_convo_baseline/{env_name}_reward_ranges.json and return:
        binary_ranges, continuous_ranges, binary_flags, feature_names, feature_rewards
    Continuous features come first in feature_names; binary/discrete come last.
    """

    # --------------------------------------------------------------------- #
    # 1.  Load the JSON file
    # --------------------------------------------------------------------- #
    base_dir = "generated_objectives_no_convo_baseline" if use_no_convo_baseline else "generated_objectives"
    ranges_file = f"{base_dir}/{env_name}_reward_ranges.json"
    if not os.path.exists(ranges_file):
        raise FileNotFoundError(f"Reward-ranges file not found: {ranges_file}")

    with open(ranges_file, "r", encoding="utf-8") as f:
        reward_ranges = json.load(f)

    # --------------------------------------------------------------------- #
    # 2.  Helpers
    # --------------------------------------------------------------------- #
    _INF_PAT = re.compile(r"(?:-)?(?:∞|inf)", re.IGNORECASE)

    def _is_inf(tok: str) -> bool:
        """Return True if token contains 'inf' or the infinity symbol."""
        return bool(_INF_PAT.search(tok))

    def _parse_bound(tok: str, is_lower: bool) -> float:
        """
        Convert a token into a float, substituting ±range_ceiling for ±∞.
        `tok` has already been stripped of whitespace and quotes.
        """
        if _is_inf(tok):
            return -range_ceiling if tok.strip().startswith("-") or is_lower else range_ceiling

        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", tok)
        if m:
            return float(m.group())
        # Fallback if no number is found
        return -range_ceiling if is_lower else range_ceiling

    # Regex to capture the first {...} or [../(..)..] substring
    _RANGE_RE = re.compile(r"\{[^}]+\}|[\[\(][^\]\)]+[\]\)]")

    # --------------------------------------------------------------------- #
    # 3.  Parse each feature
    # --------------------------------------------------------------------- #
    cont_names, disc_names   = [], []
    cont_ranges, bin_ranges  = [], []

    for feat, desc in reward_ranges.items():

        m = _RANGE_RE.search(desc)
        if not m:
            raise ValueError(f"Cannot locate a bracket/brace range in: {desc}")

        rng_str = m.group()

        # --------------- Discrete / binary features --------------------- #
        if rng_str.startswith("{"):
            inside = rng_str[1:-1]                # strip braces
            tokens = [v.strip().strip("'\"") for v in inside.split(",") if v.strip()]

            # Convert tokens → numeric values (skip infinities for totals)
            vals = []
            for t in tokens:
                if _is_inf(t):
                    vals.append(range_ceiling if not t.strip().startswith("-") else -range_ceiling)
                else:
                    #ingore LLM filled values like {1,2,3, ...}; this is a potential limitation but IMO I really don't think it will effect things
                    if t == "...":
                        continue
                    vals.append(float(t))

            totals = {val * k for val in vals for k in range(horizon+1)}
            bin_ranges.append(sorted(totals))
            disc_names.append(feat)

        # --------------- Continuous features --------------------------- #
        else:
            inner = rng_str[1:-1]                 # drop leading '[' / '(' and trailing ']' / ')'
            lower_tok, upper_tok = [s.strip().strip("'\"") for s in inner.split(",", 1)]

            lo = _parse_bound(lower_tok, is_lower=True)   * horizon
            hi = _parse_bound(upper_tok, is_lower=False)   * horizon

            cont_ranges.append((lo, hi))
            cont_names.append(feat)

    # --------------------------------------------------------------------- #
    # 4.  Assemble outputs
    # --------------------------------------------------------------------- #
    feature_names  = cont_names + disc_names
    binary_flags   = [False] * len(cont_names) + [True] * len(disc_names)
    feature_rewards = {name: 0.0 for name in feature_names}

    return bin_ranges, cont_ranges, binary_flags, feature_names, feature_rewards, reward_ranges

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

# def load_reward_ranges(env_name,range_ceiling,horizon):
#     """
#     Load reward ranges and feature names for a given environment.
#     Returns feature_ranges, binary_features, feature_names, and feature_rewards
#     with binary features placed contiguously at the end of each list.
#     """
#     # Load reward ranges
#     ranges_file = f"generated_objectives/{env_name}_reward_ranges.json"
#     if not os.path.exists(ranges_file):
#         raise ValueError(f"Reward ranges file not found: {ranges_file}")
    
#     with open(ranges_file, 'r', encoding='utf-8') as f:
#         reward_ranges = json.load(f)
    
#     # Process ranges and determine binary features
#     feature_ranges = []
#     binary_features = []
#     feature_names = list(reward_ranges.keys())
    
#     # Initialize feature rewards dictionary with zeros
#     feature_rewards = {name: 0.0 for name in feature_names}
    
#     # First pass: collect all features and determine which are binary
#     continuous_features = []
#     binary_features_list = []
#     continuous_ranges = []
#     binary_ranges = []
    
#     for feature_name in feature_names:
#         range_str = reward_ranges[feature_name]
#         # Check if range is discrete (denoted by curly braces)
#         if '{' in range_str:
#             # Extract values from {val1, val2} format
#             values = range_str.split('{')[1].split('}')[0].split(',')
#             min_val = float(values[0].strip())
#             max_val = float(values[1].strip())
#             binary_features_list.append(True)
#             binary_ranges.append(list(set([min_val*i for i in range(horizon)]+[max_val*i for i in range(horizon)])))
#             binary_features.append(feature_name)
#         else:
#             # Extract values from (min, max) or [min, max] format
#             range_str = range_str.split('Range: ')[1].strip("'")
#             if "(-inf" in range_str:
#                 min_val = -range_ceiling
#             else:
#                 match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', range_str.split(',')[0])
#                 min_val = float(match.group())
            
#             if "inf)" in range_str:
#                 max_val = range_ceiling
#             else:
#                 match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', range_str.split(',')[1])
#                 max_val = float(match.group())
            
#             continuous_features.append(feature_name)
#             continuous_ranges.append((min_val*horizon, max_val*horizon))
#             binary_features_list.append(False)
    
#     # Combine continuous and binary features in order
#     feature_names = continuous_features + binary_features
#     # feature_ranges = continuous_ranges + binary_ranges
#     binary_features = [False] * len(continuous_features) + [True] * len(binary_features)
    
#     # Reorder feature_rewards to match the new order
#     feature_rewards = {name: feature_rewards[name] for name in feature_names}
    
#     return binary_ranges,continuous_ranges , binary_features, feature_names, feature_rewards



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

def get_real_traj_feature_sums(reward_functions, env_name, policy_names, feature_names):
    rf_names = [rf.__class__.__name__ for rf in reward_functions]
    rollout_data = load_rollout_data(env_name, rollout_dir="rollout_data_pan_et_al_rew_fns/", policy_names=policy_names, n_trajectories_per_policy=50) #should be 50
    trajectories, _ = group_trajectories(rollout_data, policy_names)
    transition_cache: Dict[Any, np.ndarray] = {}
    traj_feature_sums = []
    for traj in trajectories:
        f, transition_cache = get_traj_features(env_name, traj, transition_cache, reward_functions)
        traj_feature_sums.append(f)

    #the featurs in traj_feature_sums are in the same order as rf_names but we want them to be in the same order as feature_names
    rf_name_to_idx = {name: i for i, name in enumerate(rf_names)}
    feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}
    traj_feature_sums = np.array(traj_feature_sums)
    reordered_traj_feature_sums = np.zeros_like(traj_feature_sums)
    for fname in feature_names:
        if fname not in rf_name_to_idx:
            raise ValueError(f"Feature name '{fname}' not found in reward function names.")
        reordered_traj_feature_sums[:, feature_name_to_idx[fname]] = traj_feature_sums[:, rf_name_to_idx[fname]]
    return reordered_traj_feature_sums

def main():
    np.random.seed(2)
    # Parse command line arguments
    if len(sys.argv) != 9:
        print("Usage: python active_pref_learning.py <environment_name> <pref_gen> <ll> <true/false> <use_no_convo_baseline>")
        print("Example: python active_pref_learning.py pandemic synth false")
        print("pref_gen options: synth (synthetic) or llm (LLM-based)")
        sys.exit(1)
    
    env_name = sys.argv[1]
    pref_gen = sys.argv[2]
    model_name = sys.argv[3]
    resume = sys.argv[4]
    direct_llm_preference = sys.argv[5]
    should_elicit_guiding_principles = sys.argv[6]
    stakeholder_type = sys.argv[7]
    use_no_convo_baseline = sys.argv[8]

    use_real_trajectories = False

    extra_details = ""#"_2_samp_funcs"
    personaility_prompt = None
    if stakeholder_type != "default":
        extra_details += f"_{stakeholder_type}"
        personaility_prompt = personaility_prompts[env_name][stakeholder_type]

    if use_no_convo_baseline:
        extra_details += f"_no_convo_baseline"


    #python3 -m reward_learning.active_pref_learning pandemic llm o4-mini false true true default false
    #python3 -m reward_learning.active_pref_learning traffic llm o4-mini false true true default false
    #python3 -m reward_learning.active_pref_learning mujoco llm o4-mini false true true default false
    #python3 -m reward_learning.active_pref_learning mujoco_backflip llm o4-mini false true true default 
    
    assert resume == "true" or resume == "false"
    if resume =="true":
        resume = True
    else:
        resume = False

    assert direct_llm_preference == "true" or direct_llm_preference=="false"
    direct_llm_preference = True if direct_llm_preference=="true" else False
    
    should_elicit_guiding_principles = True if should_elicit_guiding_principles=="true" else False
    use_no_convo_baseline = True if use_no_convo_baseline=="true" else False

    if pref_gen not in ["synth", "llm", "human"]:
        print("Error: pref_gen must be either 'synth' or 'llm' or 'human'")
        sys.exit(1)
    
    if env_name == "pandemic":
        horizon = 192
        # n_trajectories_per_policy=100
        generated_reward_feats = create_pandemic_reward()
    elif env_name == "glucose":
        horizon = 5760
        generated_reward_feats = create_glucose_reward()
        # n_trajectories_per_policy=100
    elif env_name == "traffic":
        horizon = 300
        generated_reward_feats = create_traffic_reward()
        policy_names = ["traffic_base_policy","2025-06-24_13-51-42","2025-06-17_16-14-06","2025-07-10_13-33-33","2025-07-09_16-57-36"]
    elif env_name == "mujoco":
        horizon = 1000
        generated_reward_feats = create_mujoco_ant_reward()
    elif env_name == "mujoco_backflip":
        horizon = 1000
        generated_reward_feats = create_mujoco_ant_reward(env_type="mujoco_backflip")

    reward_functions = generated_reward_feats._reward_fns
    
    # model_name="gpt-4o-mini"
    # Load reward ranges and feature information
    #TODO: need to actually figure out what range_ceiling is 
    binary_feature_ranges,continious_feature_ranges, binary_features, feature_names, feature_rewards, reward_ranges_raw = load_reward_ranges(env_name, range_ceiling=float('inf'),horizon=horizon, use_no_convo_baseline=use_no_convo_baseline)
    categorical_feature_is,categorical_feature_names = load_json_to_feature_lists(env_name, feature_names,binary_features, offset=len(continious_feature_ranges), use_no_convo_baseline=use_no_convo_baseline)

    #uncomment to use real
    if use_real_trajectories:
        traj_feature_sums = get_real_traj_feature_sums(reward_functions, env_name, policy_names, feature_names)
   
    # print (categorical_feature_is)
    # assert False
    # assert False

    for k in feature_rewards.keys():
        feature_rewards[k] = np.random.uniform(-10, 10)

    if env_name == "pandemic":
        # Load weights from saved JSON file
        # policy_names_str = "pandemic_base_policy_2025-05-05_21-29-00"  # This should match the policy names used in learn_pandemic_reward_weights.py
        # weights_path = Path("reward_learning_data") / f"pandemic_weights_{policy_names_str}.json"
        
        # if not weights_path.exists():
        #     raise ValueError(f"Weights file not found: {weights_path}")
            
        # with open(weights_path, 'r') as f:
        #     weights_dict = json.load(f)
            
        # for k in feature_rewards.keys():
        #     feature_rewards[k] = weights_dict[k]
        
        # feature_rewards = {k: v for k, v in weights_dict.items() if k not in ['r2_score', 'kendall_tau', 'kendall_tau_p_value']}
        task_description = "Choosing the level of lockdown restrictions placed on the population during the COVID-19 pandemic."

    elif env_name == "glucose":
        #glucose_glucose_base_policy_2025-05-12_14-12-46_rollout_data.pkl
        task_description = "Choosing a protocol for administering insulin to a patient with Type 1 diabetes."
    elif env_name == "traffic":
        task_description = "Choosing the accelerations for each vehicle in a fleet of autonomous vehicles on an on-ramp attempting to merge into traffic on a highway."
    elif env_name == "mujoco":
        task_description = "Choosing the torques applied at the hinge joints of the ant robot in the Mujoco Ant environment so it walks forward."
    elif env_name == "mujoco_backflip":
        task_description = "Choosing the torques applied at the hinge joints of the ant robot in the Mujoco Ant environment so it does a backflip."
    else:
        raise ValueError("Other Environments Are Not Implemented Yet")
    reward_dim = len(feature_names)
    # print ("reward_dim:", reward_dim)
    # assert False

    if use_no_convo_baseline:
        with open(f"generated_objectives_no_convo_baseline/{env_name}_objective_descriptions.json", 'r') as f:
            objective_descriptions = json.load(f)

        with open(f"generated_objectives_no_convo_baseline/{env_name}_generated_objectives.py", 'r', encoding="utf-8") as f:
            generated_objectives = f.read().strip()
    else:
        with open(f"generated_objectives/{env_name}_objective_descriptions.json", 'r') as f:
            objective_descriptions = json.load(f)

        with open(f"generated_objectives/{env_name}_generated_objectives.py", 'r', encoding="utf-8") as f:
            generated_objectives = f.read().strip()

    if pref_gen == "llm":
        client = OpenAI(api_key=openai.api_key)


    # print ("==================")
    print (feature_names)
    # print (feature_rewards)
    # print ("binary_features:", binary_features)
    # print ("binary_feature_ranges:", binary_feature_ranges)
    # print ("categorical_feature_is:", categorical_feature_is)
    # print ("continious_feature_ranges:", continious_feature_ranges)
    # print ("==================")
    # assert False

    guiding_principles= None
    if should_elicit_guiding_principles:
        if resume:
            with open(f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}_guiding_principles.txt", 'r') as f:
                guiding_principles = f.read()
            with open(f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}_direction_pref_guidance.txt", 'r') as f:
                direction_pref_guidance = f.read()
        else:
            guiding_principles = elicit_guiding_principles(personaility_prompt, task_description, objective_descriptions, feature_names, generated_objectives, client, model_name="gpt-4o")
            direction_pref_guidance = elicit_direction_preferences(guiding_principles, task_description, objective_descriptions, feature_names, generated_objectives, client, model_name="gpt-4o")
        print ("Guiding Principles:")
        print (guiding_principles)
        print ("Direction Preference Guidance:")
        print (direction_pref_guidance)
        print ("==================")

        print ("Objective Descriptions:")
        print (objective_descriptions)
        print ("==================")
        print ("Generated Objectives:")
        print (generated_objectives)
        print ("==================")

    if resume:
       #open dominating_features from file
        with open(f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}_dominating_features.pkl", 'rb') as f:
            direction_prefs = pickle.load(f)
            dominating_features = direction_prefs.keys()
    else:
        dominating_features, direction_prefs = find_dominating_features(task_description, objective_descriptions, feature_names, generated_objectives, client, model_name=model_name)
    print ("Dominating Features:", dominating_features)
    print ("==================")

    # Initialize other variables
    stopping_num = 10000
    n_pairs2sampler_per_iter = 1000
    if resume:
        n_pairs2sampler_per_iter *= 10
    # n_init_pairs = 0
    total_cost = 0
    

    preferences = []
    all_pairs = []
    pref_justifications = []

    #-------------------------------- Load Stephane Data for Traffic -----------------#
    # print ("LOADING IN STEPHANE DATA FOR TRAFFIC---COMMENT OUT LATER")
    # all_pairs = np.load("data/stephane_traffic_hand_designed_init_traj_pairs.npy", allow_pickle=True).tolist()
    # all_pairs = [(np.array(f[0]), np.array(f[1])) for f in all_pairs]
    # preferences = np.load("data/stephane_traffic_hand_designed_init_traj_prefs.npy", allow_pickle=True).tolist()


    # with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_preferences.pkl", 'rb') as f:
    #     preferences = pickle.load(f)
    # with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_pairs.pkl", 'rb') as f:
    #     all_pairs = pickle.load(f)

    # rew_name =  "egaleterian_stephane"
    # with open(f"data/gt_rew_fn_data/traffic_preferences_{rew_name}.pkl", 'rb') as f:
    #     preferences = pickle.load(f)
    # with open(f"data/gt_rew_fn_data/traffic_pairs_{rew_name}.pkl", 'rb') as f:
    #     all_pairs = pickle.load(f)

    # with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_preferences.pkl", 'rb') as f:
    #     preferences = pickle.load(f)
    # with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_pairs.pkl", 'rb') as f:
    #     all_pairs = pickle.load(f)

    #----------------------------------------------------------------------#

    if resume:
        # Load previous state
        with open(f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}_preferences.pkl", 'rb') as f:
            preferences = pickle.load(f)
        with open(f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}_pairs.pkl", 'rb') as f:
            all_pairs = pickle.load(f)

    # print ("***n_init_pairs:", n_init_pairs)
    # for _ in range(n_init_pairs):
    #     f1, f2 = sample_random_feature_pair(binary_feature_ranges,continious_feature_ranges, binary_features, cieling=500)
    #     pref = assign_synth_pref((f1, f2), feature_rewards)
    #     if pref == 0:
    #         continue
    #     all_pairs.append((f1, f2))
    #     preferences.append(pref)

    inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
    n_failed_2find_splitting_pairs = 0

    #TOGGLE THIS TO ENABLE/DISABLE FINDING FEATURE DIRECTIONS
    in_find_feat_directions = True
    in_find_feat_tradeoffs = False

    if len(dominating_features) > 0 and not resume:
        print ("Adding dominating feature preferences to inequalities")
        all_pairs, preferences = add_dom_feat_prefs(dominating_features, direction_prefs, feature_names, binary_features, binary_feature_ranges, continious_feature_ranges,categorical_feature_is, all_pairs, preferences, reward_dim)
        pref_justifications.extend([None] * len(all_pairs))
    last_llm_sample_time = 0
    for iteration in range(stopping_num):
        if n_failed_2find_splitting_pairs > 10:
            return
        
        # construct the current feasible weight-space polyhedron
        # poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim=reward_dim)
    
        best_pair = None
        highest_uncertainty = float("-inf")
        found_splitting_pair= False
        if use_real_trajectories:
            #loop through all len(traj_feature_sums) choose 2 pairs and find the one with the highest uncertainty
            for i in range(len(traj_feature_sums)):
                for j in range(i+1, len(traj_feature_sums)):
                    if found_splitting_pair:
                        break
                    f1 = traj_feature_sums[i]
                    f2 = traj_feature_sums[j]
                    min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
                    uncertainty_in_direction = max_val - min_val
                    if uncertainty_in_direction > highest_uncertainty and np.sign(max_val) != np.sign(min_val):
                        highest_uncertainty = uncertainty_in_direction
                        best_pair = (f1, f2)

                        # if i > len(traj_feature_sums)/2:
                        #     found_splitting_pair = True

                        # if (f1[1] - f1[2] > 1.5*(f2[1] - f2[2]) and f1[3] > f2[3]) or (f2[1] - f2[2] > 1.5*(f1[1] - f1[2]) and f2[3] > f1[3]):
                        #     #this is an easy pair, so we can stop looking for more pairs
                        #     found_splitting_pair = True
                        #     break
                        
                        # if (f1[1] > f2[1] and f1[2] < 0.5*f2[2] and f1[3] > 2*f2[3]) or (f2[1] > f1[1] and f2[2] < 0.5*f1[2] and f2[3] > 2*f1[3]):
                        #     #this is an easy pair, so we can stop looking for more pairs
                        #     found_splitting_pair = True
                        #     break

        if (in_find_feat_directions and not resume) or (resume and len(all_pairs) < 50):
            # pairs2add, prefs2add = get_informative_single_feat_pair(inequalities, b, dominating_features, feature_names,binary_features, binary_feature_ranges,continious_feature_ranges, n_samps = 50)
            # all_pairs.extend(pairs2add)
            # preferences.extend(prefs2add)
            # inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
            best_pair, highest_uncertainty = get_informative_single_feat_pair(inequalities, b, dominating_features, feature_names,binary_features, binary_feature_ranges,continious_feature_ranges)
            if highest_uncertainty < 0.1 or iteration >=50:
                in_find_feat_directions = False
                in_find_feat_tradeoffs = True

        if (in_find_feat_tradeoffs and not resume) or (resume and len(all_pairs) < 100 and len(preferences) >= 50):
            best_pair, highest_uncertainty = get_informative_tradeoff_feat_pair(inequalities, b, dominating_features, feature_names,binary_features, binary_feature_ranges,continious_feature_ranges)
            if highest_uncertainty < 0.1 or iteration >=100:
                print ("Couldn't find a splitting pair among tradeoffs, switching to random sampling")
                in_find_feat_tradeoffs = False
                in_find_feat_directions = False

        if best_pair is None:
            if use_real_trajectories:
                print ("Couldn't find a splitting pair among real trajectories, switching to random sampling")
                use_real_trajectories = False

            # if len(preferences) >= 320:
            #if we have gone 10 iterations since last LLM sample, stop sampling; the llm is probably not finding any good pairs
            if iteration-last_llm_sample_time > 20:
                print ("No LLM samples found in last 20 iterations, switching to exclusively random sampling")
            if iteration % 3 == 0 and iteration-last_llm_sample_time <= 20:
                pairs = llm_sample_feature_pair(feature_names,reward_ranges_raw, horizon, client)
                llm_pairs_added = 0
                unstaged_pairs = []
                unstaged_prefs = []
                for f1, f2 in pairs:
                    try:
                        min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
                    except ValueError:
                        print ("ValueError in computing min and max dot, likely due to formatting issues of pair, skipping this pair:", f1, f2)
                        continue
                    uncertainty_in_direction = max_val - min_val
                
                    # uncertainty_in_direction = max_val - min_val
                    # assert uncertainty_in_direction >= 0
                    
                    if np.sign(max_val) != np.sign(min_val):
                        last_llm_sample_time = iteration
                        pref, total_cost, pref_justification = elicit_LLM_pref((f1, f2), objective_descriptions,dominating_features, feature_names, task_description,client,total_cost,model_name=model_name, direct_llm_preference=direct_llm_preference, categorical_features=categorical_feature_names, guiding_principles=guiding_principles, direction_pref_guidance=direction_pref_guidance)
                        if pref == -2: #"can't tell"
                            continue
                        all_pairs.append((f1, f2))
                        preferences.append(pref)
                        pref_justifications.append(pref_justification)

                        unstaged_pairs.append((f1, f2))
                        unstaged_prefs.append(pref)
                        llm_pairs_added +=1

                        #-----lets check if the LP is still feasible after adding this pair-----
                        inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
                        A_ub = np.array(inequalities, dtype=float) #Negative sign coverts this to the form where we wish to finx w that satisfies A'w < b  instead of Aw>b
                        b_ub = np.array(b, dtype=float)
                        dim_w = A_ub.shape[1]
                        result = linprog(c=[0]*dim_w, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                        assert result.success, "LP became infeasible after adding LLM suggested pair"
                        print ("LLM suggested pair added with preference and LP is still feasible")

                        with open(f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}_unstaged_prefs.pkl", 'wb') as f:
                            pickle.dump(unstaged_prefs, f)
                        with open(f"active_learning_res/{env_name}{extra_details}_{model_name}_{direct_llm_preference}_unstaged_pairs.pkl", 'wb') as f:
                            pickle.dump(unstaged_pairs, f)

                print ("Added", llm_pairs_added, "pairs from LLM suggestions")

                       
            # assert False 
            for _ in range(n_pairs2sampler_per_iter):

                #this means things are taking a while/random sampling is not finding informative pairs
                
                f1, f2 = sample_random_feature_pair(feature_names, binary_feature_ranges,continious_feature_ranges, binary_features,categorical_feature_is,dominating_features, cieling=500)
                
                min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
                uncertainty_in_direction = max_val - min_val
            
                # uncertainty_in_direction = max_val - min_val
                # assert uncertainty_in_direction >= 0
                if uncertainty_in_direction > highest_uncertainty and np.sign(max_val) != np.sign(min_val):
                    highest_uncertainty = uncertainty_in_direction
                    best_pair = (f1, f2)
        if best_pair is not None:
            # print ("-------SAMPLED--------")
            # print (best_pair)
            # print ("----------------------")

            n_failed_2find_splitting_pairs = 0
            if highest_uncertainty == 0:
                raise ValueError("Couldn't find a point where w might disagree on the preference")
            f1, f2 = best_pair
            pref_justification = None
            if pref_gen == "human":
                pref = get_human_pref((f1, f2), objective_descriptions, feature_names, task_description)
            elif pref_gen == "synth":
                pref = assign_synth_pref((f1, f2), feature_rewards)
            else:  # pref_gen == "llm"
                pref, total_cost, pref_justification = elicit_LLM_pref((f1, f2), objective_descriptions,dominating_features, feature_names, task_description,client,total_cost,model_name=model_name, direct_llm_preference=direct_llm_preference, categorical_features=categorical_feature_names, guiding_principles=guiding_principles, direction_pref_guidance=direction_pref_guidance)
            if pref == -2: #"can't tell"
                continue
            all_pairs.append((f1, f2))
            preferences.append(pref)
            pref_justifications.append(pref_justification)
        else:
            print ("Couldn't find a splitting pair, increasing n_pairs2sampler_per_iter")
            n_pairs2sampler_per_iter = int (n_pairs2sampler_per_iter*1.1)
            n_failed_2find_splitting_pairs +=1

        inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)
        A_ub = np.array(inequalities, dtype=float) #Negative sign coverts this to the form where we wish to finx w that satisfies A'w < b  instead of Aw>b
        b_ub = np.array(b, dtype=float)

        dim_w = A_ub.shape[1]
        #TODO: we assume assume no conflicts for now
        # all_pairs, preferences = mod_check_and_remove_conflicts(all_pairs, preferences, task_description, feature_names, binary_features)
        result = linprog(c=[0]*dim_w, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))

        print ("A_ub.shape:",A_ub.shape)
        print ("highest uncertainty:",highest_uncertainty)
        print ("len(all_pairs):", len(all_pairs))
        print ("len(preferences):", len(preferences))

        # # print ("Uncertainty after update:")
        # min_val, max_val = compute_min_and_max_dot(inequalities, b, best_pair[1]-best_pair[0])
        # uncertainty_in_direction = max_val - min_val
        # print ("Uncertainty in direction after update:", uncertainty_in_direction)
        # print (np.sign(max_val), np.sign(min_val))
        # print ("-----------------------")

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

        if result.success:
            feasible_w = result.x
            l1 = np.sum(np.abs(feasible_w))
            if l1 == 0:                         # degenerate, all-zero solution
                raise RuntimeError("LP returned w = 0; check constraints.")
            feasible_w /= l1
            if pref_gen == "human":   
                print (f"Feasible w: {feasible_w}")
            
            # Save all results using the new function
            save_learning_results(
                env_name=env_name,
                model_name=model_name,
                direct_llm_preference=direct_llm_preference,
                extra_details=extra_details,
                feasible_w=feasible_w,
                A_ub=A_ub,
                b_ub=b_ub,
                preferences=preferences,
                all_pairs=all_pairs,
                should_elicit_guiding_principles=should_elicit_guiding_principles,
                guiding_principles=guiding_principles,
                direction_pref_guidance=direction_pref_guidance,
                direction_prefs=direction_prefs,
                pref_justifications=pref_justifications
            )
        

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
                if pref == -1 and dot1 >= dot2:
                    print(f"Warning: Preference not satisfied for pair {f1}, {f2}")
                    print(f"Expected f1 < f2 but got {dot1} >= {dot2}")
                    all_satisfied = False
                elif pref == 1 and dot1 <= dot2:
                    print(f"Warning: Preference not satisfied for pair {f1}, {f2}")
                    print(f"Expected f1 > f2 but got {dot1} <= {dot2}")
                    all_satisfied = False
            
            if not all_satisfied:
                raise ValueError("Warning: Found weights do not satisfy all preferences!")
            
            # Construct dictionary mapping feature names to feasible weights
            feasible_w_dict = {name: weight for name, weight in zip(feature_names, feasible_w)}
            print ("feasible weight vector:", feasible_w_dict)
            # print ("true weight vector:", feature_rewards)
            # evaluate_reward_weights(env_name, "rollout_data/", eval_policy_names, feature_rewards, feasible_w_dict, n_trajectories_per_policy=n_trajectories_per_policy)
            
            # Compute cross entropy loss over 1000 random samples
            # n_correct = 0
            # for _ in range(1000):
            #     f1, f2 = sample_random_feature_pair(binary_feature_ranges, continious_feature_ranges, binary_features)
            #     # Get predictions using both weight vectors
            #     true_pred = assign_synth_pref((f1, f2), feature_rewards)
            #     if true_pred == 0:
            #         continue
            #     # pred_prob = assign_synth_pref((f1, f2), feasible_w_dict, return_pref_prob=True)
            #     # pred_prob = max(pred_prob, 1e-10)
            #     # if pred_prob == 1:
            #     #     loss = -np.log(pred_prob)*(1 if true_pred == -1 else 0)
            #     # else:
            #     #     loss = -(np.log(pred_prob)*(1 if true_pred == -1 else 0) + np.log(1-pred_prob)*(1 if true_pred == 1 else 0))
            #     # total_loss += loss
                
            #     n_correct += assign_synth_pref((f1, f2), feasible_w_dict) == true_pred
            
            # print(f"Accuracy: {(n_correct/1000):.4f}")
            print ("======================\n")

        elif not result.success:
            save_learning_results(
                env_name=env_name,
                model_name=model_name,
                direct_llm_preference=direct_llm_preference,
                extra_details=extra_details + "_no_feasible_solution",
                feasible_w=feasible_w,
                A_ub=A_ub,
                b_ub=b_ub,
                preferences=preferences,
                all_pairs=all_pairs,
                should_elicit_guiding_principles=should_elicit_guiding_principles,
                guiding_principles=guiding_principles,
                direction_pref_guidance=direction_pref_guidance,
                direction_prefs=direction_prefs,
                pref_justifications=pref_justifications
            )
            raise ValueError("  uh oh! No feasible solution found..")

        

def generate_inequalities(pairs, preferences, dim,scale=1.0):
    weight_bounds = [(-scale, scale) for _ in range(dim)]

    pref_matrix = []
    b = []
    epsilon = 1e-4 #turns non-strict inequality of LP to strict inequality
    
    if(len(pairs) != 0):
        assert dim == len(pairs[0][0])
        for (f0, f1), pref in zip(pairs, preferences):
            delta_f = f1 - f0
            if pref == -1:
                pref_matrix.append(list(-delta_f))
                b.append(-epsilon)
            elif pref == 1:
                pref_matrix.append(list(delta_f))
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

    return pref_matrix, b


if __name__ == "__main__":
    main()
