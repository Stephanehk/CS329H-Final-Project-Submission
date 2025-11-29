import numpy as np
from scipy.optimize import linprog
import pickle

from reward_learning.active_pref_learning import load_reward_ranges

# from reward_learning.active_pref_learning import generate_inequalities

env_name = "mujoco"
model_name = "o4-mini"
direct_llm_preference="True"
feasible_w = np.load(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_feasible_weights.npy")
A_ub = np.load(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_A_ub.npy")
b_ub = np.load(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_b_ub.npy")


_,_, _, feature_names, _, _ = load_reward_ranges(env_name, range_ceiling=float('inf'),horizon=1000)


print ("Feasible weights: ", feasible_w)
print ("Feature names: ", feature_names)
assert False

# feasible_w = [-100000, 1, 1, -1, -0.1, -1, 1]
feasible_w *= 1e5


# feasible_w = [-1e5,1, 1, 0, 0, 0, 0 ]

with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_preferences.pkl", 'rb') as f:
    preferences = pickle.load(f)
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_pairs.pkl", 'rb') as f:
    all_pairs = pickle.load(f)

# w_agressive = np.load("data/gt_rew_fn_data/traffic_feasible_w_aggressive_stephane.npy")
# w_balanced = np.load("data/gt_rew_fn_data/traffic_feasible_w_for_stephane.npy")

# print ("Agressive weights: ", w_agressive)
# print ("Balanced weights: ", w_balanced)
# assert False

# rew_name = "aggressive_stephane"
#rew_name = "for_stephane"
rew_name = "egaleterian_stephane"

# feasible_w = np.load(f"data/gt_rew_fn_data/traffic_feasible_w_{rew_name}.npy")
print ("Feasible weights: ", feasible_w)

# inequalities, b = generate_inequalities(all_pairs, preferences, dim=7)
# A_ub = np.array(inequalities, dtype=float)
# b_ub = np.array(b, dtype=float)

np.save(f"data/gt_rew_fn_data/traffic_feasible_w_{rew_name}.npy", feasible_w)
np.save(f"data/gt_rew_fn_data/traffic_A_ub_{rew_name}.npy", A_ub)
np.save(f"data/gt_rew_fn_data/traffic_b_ub_{rew_name}.npy", b_ub)

with open(f"data/gt_rew_fn_data/traffic_preferences_{rew_name}.pkl", 'wb') as f:
    pickle.dump(preferences, f)
with open(f"data/gt_rew_fn_data/traffic_pairs_{rew_name}.pkl", 'wb') as f:
    pickle.dump(all_pairs, f)

# with open(f"data/gt_rew_fn_data/traffic_preferences_{rew_name}.pkl", 'rb') as f:
#     preferences = pickle.load(f)
# with open(f"data/gt_rew_fn_data/traffic_pairs_{rew_name}.pkl", 'rb') as f:
#     all_pairs = pickle.load(f)

# print("A_ub shape:", A_ub.shape)
# print("b_ub shape:", b_ub.shape)
# print("Feasible weights:", feasible_w)
all_satisfied=True
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
else:
    print (f"All {len(all_pairs)} preferences satisfied by the feasible weights.")

# feasible_w = [-100000, 1, 1, -1, -0.1, -1, 1]

# feasible_w *= 10
# b_ub *= 10
# [-9.99976416e-01  4.07411174e-07  5.29150355e-06 -6.27672495e-07
#  -7.23816196e-08 -5.70674628e-07  1.66145552e-05]
# feasible_w*=10
# print ("Feasible weights: ", feasible_w)

# feasible_w= [-100000, 1, 1, -1, -0.1, -1, 1]
