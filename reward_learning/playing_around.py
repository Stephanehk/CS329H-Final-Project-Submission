import numpy as np
import pickle

env_name = "pandemic"
model_name = "o4-mini"
direct_llm_preference="True"
# personality_type = "_healthy_extreme"
personality_type = "_prevent_lockdown"
feasible_w = np.load(f"active_learning_res/{env_name}{personality_type}_{model_name}_{direct_llm_preference}_prefs_feasible_weights.npy")
A_ub = np.load(f"active_learning_res/{env_name}{personality_type}_{model_name}_{direct_llm_preference}_prefs_A_ub.npy")
b_ub = np.load(f"active_learning_res/{env_name}{personality_type}_{model_name}_{direct_llm_preference}_prefs_b_ub.npy")

print ("Feasible weights:", feasible_w)

assert False

with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_preferences.pkl", 'rb') as f:
    preferences = pickle.load(f)
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_pairs.pkl", 'rb') as f:
    all_pairs = pickle.load(f)
#change these to loads
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_guiding_principles.txt", 'r') as f:
    guiding_principles = f.read()
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_direction_pref_guidance.txt", 'r') as f:
    direction_pref_guidance = f.read()
#save dominating_features
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_dominating_features.pkl", 'rb') as f:
    dominating_features = pickle.load(f)

print("Feasible weights:", feasible_w)
print("A_ub shape:", A_ub.shape)
print("b_ub shape:", b_ub.shape)

save_name = "learned_8-28-25"

np.save(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_feasible_weights.npy", feasible_w)
np.save(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_A_ub.npy", A_ub)
np.save(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_b_ub.npy", b_ub)

with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_preferences.pkl", 'wb') as f:
    pickle.dump(preferences, f)
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_pairs.pkl", 'wb') as f:
    pickle.dump(all_pairs, f)
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_guiding_principles.txt", 'w') as f:
    f.write(guiding_principles)
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_direction_pref_guidance.txt", 'w') as f:
    f.write(direction_pref_guidance)
with open(f"active_learning_res/{env_name}_{model_name}_{direct_llm_preference}_prefs_{save_name}_dominating_features.pkl", 'wb') as f:
    pickle.dump(dominating_features, f)