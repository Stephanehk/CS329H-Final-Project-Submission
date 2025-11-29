import numpy as np
import argparse


from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from test_traffic_reward import create_traffic_reward

from reward_learning.learn_reward_weights import load_rollout_data, get_traj_features, group_trajectories
from reward_learning.active_pref_learning import generate_inequalities, compute_min_and_max_dot

if __name__ == "__main__":
    rollout_dir = "rollout_data/"

    # Example for the *traffic* environment
    # env_name = "traffic"
    parser = argparse.ArgumentParser(
        description="Train models on a chosen environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e", "--env_name",
        type=str,
        required=True,
        help="Gymnasium environment ID (e.g., 'LunarLander-v3')",
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        required=True,
        default="gpt-4o-mini",
        help="Model name (e.g., 'gpt-4o-mini')",
    )
    args = parser.parse_args()
    env_name = args.env_name
    model_name = args.model_name

    if env_name == 'traffic':
        policy_names = [
            "traffic_base_policy",
            "2025-06-24_13-51-42",
            "2025-06-17_16-14-06",
            "2025-07-10_13-33-33",
            "2025-07-09_16-57-36",
            "traiffc-uniform-policy",
        ]
        reward_fns = ["true_reward", "proxy_reward"]
        true_reward_description = "-1.0*(normalized measure of all vehicle's closeness to the target velocity) + -0.1*(normalized measure of closeness of max_over_all_vehicles(distance to closest vehicle/vehicle speed) to target headway value) + -1.0*(closeness of mean vehicle acceleration to target acceleration)"
    elif env_name == "glucose":
        policy_names = ["glucose_base_policy","2025-06-24_13-53-32","2025-05-12_14-12-46", "2025-07-09_16-56-49_checkpoint_000025", "2025-07-09_16-56-49_checkpoint_000050", "glucose-uniform-policy_"] #"glucose_base_policy", "2025-05-12_14-12-46"
        reward_fns = ["magni_rew", "expected_cost_rew"]
        true_reward_description = "a measure of risk of hyper-glycemia, computed as -10 * (3.5506 * (np.log(most recent blood gluocse measure) ** 0.8353 - 3.7932)^2 - 25*(amount of insulin just administered))"
    elif env_name == "pandemic":
        policy_names = ["pandemic_base_policy","2025-06-24_13-49-08","2025-05-05_21-29-00", "2025-07-10_11-40-34","2025-07-09_16-58-20", "pandemic-uniform-policy"]#,pandemic_base_policy "2025-05-05_21-29-00","2025-06-24_13-49-08"
        reward_fns = ["true_reward", "proxy_reward"]
        true_reward_description = "-10*(number of infections + number of critical cases + number of deaths) + -10*(penalty for raising the lockdown stage if infection rate is lower than a treshold) + -0.1*(lockdown stage) + -0.02*(|current lockdown stage - previous lockdown stage|)"

    with open(f"active_learning_res/{env_name}_{model_name}_preferences.pkl", 'rb') as f:
        preferences = pickle.load(f)
    with open(f"active_learning_res/{env_name}_{model_name}_pairs.pkl", 'rb') as f:
        all_pairs = pickle.load(f)
    
    n_trajectories_per_policy = 50
    rollout_data = load_rollout_data(env_name, rollout_dir, policy_names, n_trajectories_per_policy)

    # 2) Instantiate reward functions -----------------------------------------
    if env_name == "glucose":
        reward = create_glucose_reward()
    elif env_name == "pandemic":
        reward = create_pandemic_reward()
    elif env_name == "traffic":
        reward = create_traffic_reward()
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    reward_functions = reward._reward_fns  # noqa: SLF001 – preserve original API

    # 3) Trajectory grouping & preference data --------------------------------
    trajectories, policy2trajectories = group_trajectories(rollout_data, policy_names)

    # 4) Load inequalities used for learning reward weights --------------------------------
    reward_dim = len(reward_functions)
    inequalities, b = generate_inequalities(all_pairs, preferences, dim=reward_dim)


    transition_cache: Dict[Any, np.ndarray] = {}
    for i, traj1 in enumerate(trajectories):
        for traj2 in trajectories[i + 1 :]:
            # Ground‑truth returns
            r1 = sum(step[gt_reward_key] for step in traj1)
            r2 = sum(step[gt_reward_key] for step in traj2)

            # Feature differences
            f1, transition_cache = get_traj_features(env_name, traj1, transition_cache, reward_functions)
            f2, transition_cache = get_traj_features(env_name, traj2, transition_cache, reward_functions)

            min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
            assert min_val != float("-inf") and max_val != float("-inf")
            #we are making sure that all feasible weights within the region we have found at the very least agree on which trajectory should be preferred
            assert  np.sign(max_val) == np.sign(min_val)