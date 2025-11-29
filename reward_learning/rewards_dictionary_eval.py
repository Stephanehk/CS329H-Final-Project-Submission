import json
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import argparse
from reward_learning.active_pref_learning import load_reward_ranges

from reward_learning.learn_reward_weights import load_rollout_data, group_trajectories, build_preference_dataset, evaluate_model

from scipy.stats import kendalltau

# -----------------------------------------------------------------------------
# 0) Environment‑specific reward function factories – unchanged
# -----------------------------------------------------------------------------
# from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from test_traffic_reward import create_traffic_reward
from test_mujoco_ant_reward import create_mujoco_ant_reward

# NOTE: The dependencies below are only used for typing; they do not introduce
#       heavy imports at runtime.
from utils.glucose_rollout_and_save import TrajectoryStep  # noqa: F401 – type hints
from utils.glucose_gt_rew_fns import (
    MagniGroundTruthReward,
    ExpectedCostGroundTruthReward,
)

POLICY_ALIASES = {
    0: "BC-policy",
    1: "environment-reward-opt-policy",
    2: "alt-environment-reward-opt-policy",
    3: "slightly-worse-than-BC-policy",
    4: "slightly-better-than-BC-policy",
}

def alias_policy_name(name: str, policy_names: List[str]) -> str:
    try:
        idx = policy_names.index(name)
        return POLICY_ALIASES.get(idx, name)
    except ValueError:
        return name

################################################################################
# Pretty‑printing helpers
################################################################################

def _banner(title: str, char: str = "=", length: int = 70) -> None:
    """Print a centred banner to visually separate stages."""
    pad_len = max((length - len(title) - 2) // 2, 0)
    line = char * pad_len + f" {title} " + char * pad_len
    print("\n" + line[:length])


def _kv(key: str, value: Any, indent: int = 0) -> None:
    """Consistent key–value printing."""
    print(" " * indent + f"{key:>25s}: {value}")


################################################################################
# 2) Feature extraction helpers (unchanged)
################################################################################
def extract_reward_features(
    env_name: str,
    obs: Any,
    action: Any,
    next_obs: Any,
    reward_functions: List[Any],
) -> np.ndarray:
    """Compute individual reward function values for a single transition."""

    if env_name == "glucose":
        # The glucose env stores BG values as lists; wrap in NumPy arrays for
        # faster ops and to keep key consistency.
        obs.bg = np.asarray(obs.bg)
        next_obs.bg = np.asarray(next_obs.bg)

    return np.asarray(
        [rf.calculate_reward(obs, action, next_obs) for rf in reward_functions]
    )


def get_traj_features(
    env_name: str,
    traj: List[Dict[str, Any]],
    transition_cache: Dict[Tuple[np.ndarray, ...], np.ndarray],
    reward_functions: List[Any],
) -> Tuple[np.ndarray, Dict[Tuple[np.ndarray, ...], np.ndarray]]:
    """Return *sum* of per‑transition feature vectors for the full trajectory.

    We cache transition‑level features to avoid redundant computation when the
    same (obs, action, next_obs) triple shows up across multiple trajectories.
    """

    features = []
    for step in traj:
        key = tuple(
            np.concatenate(
                (
                    step["obs"].flatten(),
                    step["action"].flatten(),
                    step["next_obs"].flatten(),
                )
            )
        )
        if key not in transition_cache:
            transition_cache[key] = extract_reward_features(
                env_name, step["obs"], step["action"], step["next_obs"], reward_functions
            )
        features.append(transition_cache[key])

    return np.sum(features, axis=0), transition_cache

################################################################################
# 5) Orchestrator – public API
################################################################################

def evaluate_reward_weights_from_preferences(
    env_name: str,
    rollout_dir: str,
    policy_names: List[str],
    gt_reward_fn: str,
    weights_dict: Dict[str, float],
    n_trajectories_per_policy: int = 10,
    no_convo_base_line: bool = False
):
    """High‑level helper that mirrors the original training function *sans* learning.

    Steps:
        1. Load rollout data (transitions).
        2. Instantiate reward functions for the chosen environment.
        3. Group transitions → trajectories.
        4. Build preference dataset (for diagnostic accuracies only).
        5. Run the evaluation routine with the supplied weights.
    """

    # _banner("START evaluation of provided weights")
    _kv("env", env_name, indent=2)
    _kv("reward fn used for eval", gt_reward_fn, indent=2)

    # 1) Load / preprocess roll‑outs ----------------------------------------
    rollout_data = load_rollout_data(env_name, rollout_dir, policy_names, n_trajectories_per_policy)

    # 2) Instantiate reward functions ----------------------------------------
    if env_name == "glucose":
        reward = create_glucose_reward()
    elif env_name == "pandemic":
        reward = create_pandemic_reward(no_convo_base_line=no_convo_base_line)
    elif env_name == "traffic":
        reward = create_traffic_reward(no_convo_base_line=no_convo_base_line)
    elif env_name == "mujoco":
        reward = create_mujoco_ant_reward()
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    reward_functions = reward._reward_fns  # noqa: SLF001 – preserve original API

    # 3) Trajectory grouping & preference data -------------------------------
    trajectories, _ = group_trajectories(rollout_data, policy_names)

    X, y, reward_diffs, policy2pairs, traj_pairs, _ = build_preference_dataset(
        env_name, trajectories, reward_functions, gt_reward_fn
    )
    _kv("Preference pairs", len(X), indent=2)

    # # 4) Run evaluation -------------------------------------------------------
    evaluate_model(env_name, trajectories, reward_functions, None, weights_dict, True, gt_reward_fn, policy_names, policy2pairs)

################################################################################
# 6) CLI – simple JSON interface
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a *given* set of reward weights (no learning).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_name", required=True, type=str, help="Environment name: glucose | pandemic | traffic")
    parser.add_argument("--weights_name", required=True, type=str, help="Environment name: glucose | pandemic | traffic")

    # parser.add_argument("--weights_json", required=True, type=str, help="Path to JSON file with objective → weight mapping")
    parser.add_argument("--rollout_dir", default="rollout_data/", type=str)
    parser.add_argument("--n_traj", default=10, type=int, help="# trajectories per policy to load")
    parser.add_argument("--no-convo-base-line", action='store_true', help="Whether to use no conversation baseline for the reward function, only used if reward-fun-type==learned_rew")
    args = parser.parse_args()

    # Load weight dictionary --------------------------------------------------
    # with open(args.weights_json, "r") as f:
    #     weights_dict = {k: float(v) for k, v in json.load(f).items()}

    #Becareful about changing the order of feature_names from what was used for saving (!!)
    _,_, _, feature_names, _ , _= load_reward_ranges(args.env_name, range_ceiling=float('inf'),horizon=100, use_no_convo_baseline=args.no_convo_base_line)#horizon doesn't matter

    weights = np.load(f"active_learning_res/{args.weights_name}_feasible_weights.npy")
    weights_dict = {name: weights[i] for i,name in enumerate(feature_names)}

    # Policy names are hard‑coded here to mimic the original main() behaviour.
    # Adjust as needed.
    if args.env_name == "traffic":
        # policy_names = [
        #     "traffic_base_policy",
        #     "2025-06-24_13-51-42",
        #     "2025-06-17_16-14-06",
        #     "2025-07-10_13-33-33",
        #     "2025-07-09_16-57-36"
        # ]
        # reward_fns = ["true_reward", "proxy_reward"]
        policy_names = [f"{args.env_name}_policy_{i}" for i in range(50)]
        policy_names += [f"{args.env_name}_policy_{i}_singleagent_merge_bus_bigger" for i in range(50)]
    elif args.env_name == "pandemic":
        # policy_names = ["pandemic_base_policy","2025-06-24_13-49-08","2025-05-05_21-29-00", "2025-07-10_11-40-34","2025-07-09_16-58-20"]#,pandemic_base_policy "2025-05-05_21-29-00","2025-06-24_13-49-08"
        # reward_fns = ["true_reward", "proxy_reward"]
        policy_names = [f"{args.env_name}_policy_{i}_medium" for i in range(50)]
        policy_names +=  [f"{args.env_name}_policy_{i}" for i in range(50)]

    elif args.env_name == "mujoco":
        checkpoint_paths_file = f"data/gt_rew_fn_data/{args.env_name}_gt_rew_fns2checkpoint_paths.pkl"
        with open(checkpoint_paths_file, "rb") as f:
            paths = pickle.load(f)
        all_keys = list(paths.keys())
        policy_names = [f"mujoco_policy_{gt_rew_i}" for gt_rew_i in all_keys]
    
    else:
        raise ValueError(f"Unknown env_name: {args.env_name}")

    reward_fns = ["sampled_gt_rewards"]
    # policy_names = [f"policy_{i}" for i in range(25)]
    # Run evaluation for *each* chosen ground‑truth reward key ----------------
    for gt_key in reward_fns:
        evaluate_reward_weights_from_preferences(
            args.env_name,
            args.rollout_dir,
            policy_names,
            gt_key,
            weights_dict,
            n_trajectories_per_policy=args.n_traj,
            no_convo_base_line=args.no_convo_base_line
        )
