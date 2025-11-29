import json
import os
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import copy
import argparse
import math
import ray  # noqa: F401 – imported by the original script; leave as‑is
from sklearn.linear_model import LinearRegression  # noqa: F401 – kept for backward‑compat
from sklearn.metrics import mean_squared_error  # noqa: F401 – kept for backward‑compat
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# from test_glucose_reward import create_glucose_reward
from test_pandemic_reward import create_pandemic_reward
from test_traffic_reward import create_traffic_reward
from test_mujoco_ant_reward import create_mujoco_ant_reward


from rl_utils.reward_wrapper import SumReward  # noqa: F401 – used elsewhere in the codebase
from utils.glucose_rollout_and_save import TrajectoryStep  # noqa: F401 – type hinting only
from utils.glucose_gt_rew_fns import (
    MagniGroundTruthReward,
    ExpectedCostGroundTruthReward,
)
from utils.traffic_gt_rew_fns import TrueTrafficRewardFunction
from utils.pandemic_gt_rew_fns import TruePandemicRewardFunction
from utils.mujoco_gt_rew_fns import TrueMujocoAntRewardFunction
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import scoreatpercentile  # or use np.percentile
from typing import Sequence, List, Tuple, Dict, Any

# ------------------------------------------------------------------ #
# Helper: save pairs in the lowest percentile of Mahalanobis distance
# ------------------------------------------------------------------ #
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
# 1) Roll‑out utilities
################################################################################

def load_rollout_data(
    env_name: str,
    rollout_dir: str,
    policy_names: List[str],
    n_trajectories_per_policy: int = 10,
) -> List[Dict[str, Any]]:
    """Unchanged from the original script – see original docstring for details."""

    data: List[Dict[str, Any]] = []
    trajectory_dir = Path(rollout_dir) / "trajectories"
    n_dones = 0

    if env_name == "pandemic":
        true_rew_fn_sampler = TruePandemicRewardFunction()
    elif env_name == "traffic":
        true_rew_fn_sampler = TrueTrafficRewardFunction()
    elif env_name == "mujoco":
        true_rew_fn_sampler = TrueMujocoAntRewardFunction()
    else:
        raise ValueError("Other environments are not implemented yet!")

    for policy_name in policy_names:

        #the g.t. reward function params are different for pandemic env with different town sizes (e.g., because of ICU capacity)
        if env_name == "pandemic" and "medium" in policy_name:
            true_rew_fn_sampler = TruePandemicRewardFunction(town_size="medium")
        elif env_name == "pandemic":
            true_rew_fn_sampler = TruePandemicRewardFunction(town_size="tiny")

        traj_length = 0
        # print ("==========================policy_name:", policy_name)
        for i in range(n_trajectories_per_policy):
            # if env_name == "pandemic":
            #     trajectory_path = trajectory_dir / f"{policy_name}_trajectory_{i}.pkl"
            # else:
            trajectory_path = trajectory_dir / f"{policy_name}_trajectory_{i}_full.pkl"
            if not trajectory_path.exists():
                print(f"Warning: Trajectory file {trajectory_path} not found")
                continue

            # print (trajectory_path)

            with open(trajectory_path, "rb") as f:
                trajectory = pickle.load(f)

            for step in trajectory:
                if env_name == "glucose":
                    ec_gt_reward = ExpectedCostGroundTruthReward()
                    magni_gt_reward = MagniGroundTruthReward()
                    data.append(
                        {
                            "obs": step.obs,
                            "action": step.action,
                            "next_obs": step.next_obs,
                            "expected_cost_rew": ec_gt_reward.calculate_reward(
                                step.obs, step.action, step.next_obs
                            ),
                            "magni_rew": magni_gt_reward.calculate_reward(
                                step.obs, step.action, step.next_obs
                            ),
                            "done": False,
                            "policy_name": policy_name,
                        }
                    )
                else:
                    
                    # imp_reward = TrueTrafficRewardFunction()
                    # val = imp_reward.calculate_reward(
                    #     step.obs, step.action, step.next_obs
                    # )

                    # print ("imp_reward:", val)
                    # print ("true_reward:", step.true_reward)
                    # print ("\n")
                    
                    
                    data.append(
                        {
                            "obs": step.obs,
                            "action": step.action,
                            "next_obs": step.next_obs,
                            "sampled_gt_rewards": true_rew_fn_sampler.sample_calc_reward(step.obs, step.action, step.next_obs),
                            "pan_true_reward": step.true_reward,
                            "pan_proxy_reward": step.proxy_reward,
                            "done": False,
                            "policy_name": policy_name,
                        }
                    )
                traj_length += 1

            # Mark episode end
            data[-1]["done"] = True
            n_dones += 1
            # _kv("traj_length", traj_length, indent=4)
            traj_length = 0

    # Persist processed data ---------------------------------------------------
    policy_names_str = "_".join(policy_names)
    data_filename = f"{env_name}_rollout_data.pkl"
    data_dir = Path(rollout_dir) / "rollout_data"
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / data_filename

    # print(f"\nSaving processed rollout data → {data_path}")
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    # _kv("n_dones", n_dones, indent=2)
    return data

################################################################################
# 2) Feature extraction helpers
################################################################################

def extract_reward_features(
    env_name: str,
    obs: Any,
    action: Any,
    next_obs: Any,
    reward_functions: List[Any],
) -> np.ndarray:
    """Extract per‑reward‑function features for a single transition."""

    if env_name == "glucose":
        obs.bg = np.asarray(obs.bg)
        obs.insulin = np.asarray(obs.insulin)
        obs.cho = np.asarray(obs.cho)
        next_obs.bg = np.asarray(next_obs.bg)
        next_obs.insulin = np.asarray(next_obs.insulin)
        next_obs.cho = np.asarray(next_obs.cho)

    # feats = [rf.calculate_reward(obs, action, next_obs) for rf in reward_functions]
    feats = []
    for rf in reward_functions:
        r_feat = rf.calculate_reward(obs, action, next_obs)
        if np.isnan(r_feat):
            feats.append(0)
        else:
            feats.append(r_feat)
    #print class names of reward_functions:
    # for rf_i, rf in enumerate(reward_functions):
    #     print (feats[rf_i])
    #     print (rf.__class__.__name__, end=", ")
    # print ("=========")
    return np.asarray(feats)


def get_traj_features(
    env_name: str,
    traj: List[Dict[str, Any]],
    transition_cache: Dict[Tuple[np.ndarray, ...], np.ndarray],
    reward_functions: List[Any],
) -> Tuple[np.ndarray, Dict[Tuple[np.ndarray, ...], np.ndarray]]:
    """Sum feature vectors over all transitions in a trajectory (with caching)."""

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


def get_traj_raw_in(
    traj: List[Dict[str, Any]],
):
    """Sum feature vectors over all transitions in a trajectory (with caching)."""

    features = []
    for step in traj:
       

        if step["obs"].flatten()[0] == 99:
            #truncate obs to be the last 33 elements (e.g., ignore hisitory) for the pandemic env
            #we need to ignore the obs history other we run into OOM issues when running LBFGS
            raw_in= np.concatenate(
                (
                    step["obs"].flatten()[-33:],
                    step["action"].flatten(),
                    step["next_obs"].flatten()[-33:],
                )
            )
        else:
            raw_in= np.concatenate(
                    (
                        step["obs"].flatten(),
                        step["action"].flatten(),
                        step["next_obs"].flatten(),
                    )
                )

        features.append(raw_in)
    return np.sum(features, axis=0)
################################################################################
# 3) Dataset construction
################################################################################
def group_trajectories(
    rollout_data: List[Dict[str, Any]],
    policy_names: List[str],
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[List[Any]]]]:
    """Collect transitions into separate trajectories."""

    trajectories: List[List[Any]] = []
    policy2trajectories: Dict[str, List[List[Any]]] = {p: [] for p in policy_names}
    current_traj: List[Any] = []

    for transition in rollout_data:
        current_traj.append(transition)
        if transition["done"]:
            trajectories.append(current_traj)
            policy2trajectories[transition["policy_name"]].append(current_traj)
            current_traj = []

    if current_traj:
        trajectories.append(current_traj)
        policy2trajectories[current_traj[0]["policy_name"]].append(current_traj)

    #make sure each trajectory has the same length
    for traj in trajectories:
        if len(traj) != len(trajectories[0]):
            raise ValueError(f"Trajectory {traj[0]['policy_name']} has length {len(traj)} but expected {len(trajectories[0])}")
    return trajectories, policy2trajectories


def build_preference_dataset(
    env_name: str,
    trajectories: List[List[Dict[str, Any]]],
    reward_functions: List[Any],
    gt_reward_key: str,
    use_raw_in=False
) -> Tuple[np.ndarray, np.ndarray, List[float], Dict[str, List[Tuple[np.ndarray, float, str]]]]:
    """Generate (Δfeatures, preference) pairs for all unordered trajectory pairs."""

    X, y, reward_diffs = defaultdict(list),defaultdict(list),defaultdict(list)

    transition_cache: Dict[Any, np.ndarray] = {}
    policy2pairs: Dict[str, List[Tuple[np.ndarray, float, str]]] = defaultdict(list)
    traj_pairs = defaultdict(list)
    reward_i2indiff_i = defaultdict(list)
    for i, traj1 in enumerate(trajectories):
        for traj2 in trajectories[i + 1 :]:
            # Feature differences
            if use_raw_in:
                f1 = get_traj_raw_in(traj1)
                f2 = get_traj_raw_in(traj2)
            else:
                f1, transition_cache = get_traj_features(env_name, traj1, transition_cache, reward_functions)
                f2, transition_cache = get_traj_features(env_name, traj2, transition_cache, reward_functions)
            delta_f = f1 - f2
            if delta_f.dtype == object:
                delta_f = np.concatenate([np.atleast_1d(x) for x in delta_f]).astype(np.float64)

            assert len(traj1[0][gt_reward_key]) ==len(traj2[0][gt_reward_key])
            r1s = [sum(step[gt_reward_key][r_i] for step in traj1) for r_i in range(len(traj1[0][gt_reward_key]))]
            r2s = [sum(step[gt_reward_key][r_i] for step in traj2) for r_i in range(len(traj2[0][gt_reward_key]))]
            
            for r_i, r1 in enumerate(r1s):
                
                r2 = r2s[r_i]
                if r1 == r2:
                    # duplicate for both directions to keep behaviour unchanged
                    y[r_i].extend([1, 0])
                    X[r_i].extend([delta_f, delta_f])
                    reward_diffs[r_i].extend([0, 0])
                    traj_pairs[r_i].extend([(traj1, traj2),(traj1, traj2)])
                    
                    reward_i2indiff_i[r_i].extend([len(y[r_i])-1,len(y[r_i])-2])
                else:
                    X[r_i].append(delta_f)
                    reward_diffs[r_i].append(r1 - r2)
                    y[r_i].append(int(r1 > r2))
                    traj_pairs[r_i].append((traj1, traj2))
            
                policy2pairs[traj1[0]["policy_name"] + "_r" +str(r_i)].append((delta_f, r1 - r2, traj2[0]["policy_name"]))
    # assert False
    # for k,v in reward_diffs.items():
    #     print ("Mean diff in return:")
    #     print (np.mean(v))

    return X, y, reward_diffs, policy2pairs, traj_pairs, reward_i2indiff_i

def load_persisted_preference_data(env_name, policy_names, gt_reward_key):
    save_dir = Path("reward_learning_data")
    save_dir.mkdir(exist_ok=True)
    name = f"{env_name}_preference_{{}}_{gt_reward_key}.pkl"

    with open(save_dir / name.format("X"), "rb") as f:
        X = pickle.load(f)
    with open(save_dir / name.format("y"), "rb") as f:
        y = pickle.load(f)
    with open(save_dir / name.format("reward_diffs"), "rb") as f:
        reward_diffs = pickle.load(f)
    with open(save_dir / name.format("traj_pairs"), "rb") as f:
        traj_pairs = pickle.load(f)
    return X,y,reward_diffs,traj_pairs


def persist_preference_data(
    env_name: str,
    policy_names: List[str],
    gt_reward_key: str,
    X: np.ndarray,
    y: np.ndarray,
    reward_diffs: List[float],
    traj_pairs
) -> Path:
    """Save X, y (and reward_diffs) to disk and return directory path."""

    save_dir = Path("reward_learning_data")
    save_dir.mkdir(exist_ok=True)
    name = f"{env_name}_preference_{{}}_{gt_reward_key}.pkl"

    with open(save_dir / name.format("X"), "wb") as f:
        pickle.dump(X, f)
    with open(save_dir / name.format("y"), "wb") as f:
        pickle.dump(y, f)
    with open(save_dir / name.format("reward_diffs"), "wb") as f:
        pickle.dump(reward_diffs, f)
    with open(save_dir / name.format("traj_pairs"), "wb") as f:
        pickle.dump(traj_pairs, f)
    return save_dir

################################################################################
# 4) Model definition & training helpers
################################################################################

class PreferenceRewardNetwork(nn.Module):
    def __init__(self, input_size: int, use_linear: bool = True):
        super().__init__()
        if use_linear:
            self.network = nn.Sequential(nn.Linear(input_size, 1, bias=False))
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

    def forward(self, x):  # noqa: D401
        return self.network(x)


def _load_saved_weights_if_any(
    model: nn.Module,
    reward_functions: List[Any],
    save_dir: Path,
    env_name: str,
    gt_reward_key: str,
    policy_names: List[str],
) -> None:
    """Reload previously saved linear weights (if *resume* is set)."""

    weights_path = save_dir / f"{env_name}_{gt_reward_key}_preference_weights.json"
    if not weights_path.exists():
        print(f"No saved weights found → starting from scratch")
        return

    with open(weights_path, "r") as f:
        saved_weights = json.load(f)
    weights_array = np.array([saved_weights[rf.__class__.__name__] for rf in reward_functions])
    with torch.no_grad():
        model.network[0].weight.data = torch.tensor(weights_array).view(1, -1)
    print(f"Loaded weights from {weights_path}")


def train_reward_network(
    all_r_X: np.ndarray,
    all_r_y: np.ndarray,
    reward_functions: List[Any],
    use_linear_model: bool,
    resume: bool,
    env_name: str,
    gt_reward_key: str,
    policy_names: List[str],
    reward_i2indiff_i
) -> Tuple[nn.Module, np.ndarray]:
    """Standard BCE training loop with cosine annealing scheduler."""

    weights_vecs = []
    models = []
    # for X,y in zip(all_r_X, all_r_y):
    # if env_name == "pandemic":
    #     true_rew_fn_sampler = TruePandemicRewardFunction()
    # elif env_name == "traffic":
    #     true_rew_fn_sampler = TrueTrafficRewardFunction()
    # else:
    #     raise NotImplementedError("Other environments are not implemented yet!")

    for r_i in all_r_X.keys():
        X = all_r_X[r_i]
        y = all_r_y[r_i]

        # Normalize features--I would prefer not to do this but LBFGS is really brittle otherwise
        
        #   save_dir = Path("reward_learning_data")

        #     if resume:
        #         _load_saved_weights_if_any(model, reward_functions, save_dir, env_name, gt_reward_key, policy_names)

        feat_is = list(range(len(X[0]))) + [-1]
        # feat_is = [-1]
        best_loss = float("inf")
        best_weights = None
        best_weights_X = None
        X = np.array(X)
        y = torch.tensor(y, dtype=torch.float32)

        for normalize_in in [True, False]:
            for feat_i in feat_is:
                #zero out all features except for feat_i
                if feat_i == -1:
                    X_single = X
                else:
                    X_single = np.zeros_like(X)
                    X_single[:, feat_i] = X[:, feat_i]

                if normalize_in:
                    scaler = StandardScaler()
                    X_single = scaler.fit_transform(X_single)   # before turning into tensors
                X_single = torch.from_numpy(X_single).float()  # see #2

                model = PreferenceRewardNetwork(len(X_single[0]), use_linear_model)

                criterion = nn.BCEWithLogitsLoss()
                #--------------LBFGS-----------------
                opt = torch.optim.LBFGS(model.parameters(), max_iter=1000, line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-12)            

                def closure():
                    opt.zero_grad()
                    logits = model(X_single).squeeze(-1)
                    loss = criterion(logits, y)
                    loss.backward()
                    return loss

                for i in range(100):   # LBFGS will stop early when converged
                    loss = opt.step(closure)
                    # print(f"Iter {i} | loss {loss.item():.6f}")

                model_weights = model.network[0].weight.data.numpy().flatten()
                if feat_i != -1:
                    #zero weights out except for feat_i
                    model_weights = np.zeros_like(model_weights)
                    model_weights[feat_i] = model.network[0].weight.data.numpy().flatten()[feat_i]
                
                #TODO: we should use accuracy instead of loss to determine best_weights (I think)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_weights = copy.deepcopy(model_weights)
                    #we need to store the X_single that corresponds to best_weights because we scale X when training the weights to avoid ill-conditioning issues with LBFGS
                    best_weights_X = copy.deepcopy(X_single)

        # with torch.no_grad():  # avoids tracking in autograd
        #     model.network[0].weight.copy_(torch.tensor([[100, 100]]))
        #     logits = model(X).squeeze(-1)
        #     loss = criterion(logits, y)
        #     print(f"Iter MANUAL | loss {loss.item():.6f}")

        # with torch.no_grad():
        #     model.network[0].weight.copy_(torch.tensor([best_weights], dtype=torch.float32))
        #     logits = model(X_single).squeeze(-1)
        #     loss = criterion(logits, y)
        #     print(f"Best weights | loss {loss.item():.6f}")

        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)   # before turning into tensors
        # X = torch.from_numpy(X).float()  # see #2

        # model = PreferenceRewardNetwork(len(X[0]), use_linear_model)

        # criterion = nn.BCEWithLogitsLoss()
        # #--------------LBFGS-----------------
        # opt = torch.optim.LBFGS(model.parameters(), max_iter=1000, line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-12)            

        # def closure():
        #     opt.zero_grad()
        #     logits = model(X).squeeze(-1)
        #     loss = criterion(logits, y)
        #     loss.backward()
        #     return loss
        # for i in range(100):   # LBFGS will stop early when converged
        #     loss = opt.step(closure)


        # with torch.no_grad():
        #     best_weights = model.network[0].weight.data.numpy().flatten()
        #     model.network[0].weight.copy_(torch.tensor([best_weights], dtype=torch.float32))
        #     logits = model(X).squeeze(-1)
        #     loss = criterion(logits, y)
        #     print(f"Best weights | loss {loss.item():.6f}")

        # best_state = copy.deepcopy(model.state_dict())
        # Quick training accuracy check -------------------------------------------
        # print ("best_loss:", best_loss)
        with torch.no_grad():
            total = 0
            acc = 0.0
            for x_i, x in enumerate(best_weights_X):
                # if true_rew_fn_sampler.feature2_reward(x.numpy(), r_i) == 0:
                if x_i in reward_i2indiff_i[r_i]:
                    continue
                pred = model(x)
                total+=1
                # print (x, torch.sigmoid(pred), y[x_i])
                if (torch.sigmoid(pred) > 0.5).float().item() == y[x_i].item():
                    acc+=1

        if total == 0:
            _kv("Accuracy over training set, indifferent prefs. removed:", "N/A")
        else:
            _kv("Accuracy over training set, indifferent prefs. removed:", f"{acc/total:.3f}")

        print ("Note: we are allowing the input features to be normalized(!!)")
        #print model weights
        # for rf, w in zip(reward_functions, model.network[0].weight.data.numpy().flatten()):
        #     print (rf.__class__.__name__, w)
        # assert False
        # Linear model ⇒ return learned weights vector for convenience
        if use_linear_model:
            with torch.no_grad():
                weights_vec = model.network[0].weight.data.numpy().flatten()
        else:
            weights_vec = np.array([])
        weights_vecs.append(weights_vec)
        models.append(model)
    return models, weights_vecs

################################################################################
# 5) Evaluation helpers
################################################################################

def evaluate_model(
    env_name: str,
    trajectories: List[List[Dict[str, Any]]],
    reward_functions: List[Any],
    models: nn.Module,
    weights_dict: Dict[str, np.ndarray],
    use_weights_dict: bool,
    gt_reward_key: str,
    policy_names: List[str],
    policy2pairs: Dict[str, List[Tuple[np.ndarray, float, str]]],
    use_raw_in= False,
) -> None:
    """Replicates the original evaluation & pretty‑prints results."""

    # _banner("Evaluating learned weights")
    trans2feat_cache = {}
    for r_i in range(len(trajectories[0][0][gt_reward_key])):
        _banner(f"Reward function {r_i}")

        if use_weights_dict:
            try:
                weights_vec = np.array([weights_dict[rf.__class__.__name__] for rf in reward_functions])
            except KeyError as e:
                missing = e.args[0]
                raise KeyError(
                    f"Weight for reward function '{missing}' missing in the supplied dictionary."
                ) from None
        else:
            model = models[r_i]


        actual_returns, predicted_returns = [], []
        policy2_actual, policy2_pred = {p: [] for p in policy_names}, {p: [] for p in policy_names}

        for traj in trajectories:
            gt_r = sum(step[gt_reward_key][r_i] for step in traj)
            actual_returns.append(gt_r)

            if use_raw_in:
                feats = get_traj_raw_in(traj)
            else:
                feats = []
                for s in traj:
                    key = tuple(
                        np.concatenate(
                            (
                                s["obs"].flatten(),
                                s["action"].flatten(),
                                s["next_obs"].flatten(),
                            )
                        )
                    )
                    if key in trans2feat_cache:
                        feat = trans2feat_cache[key]
                    else:
                        feat = extract_reward_features(env_name, s["obs"], s["action"], s["next_obs"], reward_functions)
                        trans2feat_cache[key] = feat
                    feats.append(feat)
            
                feats = np.sum(feats,axis=0)

            if feats.dtype == np.object:
                feats = np.concatenate([np.atleast_1d(x) for x in feats]).astype(np.float64)

            if use_weights_dict:
                pred_r = float(feats @ weights_vec)
            else:
                with torch.no_grad():
                    pred_r = model(torch.tensor(feats, dtype=torch.float32)).item()
            predicted_returns.append(pred_r)

            pol = traj[0]["policy_name"]
            policy2_actual[pol].append(gt_r)
            policy2_pred[pol].append(pred_r)

        # Global Kendall‑Tau -------------------------------------------------------
        print ("actual_returns:", actual_returns)
        print ("predicted_returns:", predicted_returns)
        print ("--------------------------------")
        tau, p = kendalltau(actual_returns, predicted_returns)
        _kv("Global Kendall‑Tau", f"{tau:.3f}  (p={p:.2e})")
        continue
        # Intra‑ & cross‑policy analyses ------------------------------------------
        # _banner("Per‑policy diagnostics", "-")
        # for p in policy_names:
        #     if len(policy2_actual[p]) < 2:
        #         _kv(alias_policy_name(p, policy_names), "<2 trajectories – skipped", indent=2)
        #         continue
        #     tau_p, _ = kendalltau(policy2_actual[p], policy2_pred[p])
        #     _kv(alias_policy_name(p, policy_names), f"Kendall‑Tau = {tau_p:.3f}", indent=2)

        # Exhaustive accuracy (same as original script) ---------------------------
        # Intra-policy preference accuracy -----------------------------------------
        intra_n, intra_d = defaultdict(int), defaultdict(int)
        for p, pairs in policy2pairs.items():
            
            #skip pairs not associated with the g.t. reward function r_i
            pair_r_i = int(p.split("_")[-1].replace("r",""))

            if pair_r_i != r_i:
                continue
            # print (".    here")
            # print (".     ", len(pairs))
            for feat_diff, reward_diff, other_p in pairs:
                if other_p != p.replace("_r"+str(pair_r_i),"") or reward_diff == 0:
                    continue
                gt_pref = int(reward_diff > 0)
                with torch.no_grad():
                    pred_pref = int(model(torch.tensor(feat_diff, dtype=torch.float32)).item() > 0)
                intra_d[p] += 1
                intra_n[p] += int(gt_pref == pred_pref)

        # Collect and sort intra accuracies
        intra_results = []
        for p in policy_names:
            p += "_r" +str(r_i)
            acc = intra_n[p] / intra_d[p] if intra_d[p] else float("nan")
            intra_results.append((acc, p))
        intra_results.sort()  # sort by accuracy (ascending)

        _banner("Intra‑policy preference accuracy (sorted, indifferent g.t. prefs removed)", "-")
        for acc, p in intra_results:
            p += "_r" + str(r_i)
            _kv(p, f"Accuracy = {acc:.3f}  (pairs = {intra_d[p]})", indent=2)


        # Inter-policy preference accuracy ----------------------------------------
        inter_correct, inter_total = defaultdict(int), defaultdict(int)
        inter_results = []

        for i, p1 in enumerate(policy_names):
            for p2 in policy_names[i + 1:]:
                key = tuple(sorted((p1, p2)))

                for pol in key:
                    for feat_diff, reward_diff, other_p in policy2pairs[pol+ "_r" +str(r_i)]:
                        if {pol, other_p} != set(key) or reward_diff == 0:
                            continue

                        gt_pref = int(reward_diff > 0)
                        with torch.no_grad():
                            pred_pref = int(model(torch.tensor(feat_diff, dtype=torch.float32)).item() > 0)

                        inter_total[key] += 1
                        inter_correct[key] += int(gt_pref == pred_pref)

                acc = inter_correct[key] / inter_total[key] if inter_total[key] else float("nan")
                inter_results.append((acc, key))

        inter_results.sort()  # sort by accuracy (ascending)

        _banner("Cross‑policy preference accuracy (sorted, indifferent g.t. prefs removed)", "-")
        for acc, (p1, p2) in inter_results:
            label = f"{p1} ↔ {p2}"
            _kv(label, f"Accuracy = {acc:.3f}  (pairs = {inter_total[(p1, p2)]})", indent=2)
# ------------------------------------------------------------------ #
# Helper: accuracy as a function of Mahalanobis-distance percentile
# ------------------------------------------------------------------ #
def preference_accuracy_by_percentile(
    X: np.ndarray,
    y: np.ndarray,
    model: torch.nn.Module,
    percentiles=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
) -> None:
    """
    Print preference accuracy inside each percentile bucket of the
    Mahalanobis-distance distribution (feature space).

    Parameters
    ----------
    X : (N, d) array
        Δ-feature vectors used for training / evaluation.
    y : (N,) array
        Binary preference labels (1 ⇒ x ≻ 0, 0 ⇒ 0 ≻ x).
    model : torch.nn.Module
        Trained preference model (outputs logits).
    percentiles : iterable of ints
        Boundaries for percentile buckets.  Example above gives deciles.
    """
    # 1) Mahalanobis distances ------------------------------------------------
    mu  = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    D = np.array([mahalanobis(x, mu, cov_inv) for x in X])

    # 2) Compute bucket edges -------------------------------------------------
    pct_edges = np.percentile(D, percentiles)

    # 3) Predictions ----------------------------------------------------------
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

    # 4) Evaluate inside each bucket -----------------------------------------
    print("\nPreference accuracy by Mahalanobis-distance percentile")
    print("-------------------------------------------------------")
    for lo, hi in zip(pct_edges[:-1], pct_edges[1:]):
        mask = (D >= lo) & (D < hi) if hi < pct_edges[-1] else (D >= lo) & (D <= hi)
        n = mask.sum()
        if n == 0:
            acc = np.nan
        else:
            acc = (preds[mask] == y[mask]).mean()
        pct_lo = percentiles[list(pct_edges).index(lo)]
        pct_hi = percentiles[list(pct_edges).index(hi)]
        print(f"{pct_lo:>3d}–{pct_hi:>3d} percentile:  n = {n:>5d}   accuracy = {acc:6.3f}")

    print("-------------------------------------------------------\n")

################################################################################
# 6) Print learned weights helper
################################################################################

def print_learned_weights(
    weights_dict: Dict[str, float],
    reward_functions: List[Any],
    env_name: str,
    gt_reward_fn: str,
) -> None:
    """Print the learned weights corresponding to each reward function feature."""
    
    # _banner("LEARNED WEIGHTS")
    # _kv("Environment", env_name, indent=2)
    # _kv("Ground truth reward", gt_reward_fn, indent=2)
    # print()
    
    # Print weights with their corresponding reward function names
    print("  Feature weights:")
    print("  " + "="*50)
    
    total_weight = 0.0
    for rf in reward_functions:
        weight = weights_dict.get(rf.__class__.__name__, 0.0)
        total_weight += abs(weight)
        print(f"  {rf.__class__.__name__:>35s}: {weight:>10.6f}")
    
    print("  " + "="*50)
    print()

def plot_accuracy_by_return_gap_fast(
    all_r_model: nn.Module,
    all_r_reward_diffs: List[float],
    all_r_X: np.ndarray,
    all_r_y: np.ndarray,
    env_name: str,
):
    """
    Plot preference accuracy as a function of absolute return difference threshold,
    using already-computed delta features and reward_diffs from build_preference_dataset.
    """

    for r_i in all_r_X.keys():
        X = all_r_X[r_i]
        y = all_r_y[r_i]
        reward_diffs = all_r_reward_diffs[r_i]
        model = all_r_model[r_i]

        # Step 1: sort all (|reward_diff|, delta_feat, label)
        entries = sorted(
            [(abs(d), f, label) for d, f, label in zip(reward_diffs, X, y)],
            key=lambda x: x[0]
        )
        abs_diffs = [e[0] for e in entries]

        # Step 2: define ≤10 thresholds
        if len(abs_diffs) <= 10:
            thresholds = sorted(set(abs_diffs))
        else:
            thresholds = np.percentile(abs_diffs, np.linspace(0, 100, 10)).tolist()

        # Step 3: compute accuracy per threshold bucket
        xs, ys, counts = [], [], []
        # print ("\n---------------------")
        for t in thresholds:
            filtered = [(f, l) for d, f, l in entries if d >= t]
            if not filtered:
                continue
            feats = np.array([f for f, _ in filtered])
            labels = np.array([l for _, l in filtered])

            with torch.no_grad():
                logits = model(torch.tensor(feats, dtype=torch.float32)).squeeze()
                preds = (torch.sigmoid(logits) > 0.5).int().numpy()
            acc = (preds == labels).mean()
            xs.append(t)
            ys.append(acc)
            counts.append(len(filtered))

            # print ("t:", t, "acc:", acc, "n:", len(filtered))


        # Step 4: plot
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, marker='o')
        for x, y, c in zip(xs, ys, counts):
            plt.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.xlabel("Minimum |Return Difference|")
        plt.ylabel("Preference Accuracy")
        plt.title(f"{env_name} – Accuracy vs Return Gap (≥ threshold)")
        plt.grid(True)
        plt.tight_layout()
        save_dir=f"plots/{env_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/rm={r_i}_accuracy_vs_return_gap.png")

################################################################################
# 7) Orchestrator – original public API
################################################################################

def learn_reward_weights_from_preferences(
    env_name: str,
    rollout_dir: str,
    policy_names: List[str],
    gt_reward_fn: str,
    n_trajectories_per_policy: int = 10,
    n_preference_pairs: int = 2000,  # kept for backwards‑compat – not used directly
    use_linear_model: bool = True,
    resume: bool = False,
    use_raw_in= False,  # if True, use raw input features instead of reward function features
    no_convo_base_line: bool = False
):
    """Thin wrapper that reproduces the original behaviour with modular helpers."""

    # _banner("START preference learning")
    _kv("env", env_name, indent=2)
    _kv("reward fn used for eval", gt_reward_fn, indent=2)

    # 1) Load / preprocess roll‑outs ------------------------------------------
    rollout_data = load_rollout_data(env_name, rollout_dir, policy_names, n_trajectories_per_policy)
    # _kv("Transitions loaded", len(rollout_data), indent=2)

    # 2) Instantiate reward functions -----------------------------------------
    if env_name == "glucose":
        reward = create_glucose_reward()
    elif env_name == "pandemic":
        reward = create_pandemic_reward(no_convo_base_line=no_convo_base_line)
    elif env_name == "traffic":
        reward = create_traffic_reward(no_convo_base_line=no_convo_base_line)
    elif env_name == "mujoco":
        reward = create_mujoco_ant_reward(no_convo_base_line=no_convo_base_line)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    reward_functions = reward._reward_fns  # noqa: SLF001 – preserve original API

    # 3) Trajectory grouping & preference data --------------------------------
    trajectories, policy2trajectories = group_trajectories(rollout_data, policy_names)
    _kv("Trajectories", len(trajectories), indent=2)

    X, y, reward_diffs, policy2pairs, traj_pairs,reward_i2indiff_i = build_preference_dataset(
        env_name, trajectories, reward_functions, gt_reward_fn,use_raw_in=use_raw_in
    )
    _kv("Preference pairs", len(X[0]), indent=2)
    _kv("# of g.t. reward functions", len(X.keys()), indent=2)

    # # 4) Persist dataset (unchanged side‑effect) ------------------------------
    # if not use_raw_in:
    #     persist_preference_data(env_name, policy_names, gt_reward_fn, X, y, reward_diffs, traj_pairs)
    #     X, y, reward_diffs, traj_pairs = load_persisted_preference_data(env_name, policy_names, gt_reward_fn)

    # 5) Train / resume model --------------------------------------------------
    models, weights_vecs = train_reward_network(
        X, y, reward_functions, use_linear_model, resume, env_name, gt_reward_fn, policy_names,reward_i2indiff_i,
    )

    # preference_accuracy_by_percentile(X, y, model)  # default deciles

    # 6) Save learned weights --------------------------------------------------
    weights_dict = [{rf.__class__.__name__: float(w) for rf, w in zip(reward_functions, weights_vec)} for weights_vec in weights_vecs]
    save_dir = Path("reward_learning_data")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{env_name}_{gt_reward_fn}_preference_weights.json"
    with open(save_path, "w") as f:
        json.dump(weights_dict, f, indent=4)
    # print(f"\nLearned weights saved → {save_path}")

    # 7) Print learned weights ------------------------------------------------
    # if not use_raw_in:
    #     print_learned_weights(weights_dict, reward_functions, env_name, gt_reward_fn)

    # 8) Evaluation ------------------------------------------------------------
    evaluate_model(
        env_name, trajectories, reward_functions, models, None, False, gt_reward_fn, policy_names, policy2pairs,use_raw_in=use_raw_in
    )

    # print_learned_weights(weights_dict, reward_functions, env_name, gt_reward_fn)
    plot_accuracy_by_return_gap_fast(
        models,
        reward_diffs,
        X,
        y,
        env_name + "_"+ gt_reward_fn,
    )

################################################################################
# 8) CLI – unchanged behaviour
################################################################################

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
        "--use_raw_ins",        # long-form flag only (short flag unnecessary here)
        action="store_true",    # when the flag is present, the value becomes True
        default=False,          # otherwise it stays False
        help="If set, the agent receives raw (s,a,s') instead of reward features.",
    )

    parser.add_argument(
        "--no_convo_base_line",        # long-form flag only (short flag unnecessary here)
        action="store_true",    # when the flag is present, the value becomes True
        default=False,          # otherwise it stays False
        help="If set, do not include convolutional base line features in the reward function.",
    )
    
    args = parser.parse_args()
    env_name = args.env_name
    use_raw_in = args.use_raw_ins

    if env_name == 'traffic':
        # policy_names = [
        #     "traffic_base_policy",
        #     "2025-06-24_13-51-42",
        #     "2025-06-17_16-14-06",
        #     "2025-07-10_13-33-33",
        #     "2025-07-09_16-57-36",
        #     "traiffc-uniform-policy",
        # ]
        # reward_fns = ["true_reward", "proxy_reward"]
        # true_reward_description = "-1.0*(normalized measure of all vehicle's closeness to the target velocity) + -0.1*(normalized measure of closeness of max_over_all_vehicles(distance to closest vehicle/vehicle speed) to target headway value) + -1.0*(closeness of mean vehicle acceleration to target acceleration)"
        extra_details = ""
        policy_names = [f"{env_name}_policy_{i}" for i in range(50)]
        policy_names += [f"{env_name}_policy_{i}_singleagent_merge_bus_bigger" for i in range(50)]
    elif env_name == "pandemic":
        # policy_names = ["pandemic_base_policy","2025-06-24_13-49-08","2025-05-05_21-29-00", "2025-07-10_11-40-34","2025-07-09_16-58-20", "pandemic-uniform-policy"]#,pandemic_base_policy "2025-05-05_21-29-00","2025-06-24_13-49-08"
        # reward_fns = ["true_reward", "proxy_reward"]
        # true_reward_description = "-10*(number of infections + number of critical cases + number of deaths) + -10*(penalty for raising the lockdown stage if infection rate is lower than a treshold) + -0.1*(lockdown stage) + -0.02*(|current lockdown stage - previous lockdown stage|)"
        policy_names = [f"{env_name}_policy_{i}_medium" for i in range(50)]
        policy_names +=  [f"{env_name}_policy_{i}" for i in range(50)]

    elif env_name == "mujoco":
        checkpoint_paths_file = f"data/gt_rew_fn_data/{args.env_name}_gt_rew_fns2checkpoint_paths.pkl"
        with open(checkpoint_paths_file, "rb") as f:
            paths = pickle.load(f)
        all_keys = list(paths.keys())
        policy_names = [f"mujoco_policy_{gt_rew_i}" for gt_rew_i in all_keys]
    
    reward_fns = ["sampled_gt_rewards"]
    
    # print ("The environment's human designed reward function, used for this evaluation, is:", true_reward_description)
    # print ("\n")
    print ("The objectives implemented by the reward-design system")
    # Load the JSON file
    if args.no_convo_base_line:
        with open(f'generated_objectives_no_convo_baseline/{env_name}_objective_descriptions.json', 'r') as f:
            data = json.load(f)
    else:
        with open(f'generated_objectives/{env_name}_objective_descriptions.json', 'r') as f:
            data = json.load(f)
    # Pretty print each key-value pair
    for key, value in data.items():
        print(f"{key}:\n  {value}")
    print ("===============================================")

    if args.no_convo_base_line:
        print ("USING OBJECTIVE FEATURES GENERATED BY NO-CONVERSATION BASELINE")

    for reward_fn in reward_fns:
        learn_reward_weights_from_preferences(
            env_name,
            rollout_dir,
            policy_names,
            gt_reward_fn=reward_fn,
            n_trajectories_per_policy=10, #10, #50
            use_linear_model=True,
            resume=False,
            use_raw_in=use_raw_in,  # if True, use raw input features instead of reward function features
            no_convo_base_line=args.no_convo_base_line
        )
