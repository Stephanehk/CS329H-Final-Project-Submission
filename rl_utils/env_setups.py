import numpy as np
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from occupancy_measures.experiments.traffic_experiments import create_traffic_config
from flow.utils.registry import make_create_env
import json
from rl_utils.reward_wrapper import RewardWrapper
from rl_utils.reward_wrapper import SumReward
from reward_learning.active_pref_learning import load_reward_ranges
import gymnasium as gym

def load_learned_reward(env_type, no_convo_base_line=False):
    """Load the learned reward function for the specified environment."""
    # pandemic_preference_weights_{policy_names_str}.json
   

    if "pandemic" in env_type:
        from test_pandemic_reward import create_pandemic_reward
        reward = create_pandemic_reward(no_convo_base_line=no_convo_base_line)
        reward_functions = reward._reward_fns

        # policy_names = ["pandemic_base_policy","2025-05-05_21-29-00"]
        # policy_names_str = "_".join(policy_names)
        # save_path = f"reward_learning_data/pandemic_preference_weights_{policy_names_str}.json"
    elif "traffic" in env_type:
        from test_traffic_reward import create_traffic_reward
        reward = create_traffic_reward(no_convo_base_line=no_convo_base_line)
        reward_functions = reward._reward_fns

    elif "mujoco" in env_type:
        from test_mujoco_ant_reward import create_mujoco_ant_reward
        reward = create_mujoco_ant_reward(env_type=env_type)
        reward_functions = reward._reward_fns

    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    _,_, _, feature_names, _,_ = load_reward_ranges(env_type, range_ceiling=float('inf'),horizon=100, use_no_convo_baseline=no_convo_base_line)#note: horizon val doesn't matter here
    if no_convo_base_line:
        weights = np.load(f"active_learning_res/{env_type}_no_convo_baseline_o4-mini_True_prefs_feasible_weights.npy")
    else:
        weights = np.load(f"active_learning_res/{env_type}_o4-mini_True_prefs_feasible_weights.npy")
    learned_reward_weights = {name: weight for name, weight in zip(feature_names, weights)}

    # with open(save_path, 'r') as f:
    #     learned_reward_weights = json.load(f)
    # print(f"Loaded learned reward function from {save_path}")
    learned_reward = SumReward(reward_functions, learned_reward_weights)
    return learned_reward

def create_env_pandemic(config,reward_function,wrap_env):
    base_env = PandemicPolicyGymEnv(config)
    if not wrap_env:
        return base_env
    return RewardWrapper(base_env, env_name="pandemic", reward_function=reward_function)

def create_env_traffic(config,reward_function,wrap_env):
   
    create_env, env_name = make_create_env(
        params=config["flow_params_default"],
        reward_specification=config["reward_specification"],
        reward_fun=config["reward_fun"],
        reward_scale=config["reward_scale"],
    )
    base_env = create_env()
    if not wrap_env:
        return base_env
    return RewardWrapper(base_env, env_name="traffic", reward_function=reward_function, use_shaping=config.get("use_shaping", False), shaping_checkpoint=config.get("shaping_checkpoint", None))

def create_env_glucose(config, reward_function,wrap_env):
    base_env = SimglucoseEnv(config)
    if not wrap_env:
        return base_env
    return RewardWrapper(base_env, env_name="glucose", reward_function=reward_function)


def setup_glucose_env(config,wrap_env):
    # from utils.glucose_config import get_config
    # config = get_config()
    learned_reward = load_learned_reward("glucose")#"glucose_"+config["gt_reward_fn"] 
    return create_env_glucose(config,reward_function=learned_reward,wrap_env=wrap_env)

def setup_traffic_env(config,wrap_env):
    # from utils.traffic_config import get_config
    # config = get_config()
    learned_reward = load_learned_reward("traffic", no_convo_base_line=config["no_convo_base_line"])
    return create_env_traffic(config, reward_function=learned_reward,wrap_env=wrap_env)

def setup_pandemic_env(config,wrap_env):
    # from utils.pandemic_config import get_config
    # config = get_config()
    learned_reward = load_learned_reward("pandemic", no_convo_base_line=config["no_convo_base_line"])
    return create_env_pandemic(config,reward_function=learned_reward,wrap_env=wrap_env)


def setup_pandemic_env_w_gt_rew_set(config, wrap_env):
    # from utils.pandemic_config import get_config
    # config = get_config()
    from utils.pandemic_gt_rew_fns import TruePandemicRewardFunction
    gt_reward_set = TruePandemicRewardFunction(town_size = config["town_size"])
    if "gt_rew_i" not in config:
        raise ValueError("Must specify gt_rew_i in config for pandemic env with gt reward set")
    gt_reward_set.set_specific_reward(config.get("gt_rew_i", 0), flip_sign=config.get("flip_sign", False))
    return create_env_pandemic(config, reward_function=gt_reward_set, wrap_env=wrap_env)

def setup_traffic_env_pan_reward_fn(config, wrap_env):
    from utils.traffic_gt_rew_fns import PanEtAlTrueTrafficRewardFunction
    gt_reward_fn = PanEtAlTrueTrafficRewardFunction()
    return create_env_traffic(config, reward_function=gt_reward_fn, wrap_env=wrap_env)

def setup_traffic_env_w_gt_rew_set(config, wrap_env):
    # from utils.traffic_config import get_config
    # config = get_config()
    from utils.traffic_gt_rew_fns import TrueTrafficRewardFunction
    gt_reward_set = TrueTrafficRewardFunction()
    if "gt_rew_i" not in config:
        raise ValueError("Must specify gt_rew_i in config for traffic env with gt reward set")
    gt_reward_set.set_specific_reward(config.get("gt_rew_i", 0), flip_sign=config.get("flip_sign", False))
    return create_env_traffic(config, reward_function=gt_reward_set, wrap_env=wrap_env)


def create_env_mujoco(config, reward_function, wrap_env):
    """Create Mujoco Ant environment with custom reward wrapper."""
    env_name = config.get("env_name", "Ant-v4")
    # base_env = gym.make(env_name, terminate_when_unhealthy=False if "mujoco_backflip" in config["env_type"] else True)#terminate_when_unhealthy=False
    base_env = gym.make("Ant-v4")
    if not wrap_env:
        return base_env
    return RewardWrapper(base_env, env_name="mujoco", reward_function=reward_function)


def setup_mujoco_env(config, wrap_env):
    """Setup Mujoco environment with learned reward function."""
    learned_reward = load_learned_reward(config["env_type"])
    # print (learned_reward._weights)
    # print (len(learned_reward._weights))
    # assert False
    # for k in learned_reward._weights.keys():
    #     learned_reward._weights[k] = 0
    # learned_reward._weights ["ForwardDistanceReward"] = 1.0
    # learned_reward._weights ["HealthyZPositionCount"] = 1.0
    return create_env_mujoco(config, reward_function=learned_reward, wrap_env=wrap_env)


def setup_mujoco_env_w_gt_rew_set(config, wrap_env):
    """Setup Mujoco environment with ground truth reward set."""
    from utils.mujoco_gt_rew_fns import TrueMujocoAntRewardFunction
    #we only have one gt reward set for mujoco
    gt_reward_fn = TrueMujocoAntRewardFunction()


    return create_env_mujoco(config, reward_function=gt_reward_fn, wrap_env=wrap_env)

def setup_mujoco_env_w_data_generation_mode(config, wrap_env):
    """Setup Mujoco environment with data generation mode."""
    from utils.mujoco_gt_rew_fns import DataGenerationMujocoAntRewardFunction
    if "gt_rew_i" not in config:
        raise ValueError("Must specify gt_rew_i in config for mujoco env with data generation mode")
    reward_fn = DataGenerationMujocoAntRewardFunction(data_generation_mode=config["gt_rew_i"])
    return create_env_mujoco(config, reward_function=reward_fn, wrap_env=wrap_env)