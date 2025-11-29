import pandemic_simulator as ps
import torch
from pandemic_simulator.environment.interfaces import InfectionSummary
# from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from pandemic_simulator.environment.reward import (
    RewardFunctionFactory,
    RewardFunctionType,
    SumReward,
)
from pandemic_simulator.environment.simulator_opts import PandemicSimOpts
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.typing import AlgorithmConfigDict

from occupancy_measures.envs.pandemic_callbacks import PandemicCallbacks
from occupancy_measures.models.model_with_discriminator import (
    ModelWithDiscriminatorConfig,
)

def get_ppo_config(env_config, num_gpus, seed,num_rollout_workers=1):
    env_name = "pandemic_env"
    horizon = 192
    rollout_fragment_length = (horizon + 1) * 2
    lr = 0.0003
    gamma = 0.99
    kl_target = 0.01
    entropy_coeff = 0.01
    entropy_coeff_start = 0.1
    entropy_coeff_end = entropy_coeff
    entropy_coeff_horizon = 500000
    entropy_coeff_schedule = [
        [0, entropy_coeff_start],
        [entropy_coeff_horizon, entropy_coeff_end],
    ]
    train_batch_size = max(
        rollout_fragment_length * 10, rollout_fragment_length
    )
    sgd_minibatch_size = min(64, train_batch_size)
    batch_mode = "truncate_episodes"

    vf_loss_coeff = 0.5
    vf_clip_param = 20
    clip_param = 0.3
    gae_lambda = 0.95
    num_sgd_iter = 5
    grad_clip = 10
    callbacks = PandemicCallbacks

    obs_history_size = 3
    num_days_in_obs = 8

    # model
    width = 128  # Varied from 4 to 128
    depth = 2  # Varied from 1 to 2
    fcnet_hiddens = [width] * depth
    discriminator_width = 256
    discriminator_depth = 2
    discriminator_state_dim = 0
    use_action_for_disc = True
    use_history_for_disc = False
    time_dim = 1
    disc_history = num_days_in_obs  # how many days of history to store
    if not use_history_for_disc:
        disc_history = 1
    use_env_sim_step_states = False
    flattened_history_size = disc_history
    if use_env_sim_step_states:
        flattened_history_size *= obs_history_size
    if disc_history < num_days_in_obs:
        discriminator_state_dim = flattened_history_size * 13
    history_range = (flattened_history_size * -1, 0)
    vf_share_layers = True
    custom_model_config: ModelWithDiscriminatorConfig = {
        "discriminator_depth": discriminator_depth,
        "discriminator_width": discriminator_width,
        "discriminator_state_dim": discriminator_state_dim,
        "use_action_for_disc": use_action_for_disc,
        "use_history_for_disc": use_history_for_disc,
        "time_dim": time_dim,
        "history_range": history_range,
    }
    model_config = {
        "custom_model": "model_with_discriminator",
        "fcnet_hiddens": fcnet_hiddens,
        "custom_model_config": custom_model_config,
        "vf_share_layers": vf_share_layers,
    }

    config = PPOConfig().rl_module(_enable_rl_module_api=False)

    config_updates: AlgorithmConfigDict = {  # noqa: F841
        "env": env_name,
        "env_config": env_config,
        "disable_env_checking": True,
        "callbacks": callbacks,
        "num_rollout_workers": num_rollout_workers,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "lr": lr,
        "gamma": gamma,
        "lambda": gae_lambda,
        "kl_target": kl_target,
        "vf_loss_coeff": vf_loss_coeff,
        "vf_clip_param": vf_clip_param,
        "grad_clip": grad_clip,
        "entropy_coeff": entropy_coeff,
        "entropy_coeff_schedule": entropy_coeff_schedule,
        "clip_param": clip_param,
        "num_gpus": num_gpus,
        "rollout_fragment_length": rollout_fragment_length,
        "model": model_config,
        "batch_mode": batch_mode,
        "framework_str": "torch",
    }
    config.update_from_dict(config_updates)
    
    # RLlib 2.7 compatibility flags
    config._enable_rl_module_api = False
    config._enable_learner_api = False
    config.enable_connectors = False
    
    return config

def make_cfg_tiny(delta_hi, delta_lo, num_persons=500):
    location_configs = [
        ps.env.LocationConfig(ps.env.Home, num=150),
        ps.env.LocationConfig(
            ps.env.GroceryStore,
            num=2,
            num_assignees=5,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.Office, num=2, num_assignees=150, state_opts=dict(visitor_capacity=0)
        ),
        ps.env.LocationConfig(
            ps.env.School, num=10, num_assignees=2, state_opts=dict(visitor_capacity=30)
        ),
        ps.env.LocationConfig(
            ps.env.Hospital,
            num=1,
            num_assignees=15,
            state_opts=dict(patient_capacity=5),
        ),
        ps.env.LocationConfig(
            ps.env.RetailStore,
            num=2,
            num_assignees=5,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.HairSalon,
            num=2,
            num_assignees=3,
            state_opts=dict(visitor_capacity=5),
        ),
        ps.env.LocationConfig(
            ps.env.Restaurant,
            num=1,
            num_assignees=6,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.Bar, num=1, num_assignees=3, state_opts=dict(visitor_capacity=30)
        ),
    ]

    return ps.env.PandemicSimConfig(
        num_persons=num_persons,
        location_configs=location_configs,
        person_routine_assignment=ps.sh.DefaultPersonRoutineAssignment(),
        delta_start_lo=delta_lo,
        delta_start_hi=delta_hi,
    )

#these are the configs from medium town in https://github.com/SonyResearch/PandemicSimulator/blob/main/python/pandemic_simulator/script_helpers/sim_configs.py#L63
def make_cfg_med(delta_hi, delta_lo, num_persons=2000):
    location_configs = [
        ps.env.LocationConfig(ps.env.Home, num=600),
        ps.env.LocationConfig(
            ps.env.GroceryStore,
            num=8,
            num_assignees=5,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.Office, num=10, num_assignees=150, state_opts=dict(visitor_capacity=0)
        ),
        ps.env.LocationConfig(
            ps.env.School, num=20, num_assignees=4, state_opts=dict(visitor_capacity=30)
        ),
        ps.env.LocationConfig(
            ps.env.Hospital,
            num=2,
            num_assignees=30,
            state_opts=dict(patient_capacity=10),
        ),
        ps.env.LocationConfig(
            ps.env.RetailStore,
            num=8,
            num_assignees=5,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.HairSalon,
            num=8,
            num_assignees=3,
            state_opts=dict(visitor_capacity=5),
        ),
        ps.env.LocationConfig(
            ps.env.Restaurant,
            num=4,
            num_assignees=6,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.Bar, num=4, num_assignees=3, state_opts=dict(visitor_capacity=30)
        ),
    ]

    return ps.env.PandemicSimConfig(
        num_persons=num_persons,
        location_configs=location_configs,
        person_routine_assignment=ps.sh.DefaultPersonRoutineAssignment(),
        delta_start_lo=delta_lo,
        delta_start_hi=delta_hi,
    )

def make_reg():
    return ps.sh.austin_regulations


def get_config(town_size="tiny"):
    horizon = 192
    if town_size == "tiny":
        num_persons = 500
    elif town_size == "medium":
        num_persons = 2000
    else:
        raise NotImplementedError("Only tiny and medium town sizes are supported currently")

    delta_start_lo = 95
    delta_start_hi = 105
    # (INFECTION_SUMMARY_ABSOLUTE, POLITICAL, LOWER_STAGE, SMOOTH_STAGE_CHANGES)
    true_weights = [10, 10, 0.1, 0.01]  # true reward weights
    proxy_weights = [10, 0, 0.1, 0.01]  # proxy reward weights
    if town_size == "tiny":
        sim_config = make_cfg_tiny(delta_start_hi, delta_start_lo, num_persons)
    elif town_size == "medium":
        sim_config = make_cfg_med(delta_start_hi, delta_start_lo, num_persons)
    regulations = make_reg()
    done_fn = ps.env.DoneFunctionFactory.default(
        ps.env.DoneFunctionType.TIME_LIMIT, horizon=horizon
    )

    proxy_reward_fn = SumReward(
                reward_fns=[
                    RewardFunctionFactory.default(
                        RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
                        summary_type=InfectionSummary.CRITICAL,
                    ),
                    RewardFunctionFactory.default(
                        RewardFunctionType.POLITICAL,
                        summary_type=InfectionSummary.CRITICAL,
                    ),
                    RewardFunctionFactory.default(
                        RewardFunctionType.LOWER_STAGE, num_stages=len(regulations)
                    ),
                    RewardFunctionFactory.default(
                        RewardFunctionType.SMOOTH_STAGE_CHANGES,
                        num_stages=len(regulations),
                    ),
                ],
                weights=proxy_weights,
            )

    true_reward_fn = SumReward(
        reward_fns=[
            RewardFunctionFactory.default(
                RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
                summary_type=InfectionSummary.CRITICAL,
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.POLITICAL,
                summary_type=InfectionSummary.CRITICAL,
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.LOWER_STAGE, num_stages=len(regulations)
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.SMOOTH_STAGE_CHANGES,
                num_stages=len(regulations),
            ),
        ],
        weights=true_weights,
    )

    sim_opt = PandemicSimOpts(
        spontaneous_testing_rate=0.3
    )  # te
    reward_fun = "true"
    use_safe_policy_actions = False
    safe_policy = "S0-4-0"

    obs_history_size = 3
    num_days_in_obs = 8
    env_config = {
        "sim_config": sim_config,
        "sim_opts": sim_opt,
        "pandemic_regulations": regulations,
        "done_fn": done_fn,
        "reward_fun": reward_fun,
        "true_reward_fun": true_reward_fn,
        "proxy_reward_fun": proxy_reward_fn,
        "constrain": True,
        "four_start": False,
        "obs_history_size": obs_history_size,
        "num_days_in_obs": num_days_in_obs,
        "use_safe_policy_actions": use_safe_policy_actions,
        "safe_policy": safe_policy,
        "horizon": horizon,
        "town_size": town_size,
    }
    return env_config