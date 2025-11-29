from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np
from typing import List, Sequence

from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation
from generated_objectives.glucose_generated_objectives import RewardFunction
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation
from utils.mujoco_observation import MujocoObservation

import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.algorithms.ppo import PPO

class SumReward:
    def __init__(self, reward_functions: Sequence[RewardFunction], weights: dict[str, float]):
        """Initialize a sum reward function.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: Dictionary mapping reward function class names to their weights
        """
        if not all(rf.__class__.__name__ in weights for rf in reward_functions):
            raise ValueError("Each reward function's class name must have a corresponding weight")
        
        self._reward_fns = reward_functions
        self._weights = weights

    def calculate_reward(
        self, prev_obs, action: int, obs
    ) -> float:
        """Calculate the weighted sum of all reward functions.
        
        Args:
            prev_obs: Previous observation
            action: Action taken
            obs: Current observation
            
        Returns:
            float: Weighted sum of all reward values
        """
        total_reward = 0.0
        for reward_fn in self._reward_fns:
            reward = reward_fn.calculate_reward(prev_obs, action, obs)
            weight = self._weights[reward_fn.__class__.__name__]
            total_reward += weight * reward
        return total_reward



class RewardWrapper(Wrapper):
    def __init__(self, env, env_name, reward_function, use_shaping=False, shaping_checkpoint=None):
        super().__init__(env)
        self.env_name = env_name
        self.reward_function = reward_function #is of type SumReward
        print ("created reward wrapper...")
        self.ep_return = 0
        self.use_shaping = use_shaping
        self.shaping_checkpoint = shaping_checkpoint

        if self.use_shaping:
            assert self.shaping_checkpoint is not None, "Shaping checkpoint must be provided if use_shaping is True"
            # algo = Algorithm.from_checkpoint(self.shaping_checkpoint)

            # state = Algorithm._checkpoint_info_to_algorithm_state(get_checkpoint_info(self.shaping_checkpoint))
            # # Extract the config and override GPU settings
            # config = state["config"].copy()
            # config["num_gpus"] = 0
            # config["num_gpus_per_worker"] = 0
            # config["num_rollout_workers"] = 1

            # algo = PPO(config=config)
            # algo.restore(self.shaping_checkpoint)

            # self.shaping_function = algo.get_policy("default_policy").model

            from utils.traffic_gt_rew_fns import PanEtAlTrueTrafficRewardFunction

            assert env_name == "traffic", "Shaping is only supported for traffic environment"
            self.shaping_reward_function = PanEtAlTrueTrafficRewardFunction()

  
    def reset(self, **kwargs):
        print ("comptued return inside RewardWrapper:", self.ep_return)
        self.ep_return = 0
        if "pandemic" in self.env_name:
            obs, obs_np, info = self.env.reset_keep_obs_obj()
        elif "glucose" in self.env_name:
            obs_np, info = self.env.reset()
            obs = GlucoseObservation()
            obs.update_obs_with_sim_state(self.env)
        elif "traffic" in self.env_name:
            obs_np,info = self.env.reset()
            obs = TrafficObservation()
            obs.update_obs_with_sim_state(self.env, np.zeros(self.env.action_space.shape), {"crash":False})
        elif "mujoco" in self.env_name:
            obs_np, info = self.env.reset(**kwargs)
            # # For Mujoco, extract position info from the environment
            # x_position = self.env.unwrapped.data.qpos[0] if hasattr(self.env.unwrapped, 'data') else 0.0
            # y_position = self.env.unwrapped.data.qpos[1] if hasattr(self.env.unwrapped, 'data') else 0.0
            # z_position = self.env.unwrapped.data.qpos[2] if hasattr(self.env.unwrapped, 'data') else 0.5
            obs = MujocoObservation(obs_np, self.env)
            
        else:
            raise ValueError("Env not recognized")


        # Re-initialize each reward function in the SumReward object; some of these reward functions track state over the course of a trajectory
        # try:
        #     for reward_fn in self.reward_function._reward_fns:
        #             # Try calling reset() method if it exists, otherwise call __init__()
        #             if hasattr(reward_fn, 'reset') and callable(getattr(reward_fn, 'reset')):
        #                 reward_fn.__init__()
        # except AttributeError as e:
        #     #if the reward_function is not a SumReward object, then we don't need to re-initialize the reward functions
        #     pass
    
        #obs_np,info = self.env.reset(**kwargs)


        self.last_obs_np = obs_np
        self.last_obs = obs
        return obs_np,info

    def step(self, action):
        # Get the original step result
        #obs_obj = self.env.
        # obs, original_reward, terminated, truncated, info = self.env.step(action)
        if "pandemic" in self.env_name:
            obs, obs_np, original_reward, terminated, truncated, info = self.env.step_keep_obs_obj(action)
            
        elif "glucose" in self.env_name:
            obs_np, original_reward, terminated, truncated, info = self.env.step(action)

            obs = GlucoseObservation()
            obs.update_obs_with_sim_state(self.env)

            # obs.bg = np.array(obs.bg)
            # obs.insulin = np.array(obs.insulin)

            obs.bg = np.asarray(obs.bg)
            obs.insulin = np.asarray(obs.insulin)
            obs.cho = np.asarray(obs.cho)

            self.last_obs.bg = np.asarray(self.last_obs.bg)
            self.last_obs.insulin = np.asarray(self.last_obs.insulin)
            self.last_obs.cho = np.asarray(self.last_obs.cho)
        elif "traffic" in self.env_name:
            obs_np, original_reward, terminated, truncated, info = self.env.step(action)
            obs = TrafficObservation()
            obs.update_obs_with_sim_state(self.env, action, info)
        elif "mujoco" in self.env_name:
            obs_np, original_reward, terminated, truncated, info = self.env.step(action)
            # Extract position and contact force info from the environment
            # x_position = self.env.unwrapped.data.qpos[0] if hasattr(self.env.unwrapped, 'data') else 0.0
            # y_position = self.env.unwrapped.data.qpos[1] if hasattr(self.env.unwrapped, 'data') else 0.0
            # z_position = self.env.unwrapped.data.qpos[2] if hasattr(self.env.unwrapped, 'data') else 0.5
            # # Get contact forces if available
            # contact_forces = None
            # if hasattr(self.env.unwrapped, 'data') and hasattr(self.env.unwrapped.data, 'cfrc_ext'):
            #     contact_forces = self.env.unwrapped.data.cfrc_ext.flat.copy()
            # obs = MujocoObservation(obs_np, x_position, y_position, z_position, contact_forces)
            obs = MujocoObservation(obs_np, self.env)
        else:
            raise ValueError("Env not recognized")
        
        done = terminated or truncated

        # if "mujoco" in self.env_name:
        #     reward = original_reward
        # else:
        reward = self.reward_function.calculate_reward(self.last_obs, action, obs)

        # print ("original reward:", original_reward)
        # print ("reward from reward wrapper:", reward)
        # print ("\n")

        # print ("reward from reward wrapper:", reward)

        if type(reward) is np.ndarray:
            reward = reward.item()


        if self.use_shaping:
            # input_dict = {"obs": torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)}
            # self.shaping_function(input_dict)
            # value_s_prime = self.shaping_function.value_function().item()

            # input_dict = {"obs": torch.tensor(self.last_obs_np, dtype=torch.float32).unsqueeze(0)}
            # self.shaping_function(input_dict)
            # value_s = self.shaping_function.value_function().item()

            # value = value_s_prime - value_s

            # reward = reward + 0.99*value

            # print ("value:", value)
            # print ("reward:", reward)

            #use pan et al. true traffic reward function to compute shaping reward
            s_shaping_rewards = []
            s_prime_shaping_rewards = []
            for _ in range (100):
                #randomly sample an action
                random_action_1 = self.env.action_space.sample()
                s_shaping_reward = self.shaping_reward_function.calculate_reward(None, random_action_1, self.last_obs)
                s_shaping_rewards.append(s_shaping_reward)

                random_action_2 = self.env.action_space.sample()
                s_prime_shaping_reward = self.shaping_reward_function.calculate_reward(None, random_action_2, obs)
                s_prime_shaping_rewards.append(s_prime_shaping_reward)


            shaping_reward =  100*(np.mean(s_prime_shaping_rewards) - 0.99*np.mean(s_shaping_rewards))
            reward = reward + shaping_reward
            # print ("shaping reward:", shaping_reward)
            # print ("reward:", reward)


        if np.isnan(reward):
            print("Warning: NaN reward encountered. Setting to 0.")
            reward = 0.0
        
        # print ("reward:", reward)
        info["modified_reward"] = reward
        #no overwriting reward for now
        # reward = original_reward
        self.ep_return += original_reward
        
        # Store original reward in info for reference
        info["original_reward"] = original_reward
        # print ("overwriting reward...")
        self.last_obs_np = obs_np
        self.last_obs = obs

        # print ("new reward:",reward)
        # print ("default reward:",original_reward)
        return obs_np, reward, terminated, truncated, info

