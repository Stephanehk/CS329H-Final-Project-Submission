from typing import Optional, Tuple
import importlib
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility

from flow.core.rewards import REWARD_REGISTRY


class ProxyRewardEnv(gym.Wrapper):
    """
    Wraps the given environment in a proxy reward wrapper function that changes the reward provided to the environment.

    Params
    ------
    env: the Flow environment object that will be wrapped with a proxy
    reward_specification: a dict of reward_pairs with str keys corresponding to reward type and float values corresponding to the weight of that reward.
                            The proxy reward is a linear combination of all the reward functions specified.
    reward_fun: which reward function to use, proxy or true
    path: where to save the flow rendering
    reward_scale: Optional, by how much to scale the rewards
    *args: environment args
    **kwargs: envrionment kwargs
    """
    metadata = {'render_modes': ['human', 'rgb_array']}


    def __init__(
        self,
        module,
        mod_name,
        env_params,
        sim_params,
        network,
        simulator,
        reward_specification,
        reward_fun,
        path,
        reward_scale=1,
        use_safe_policy_actions=False,
    ):
        cls = getattr(importlib.import_module(module), mod_name)
        self.env = cls(
            env_params,
            sim_params,
            network,
            simulator,
            path=path,
            use_safe_policy_actions=use_safe_policy_actions,
        )

        super().__init__(self.env)

        self.reward_scale = reward_scale

        if reward_specification is not None:
            self.use_new_spec = True
            self.reward_fun = reward_fun

            self.true_reward_specification = []
            self.proxy_reward_specification = []

            if reward_specification["true"] is not None:
                self.use_original_true = False
                for name, eta in reward_specification["true"]:
                    assert name in REWARD_REGISTRY
                    self.true_reward_specification.append((REWARD_REGISTRY[name], eta))
            else:
                self.use_original_true = True
                self.original_rew_func = getattr(self.env, "compute_reward")

            for name, eta in reward_specification["proxy"]:
                assert name in REWARD_REGISTRY
                self.proxy_reward_specification.append((REWARD_REGISTRY[name], eta))

            if self.reward_fun == "proxy":

                def proxy_reward(rl_actions, **kwargs):
                    return self._proxy(
                        self.proxy_reward_specification, rl_actions, **kwargs
                    )

                setattr(self.env, "compute_reward", proxy_reward)
            elif not self.use_original_true:

                def proxy_reward(rl_actions, **kwargs):
                    return self._proxy(
                        self.true_reward_specification, rl_actions, **kwargs
                    )

                setattr(self.env, "compute_reward", proxy_reward)
        else:
            self.use_new_spec = False

    def _proxy(self, reward_specification, rl_actions, **kwargs):
        vel = np.array(self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))
        if any(vel < -100) or kwargs["fail"]:
            return 0
        rew = 0
        for fn, eta in reward_specification:
            rew += eta * fn(self.env, rl_actions)
        return rew

    def __getattr__(self, attr):
        return self.env.__getattribute__(attr)

    def step(self, rl_actions):
        next_observation, reward, terminated, truncated, infos = self.env.step(rl_actions)
        if self.use_new_spec:
            if self.reward_fun == "proxy":
                infos["proxy_reward"] = reward
                if not self.use_original_true:
                    infos["true_reward"] = self._proxy(
                        self.true_reward_specification, rl_actions, fail=infos["crash"]
                    )
                else:
                    infos["true_reward"] = self.original_rew_func(
                        rl_actions, fail=infos["crash"]
                    )
            else:
                infos["proxy_reward"] = self._proxy(
                    self.proxy_reward_specification, rl_actions, fail=infos["crash"]
                )
                infos["true_reward"] = reward
        else:
            infos["proxy_reward"] = infos["true_reward"] = reward
        return next_observation, reward * self.reward_scale, terminated, truncated, infos 
