"""
Ground truth reward functions for Mujoco Ant environment.
Re-implements the Mujoco Ant-v4/v5 reward function with decomposed components.
"""
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional
import numpy as np
from utils.mujoco_observation import MujocoObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        pass



class HealthyReward(RewardFunction):
    """Fixed reward for being in a healthy state."""
    
    def __init__(self, healthy_reward: float = 1.0):
        super().__init__()
        self.healthy_reward = healthy_reward
    
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # In Ant, healthy is defined by z-position being in a valid range
        # Default healthy range is [0.2, 1.0]
        if obs.position[2] is not None:
            is_healthy = prev_obs.min_z_position <= obs.position[2] <= prev_obs.max_z_position
            return self.healthy_reward if is_healthy else 0.0
        return self.healthy_reward


class ForwardReward(RewardFunction):
    """Reward for forward movement along the x-axis."""
    
    def __init__(self, forward_reward_weight: float = 1.0):
        super().__init__()
        self.forward_reward_weight = forward_reward_weight
    
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Forward reward is the change in x position
        # In Mujoco, this is typically (x_after - x_before) / dt
        # With dt = 0.05 (default frame_skip=5, timestep=0.01)
        # dt = 0.05
        x_velocity = (obs.position[0] - prev_obs.position[0]) / prev_obs.dt
        return self.forward_reward_weight * x_velocity

class ControlCost(RewardFunction):
    """Penalty for large control actions."""
    
    def __init__(self, ctrl_cost_weight: float = 0.5):
        super().__init__()
        self.ctrl_cost_weight = ctrl_cost_weight
    
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Control cost is negative (penalty) for large actions
        control_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        return -control_cost


class ContactCost(RewardFunction):
    """Penalty for large contact forces."""
    
    def __init__(self, contact_cost_weight: float = 5e-4):
        super().__init__()
        self.contact_cost_weight = contact_cost_weight
    
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Contact cost is penalty for external forces
        if obs.contact_forces is not None:
            contact_cost = self.contact_cost_weight * np.sum(
                np.square(np.clip(obs.contact_forces, -1, 1))
            )
            return -contact_cost
        return 0.0


class TrueMujocoAntRewardFunction(RewardFunction):
    """
    The true reward function used by the Mujoco Ant environment.
    This replicates the exact reward structure from Ant-v4/v5.
    
    reward = healthy_reward + forward_reward + ctrl_cost + contact_cost
    Note: ctrl_cost and contact_cost are already negative (penalties).
    """
    
    def __init__(
        self,
        healthy_reward: float = 1.0,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        include_shaping_terms = False,
    ):
        super().__init__()
        self.healthy_reward_fn = HealthyReward(healthy_reward)
        self.forward_reward_fn = ForwardReward(forward_reward_weight)
        self.ctrl_cost_fn = ControlCost(ctrl_cost_weight)
        self.contact_cost_fn = ContactCost(contact_cost_weight)
        self.include_shaping_terms = include_shaping_terms
    
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        healthy_reward = self.healthy_reward_fn.calculate_reward(prev_obs, action, obs)
        forward_reward = self.forward_reward_fn.calculate_reward(prev_obs, action, obs)

        if self.include_shaping_terms:
            ctrl_cost = self.ctrl_cost_fn.calculate_reward(prev_obs, action, obs)
            contact_cost = self.contact_cost_fn.calculate_reward(prev_obs, action, obs)
            
            return healthy_reward + forward_reward + ctrl_cost + contact_cost
        return healthy_reward + forward_reward

    def sample_calc_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        return [self.calculate_reward(prev_obs, action, obs)]

class DataGenerationMujocoAntRewardFunction(RewardFunction):
    """
    A reward function that generates data for the Mujoco Ant environment.
    """
    
    def __init__(self, data_generation_mode: int = 0):
        super().__init__()
        self.data_generation_mode = data_generation_mode
        if data_generation_mode == 0 or data_generation_mode == 1:
            self.reward_fn = TrueMujocoAntRewardFunction(include_shaping_terms=True)
    
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        if self.data_generation_mode == 0: # "gt_rew_fn":
            return self.reward_fn.calculate_reward(prev_obs, action, obs)
        elif self.data_generation_mode == 1: #"flipped_gt_rew_fn":
            return -self.reward_fn.calculate_reward(prev_obs, action, obs)
        elif self.data_generation_mode == 2: #"random":
            return 0
        else:
            raise ValueError(f"Invalid data generation mode: {self.data_generation_mode}")
