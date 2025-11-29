from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type, Union
import numpy as np
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

# Implementation Notes:
# (1) To aggregate metrics over all vehicles, use operations over the corresponding arrays (e.g., np.mean(obs.ego_speeds) computes the mean normalized speed of all vehicles, np.mean(obs.follow_headways) computes the mean normalized headway of all follower vehicles to their leader, and so on)
# (2) action consists of a vector of bounded accelerations for each autonomous vehicle $i$.

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Input: the previous observation, the action taken, and the current observation
        # resulting from that action and the previous observation
        pass


# Ground Truth Reward Functions (matching the actual merge environment implementation)

class DesiredVelocityReward(RewardFunction):
    """Ground truth reward: Encourage proximity to desired velocity (cost1 in merge environment)."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # This mimics the desired_velocity function from flow.core.rewards
        # Convert normalized speeds back to actual speeds (assuming max_speed = 30)
        max_speed = 30.0  # Typical max speed in merge environment
        actual_speeds = obs.ego_speeds * max_speed
        
        # Filter out invalid speeds
        valid_speeds = actual_speeds[actual_speeds >= -100]
        if len(valid_speeds) == 0 or len(obs.rl_vehicles) == 0:
            return 0.0
        
        # Calculate cost based on deviation from target velocity
        max_cost = np.linalg.norm(np.array([obs.target_velocity] * len(valid_speeds)))
        cost = np.linalg.norm(valid_speeds - obs.target_velocity)
        
        # epsilon term to avoid division by zero
        eps = np.finfo(np.float32).eps
        
        return max(max_cost - cost, 0) / (max_cost + eps)


class HeadwayPenaltyReward(RewardFunction):
    """Ground truth reward: Penalize small time headways (cost2 in merge environment)."""
    
    def __init__(self, min_time_headway: float = 1.0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.min_time_headway = min_time_headway
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # This mimics the headway penalty calculation in merge environment
        # Convert normalized speeds and headways back to actual values
        max_speed = 30.0
        max_length = 1000.0  # Typical network length
        
        actual_speeds = obs.ego_speeds * max_speed
        actual_headways = obs.lead_headways * max_length
        
        cost2 = 0.0
        for i in range(len(actual_speeds)):
            if actual_speeds[i] > 0:
                # Calculate time headway
                time_headway = max(actual_headways[i] / actual_speeds[i], 0)
                # Penalize if time headway is less than minimum
                cost2 += min((time_headway - self.min_time_headway) / self.min_time_headway, 0)
        
        return cost2
        


class AccelerationPenaltyReward(RewardFunction):
    """Ground truth reward: Penalize large accelerations (cost3 in merge environment)."""
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # This mimics the acceleration penalty calculation in merge environment
        if action is not None and len(action) > 0:
            mean_actions = np.mean(np.abs(np.array(action)))
            accel_threshold = 0
            
            if mean_actions > accel_threshold:
                return accel_threshold - mean_actions
        
        return 0.0


class TrueTrafficRewardFunction(RewardFunction):
    """Ground truth reward function matching the actual merge environment implementation."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.desired_velocity = DesiredVelocityReward()
        self.headway_penalty = HeadwayPenaltyReward()
        self.accel_penalty = AccelerationPenaltyReward()
        # Weights matching the merge environment: eta1, eta2, eta3 = 1.00, 0.10, 1
        self.eta1, self.eta2, self.eta3 = 1.00, 0.10, 1.00
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Calculate individual components
        cost1 = self.desired_velocity.calculate_reward(prev_obs, action, obs)
        cost2 = self.headway_penalty.calculate_reward(prev_obs, action, obs)
        cost3 = self.accel_penalty.calculate_reward(prev_obs, action, obs)
        
        # Combine with weights and apply max(0, ...) as in merge environment
        reward = self.eta1 * cost1 + self.eta2 * cost2 + self.eta3 * cost3
        return reward



# Proxy Reward Functions (Simplified/Approximate versions)

class SpeedReward(RewardFunction):
    """Proxy reward: Simple reward based on vehicle speeds."""
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Simple reward based on average speed
        avg_speed = np.mean(obs.ego_speeds)
        return avg_speed * 50


class HeadwayReward(RewardFunction):
    """Proxy reward: Simple reward based on following distances."""
    
    def __init__(self, target_headway: float = 0.2, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_headway = target_headway
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Reward for maintaining target headway
        headway_deviation = -np.mean(np.abs(obs.lead_headways - self.target_headway))
        return headway_deviation * 25


class ActionSmoothnessReward(RewardFunction):
    """Proxy reward: Penalize jerky actions."""
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Simple penalty for action magnitude
        if action is not None and len(action) > 0:
            return -np.mean(np.abs(action)) * 10
        return 0.0


class CollisionAvoidanceReward(RewardFunction):
    """Proxy reward: Simple collision avoidance based on headways."""
    
    def __init__(self, danger_threshold: float = 0.05, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.danger_threshold = danger_threshold
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Penalize dangerously close headways
        dangerous_headways = np.sum(obs.lead_headways < self.danger_threshold)
        return -dangerous_headways * 20


class FlowReward(RewardFunction):
    """Proxy reward: Simple flow-based reward."""
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Reward for high average speeds and low speed variation
        avg_speed = np.mean(obs.ego_speeds)
        speed_variation = np.std(obs.ego_speeds)
        
        flow_reward = avg_speed * 30 - speed_variation * 20
        return flow_reward


class ProxyTrafficRewardFunction(RewardFunction):
    """Combined proxy reward function for traffic optimization."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.speed = SpeedReward()
        self.headway = HeadwayReward()
        self.smoothness = ActionSmoothnessReward()
        self.collision_avoidance = CollisionAvoidanceReward()
        self.flow = FlowReward()
    
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        # Combine all proxy objectives with weights
        speed_reward = self.speed.calculate_reward(prev_obs, action, obs) * 0.4
        headway_reward = self.headway.calculate_reward(prev_obs, action, obs) * 0.25
        smoothness_reward = self.smoothness.calculate_reward(prev_obs, action, obs) * 0.15
        collision_reward = self.collision_avoidance.calculate_reward(prev_obs, action, obs) * 0.1
        flow_reward = self.flow.calculate_reward(prev_obs, action, obs) * 0.1
        
        return speed_reward + headway_reward + smoothness_reward + collision_reward + flow_reward
