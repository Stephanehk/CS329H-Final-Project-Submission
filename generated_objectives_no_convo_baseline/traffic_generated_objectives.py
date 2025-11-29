from abc import ABCMeta, abstractmethod
import numpy as np

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass

# 1. Reward based on maintaining speeds close to target velocity
class SpeedDifferenceFromTarget(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward based on how close the ego vehicle speeds are to the target velocity
        velocity_diff = obs.ego_speeds - obs.target_velocity
        return -np.mean(np.abs(velocity_diff))  # Negative reward for deviation

# 2. Reward based on minimizing the speed difference with leader vehicles
class SpeedDifferenceWithLeader(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward based on minimizing the speed difference with leader vehicles
        return -np.mean(np.abs(obs.leader_speed_diffs))  # Negative reward for speed difference

# 3. Reward based on maximizing the headway to the leader
class HeadwayToLeader(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward based on maximizing the headway to the leader
        return np.mean(obs.leader_headways) / (obs.max_length + 1e-8)  # Avoid division by zero

# 4. Reward based on minimizing the speed difference with follower vehicles
class SpeedDifferenceWithFollower(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward based on minimizing the speed difference with follower vehicles
        return -np.mean(np.abs(obs.follower_speed_diffs))  # Negative reward for speed difference

# 5. Continuous penalty based on collision status
class ContinuousCollisionPenalty(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Penalize based on the occurrence of a collision
        return -np.sum(obs.ego_vehicle_accels) if obs.fail else 0.0  # More severe penalties for worse accelerations if collision occurs
