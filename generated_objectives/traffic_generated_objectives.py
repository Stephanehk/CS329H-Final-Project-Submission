from abc import ABCMeta
import numpy as np
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        pass

# Count of Failures
class FailureCountReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Return 1 if there's a failure, else 0
        return float(obs.fail)

# Mean of Ego Speeds
class MeanEgoSpeedsReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate and return the mean speed of ego vehicles
        return np.mean(obs.ego_speeds)

# Mean of Headways
class MeanHeadwaysReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate the mean of leader and follower headways
        leader_mean = np.mean(obs.leader_headways)
        follower_mean = np.mean(obs.follower_headways)
        return (leader_mean + follower_mean) / 2

# Count of Speed Exceedances
class SpeedExceedanceReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Count instances where any vehicle exceeds the target velocity
        exceedances = np.sum(obs.all_vehicle_speeds > obs.target_velocity)
        return float(exceedances)

# Mean of Ego Vehicle Accelerations
class MeanEgoVehicleAccelerationsReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate and return the mean of ego vehicle accelerations
        return np.mean(obs.ego_vehicle_accels)

# Mean of All Vehicle Speeds
class MeanAllVehicleSpeedsReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate and return the mean speed of all vehicles
        return np.mean(obs.all_vehicle_speeds)

# Safety Headway and Failure Count
class SafetyHeadwayFailureReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Penalize for failures and reward for safe headways
        failure_penalty = -float(obs.fail)
        mean_leader_headway = np.mean(obs.leader_headways)
        mean_follower_headway = np.mean(obs.follower_headways)
        # Return combined reward: negative for failures, positive for safe headways
        return failure_penalty + (mean_leader_headway + mean_follower_headway) / 2

# Minimum of Ego Speeds
class MinimumEgoSpeedsReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate and return the minimum speed among ego vehicles
        return np.min(obs.ego_speeds)

# Minimum of Headways
class MinimumHeadwaysReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate and return the minimum of leader and follower headways
        min_leader_headway = np.min(obs.leader_headways)
        min_follower_headway = np.min(obs.follower_headways)
        return min(min_leader_headway, min_follower_headway)

# Maximum of All Vehicle Speeds
class MaximumAllVehicleSpeedsReward(RewardFunction):
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate and return the maximum speed among all vehicles
        return np.max(obs.all_vehicle_speeds)