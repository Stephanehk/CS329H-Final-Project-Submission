from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any
from utils.mujoco_observation import MujocoObservation

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        pass

# Class to reward based on the z-coordinate's distance from the healthy range
class ZCoordinateHealthDistance(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Calculate the distance from the healthy z-coordinate range
        z = obs.position[2]
        healthy_min = 0.2  # Example value for healthy minimum z-coordinate
        healthy_max = 1.0  # Example value for healthy maximum z-coordinate
        
        # Continuous penalty based on how far z is from the healthy range
        if z < healthy_min:
            return -(healthy_min - z)  # Penalize for being below healthy range
        elif z > healthy_max:
            return -(z - healthy_max)  # Penalize for being above healthy range
        else:
            return 1.0  # Reward for being within the healthy range

# Class to reward full rotation based on angular velocities
class FullRotationAngularVelocity(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Calculate total rotation based on angular velocities
        angular_velocity_sum = np.sum(np.abs(obs.velocity[3:6])) * obs.dt
        return angular_velocity_sum / (2 * np.pi)  # Normalizing by 360 degrees

# Class to reward based on final x and y positions for landing location
class LandingLocationAccuracy(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Reward based on final x and y positions for landing accuracy
        x, y = obs.position[0], obs.position[1]
        target_x, target_y = 0.0, 0.0  # Assuming target landing at origin
        distance = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        return -distance  # Negative reward for further distance from target

# Class to reward stability duration post-landing
class StabilityDurationPostLanding(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Reward for each stable time step after landing
        return obs.dt if obs.is_healthy() else 0.0

# Class to track the duration of the flip
class FlipDuration(RewardFunction):
    def __init__(self):
        super().__init__()
        self.start_flip = False
        self.flip_time = 0.0

    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Track the duration of the flip based on z-coordinate changes
        if not self.start_flip and obs.position[2] > prev_obs.position[2]:
            self.start_flip = True
        if self.start_flip:
            self.flip_time += obs.dt
            if obs.is_healthy():
                return self.flip_time
        return 0.0

# Class to track the maximum z-coordinate value during the flip
class MaximumZCoordinateValue(RewardFunction):
    def __init__(self):
        super().__init__()
        self.max_z = -np.inf

    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Update max z-coordinate during the flip
        self.max_z = max(self.max_z, obs.position[2])
        return self.max_z

# Class to reward the mean angular velocity of the torso
class MeanAngularVelocityTorso(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Mean of torso angular velocities
        angular_velocity_mean = np.mean(np.abs(obs.velocity[3:6]))  
        return angular_velocity_mean

# Class to detect collisions based on contact forces
class CollisionImpactAssessment(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Calculate the total contact force to provide a continuous penalty
        total_contact_force = np.sum(np.abs(obs.contact_forces))
        return -total_contact_force  # Negative reward for higher impact forces

# Class to assess smoothness of joint movements
class JointMovementSmoothness(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Calculate mean squared difference of joint angular velocities
        joint_angular_velocity_diff = np.array(obs.velocity[6:]) - np.array(prev_obs.velocity[6:])
        smoothness = -np.mean(joint_angular_velocity_diff**2)  # Negative mean squared difference
        return smoothness

# Class to track maximum contact force during landing
class MaximumContactForceDuringLanding(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Track maximum contact force during landing
        max_contact_force = np.max(np.abs(obs.contact_forces))
        return -max_contact_force  # Negative reward for higher impact forces

# Class to estimate energy utilization related to z-coordinate changes
class EnergyUtilizationZCoordinateChanges(RewardFunction):
    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Calculate energy utilization based on z-coordinate changes
        z_change = np.abs(obs.position[2] - prev_obs.position[2])
        mean_angular_velocity = np.mean(np.abs(obs.velocity[3:6]))
        return -mean_angular_velocity * z_change  # Negative reward for higher energy use

# Class to track performance variance consistency across trials
class PerformanceVarianceConsistencyAcrossTrials(RewardFunction):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def calculate_reward(
        self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation
    ) -> float:
        # Track completion time as a metric
        completion_time = obs.dt  
        self.metrics.append(completion_time)
        if len(self.metrics) < 2:
            return 0.0
        variance = np.var(self.metrics)
        return -variance  # Negative reward for higher variance