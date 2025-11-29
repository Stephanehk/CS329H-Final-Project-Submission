from abc import ABCMeta, abstractmethod
import numpy as np
from utils.mujoco_observation import MujocoObservation

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        pass

# Class for calculating forward movement reward based on change in x-coordinate
class ForwardDistanceReward(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Calculate the difference in x-coordinate of the torso
        return (obs.position[0] - prev_obs.position[0])

# Class for measuring progress towards the goal based on change in x-coordinate
class ProgressTowardsGoalReward(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Reward is based on the change in x-coordinate
        return (obs.position[0] - prev_obs.position[0])

# Class for counting healthy z-coordinate instances
class HealthyZPositionCount(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Check if the z-coordinate is within the healthy range
        return 1.0 if obs.is_healthy() else 0.0

# Class for calculating total torque exerted by joints
class TotalTorqueExerted(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Calculate the total torque exerted by each joint
        return np.sum(np.abs(action))

# Class for calculating mean of joint angular velocities
class MeanJointAngularVelocities(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Calculate the mean of joint angular velocities
        joint_velocities = obs.velocity[6:14]
        return np.mean(np.abs(joint_velocities))

# Class for counting torque application frequency, reflecting continuous torque usage
class TorqueApplicationFrequency(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Measure torque as a continuous value without fixed thresholds
        torque_used = np.sum(np.abs(action))
        return torque_used  # Return total torque used as a continuous measure

# Class for measuring maximum joint angle deviation from expected angles
class MaxJointAngleDeviation(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Calculate maximum deviation from expected joint angles (assumed zero)
        expected_angles = np.zeros_like(obs.position[7:15])
        deviations = np.abs(obs.position[7:15] - expected_angles)
        return np.max(deviations)

# Class for counting significant trajectory changes based on distance moved
class TrajectoryChangeCount(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Calculate distance moved for trajectory changes
        changes = np.sqrt((obs.position[0] - prev_obs.position[0])**2 + (obs.position[1] - prev_obs.position[1])**2)
        return changes  # Return the continuous change instead of a binary condition

# Class for measuring orientation stability based on changes in angles
class OrientationStability(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Track changes in orientation angles and return as a negative measure
        orientation_changes = np.sum(np.abs(obs.position[3:6] - prev_obs.position[3:6]))
        return -orientation_changes  # More change should lead to a lower reward

# Class for calculating performance index based on joint velocities and torque
class JointPerformanceIndex(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Calculate performance index based on the ratio of angular velocities to torque exerted
        joint_velocities = np.abs(obs.velocity[6:14])
        torque_exerted = np.abs(action)
        performance_index = np.mean(joint_velocities / (torque_exerted + 1e-6))  # Avoid division by zero
        return performance_index

# Class for counting movement oscillations based on distance moved
class MovementOscillationCount(RewardFunction):
    def calculate_reward(self, prev_obs: MujocoObservation, action: np.ndarray, obs: MujocoObservation) -> float:
        # Track oscillations in movement patterns based on distance moved
        oscillations = np.sqrt((obs.position[0] - prev_obs.position[0])**2 + (obs.position[1] - prev_obs.position[1])**2)
        return oscillations  # Return the continuous distance instead of a binary condition