
"""
MujocoObservation class for wrapping Mujoco Ant environment observations.
"""
import numpy as np
from typing import Optional


class MujocoObservation:
    """
    Observation wrapper for Mujoco Ant environment.
    
    The Ant environment observation includes:
    - Position coordinates (x, y, z) of the torso
    - Quaternion orientation
    - Joint angles and velocities
    - Contact forces
    """
    
    def __init__(
        self,
        obs: np.ndarray,
        env,
    ):
        """
        Initialize MujocoObservation.
        
        Args:
            obs: Full observation array from the environment
            env: Mujoco environment
        """
        self.obs = obs


        self.min_z_position = 0.2
        self.max_z_position = 1.0

        #time step of the environment
        self.dt = 0.05

        #valid range of z-position for the ant robot

        # For Mujoco, extract position info from the environment
        self.env = env
        #---------------------------------------------------------
        #self.position is a 15-dimensional vector that contains the position of the ant robot
        #| index | name | min | max | joint | type | unit |
        # | 0   | x-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
        # | 1   | y-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
        # | 2   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
        # | 3   | x-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        # | 4   | y-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        # | 5   | z-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        # | 6   | w-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        # | 7   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
        # | 8   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
        # | 9   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
        # | 10  | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
        # | 11  | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
        # | 12  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
        # | 13  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
        # | 14  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg) 
        self.position = self.env.unwrapped.data.qpos.flat.copy()
        #---------------------------------------------------------
        #self.velocity is a 13-dimensional vector that contains the velocity of the ant robot
        # | 0  | x-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
        # | 1  | y-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
        # | 2  | z-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
        # | 3  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
        # | 4  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
        # | 5  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
        # | 6  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
        # | 7  | angular velocity of the angle between front left links       | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
        # | 8  | angular velocity of angle between torso and front right link | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
        # | 9  | angular velocity of the angle between front right links      | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
        # | 10  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
        # | 11  | angular velocity of the angle between back left links        | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
        # | 12  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
        # | 13  |angular velocity of the angle between back right links        | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
        self.velocity = self.env.unwrapped.data.qvel.flat.copy()
        #---------------------------------------------------------
        #self.contact_forces is a 84-dimensional vector that contains the contact forces for all contact points of the ant robot
        self.contact_forces = self.env.unwrapped.data.cfrc_ext.flat.copy()

    def is_healthy (self):
        #returns True if the ant robot is healthy, meaning it is standing such that its z-position is in a valid range
        return self.min_z_position <= self.position[2] <= self.max_z_position
        
    @classmethod
    def from_raw_obs(cls, obs: np.ndarray, x_position: float, y_position: float, 
                     z_position: Optional[float] = None, contact_forces: Optional[np.ndarray] = None):
        """Create MujocoObservation from raw observation."""
        return cls(obs, x_position, y_position, z_position, contact_forces)
    
    def __getitem__(self, key):
        """Allow array-like indexing."""
        return self.obs[key]
    
    def __len__(self):
        """Return observation length."""
        return len(self.obs)

    def flatten(self) -> np.ndarray:
        """
        Concatenate and flatten every class-level NumPy array into a single
        1-D vector.  Missing (None) arrays are skipped automatically.

        Returns
        -------
        np.ndarray
            A one-dimensional array containing all entries, ordered exactly as
            listed below.
        """
        arrays = [
            self.position,
            self.velocity,
            self.contact_forces,
        ]

        # Remove any `None` placeholders that may appear
        arrays = [a for a in arrays if a is not None]

        # Flatten each array and concatenate along the first (only) dimension
        return np.concatenate([a.ravel() for a in arrays], axis=0)