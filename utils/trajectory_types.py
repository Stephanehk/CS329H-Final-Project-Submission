from typing import Any

class TrajectoryStep:
    """Class to represent a single step in a trajectory."""
    def __init__(self, 
                 obs: Any,  # Can be PandemicObservation or other observation types
                 action: int,
                 next_obs: Any,  # Can be PandemicObservation or other observation types
                 true_reward: float,
                 proxy_reward:float,
                 done: bool):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.true_reward = true_reward
        self.proxy_reward = proxy_reward
        self.done = done 