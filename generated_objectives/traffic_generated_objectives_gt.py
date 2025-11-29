from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any

from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: TrafficObservation, action, obs: TrafficObservation
    ) -> float:
        pass


class Crash(RewardFunction):
    """
    Penalty based on the occurrence of collisions or failures.
    """
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Check if a collision occurred in the current observation
        collision_occurred = obs.fail
        # Penalize if a collision occurred
        reward = -1.0 if collision_occurred else 0.0
        return reward

#implement one class for each cost computed in TrueTrafficRewardFunction
class MeanVelocity(RewardFunction):
    """
    Reward based on the average speed of all vehicles.
    """
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate the average speed of all vehicles
        return float(np.mean(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0
   
class MinVelocity(RewardFunction):
    """
    Reward based on the minimum speed of all vehicles.
    """
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        # Calculate the minimum speed of all vehicles
        return float(np.min(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0

class SquaredSpeedAboveMax(RewardFunction):
    """
    Penalty based on the squared amount by which vehicle speeds exceed the maximum speed.
    """
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        speed = obs.all_vehicle_speeds
        if speed.size == 0:
            return 0.0
        else:
            excess = np.maximum(np.abs(speed) - obs.target_velocity, 0.0)  # overshoot per component
            penalty = np.sum(excess ** 2)  # squared penalty
            return penalty

class FuelProxy(RewardFunction):
    """
    Penalty based on a proxy for fuel consumption.
    """
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        

        g=9.80665       # [m/s^2] standard gravity (CODATA/NIST)
        rho=1.225         # [kg/m^3] standard air density at 15°C, sea level
        C_r=0.010         # [-] rolling resistance coeff. for passenger cars on asphalt
        CdA_over_m=4.7e-4 # [m^2/kg] fleet-avg. drag area per mass ≈ (Cd*A)/m
        sin_theta=0.0      # [-] road grade ≈ sin(theta); use 0.05 for ~5% uphill if known

        a = np.asarray(action, dtype=float)
        v = np.asarray(obs.ego_speeds, dtype=float)

        # Only positive acceleration contributes to *fuel* via the inertia term; engine doesn’t supply
        # negative a (that’s braking). This mirrors MOVES/VSP usage where operating modes tie emissions to power demand.
        inertial = np.clip(a, 0.0, None) * v

        # Rolling resistance power per mass: C_r * g * v
        rolling  = C_r * g * v

        # Aerodynamic power per mass: (1/(2m)) * rho * Cd * A * v^3  =>  0.5 * rho * (CdA/m) * v^3
        aero     = 0.5 * rho * CdA_over_m * (v ** 3)

        # Grade power per mass: g * sin(theta) * v  (0 by default unless you pass sin_theta)
        grade    = g * sin_theta * v

        # Total per-mass tractive power proxy (≈ VSP). Sum across vehicles for this timestep.
        return float(np.sum(inertial + rolling + aero + grade))

class BreakPenalty(RewardFunction):
    """
    Penalty based on the frequency of hard braking events.
    """
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        a = np.asarray(action, dtype=float)
        v = np.asarray(obs.ego_speeds, dtype=float)
        neg = np.clip(-a, 0.0, None)  # take only braking (make positive)
        return float(np.sum(neg * v))

class HeadwayPenalty(RewardFunction):
    """
    Penalty based on the average headway distances between vehicles.
    """
    def calculate_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation) -> float:
        t_min = 1  # smallest acceptable time headway
        cost = 0.0
        for speed, headway in zip(obs.ego_speeds, obs.leader_headways):
            if speed > 0:  # avoid divide-by-zero
                t_headway = max(headway / speed, 0.0)
                cost += min((t_headway - t_min) / t_min, 0.0)
        return cost    