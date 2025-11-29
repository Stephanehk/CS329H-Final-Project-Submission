from abc import ABCMeta, abstractmethod
import numpy as np
from pandemic_simulator.environment.interfaces import PandemicObservation, sorted_infection_summary, InfectionSummary

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        pass

class InfectedStat(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        infected_i = sorted_infection_summary.index(InfectionSummary.INFECTED)
        # Calculate the sum over time of infected and critical cases
        return obs.global_infection_summary[-1, -1, infected_i]

class DeadStat(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        dead_i = sorted_infection_summary.index(InfectionSummary.DEAD)
        # Calculate the sum over time of deaths
        return obs.global_infection_summary[-1, -1, dead_i]

class CriticalStat(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        critical_i = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        # Calculate the sum over time of critical cases
        return obs.global_infection_summary[-1, -1, critical_i]

class RecoveredStat(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        recovered_i = sorted_infection_summary.index(InfectionSummary.RECOVERED)
        # Calculate the sum over time of recoveries
        return obs.global_infection_summary[-1, -1, recovered_i]

class InfectionAboveThresholdReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # The infection_above_threshold variable is used to determine enforcement needs
        return obs.infection_above_threshold[-1].item()

class Stage(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # Reward for being in the current stage
        return obs.stage[-1][-1].item()

