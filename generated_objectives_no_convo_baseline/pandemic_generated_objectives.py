from abc import ABCMeta, abstractmethod
import numpy as np
from pandemic_simulator.environment.interfaces import PandemicObservation, sorted_infection_summary, InfectionSummary

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass

# Reward functions for each stage level
class StageLevel0Reward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward for being in Stage 0 - no restrictions
        return 1.0 if obs.stage[-1, -1].item() == 0 else 0.0

class StageLevel1Reward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward for being in Stage 1 - hygiene encouraged, limited gatherings
        return 1.0 if obs.stage[-1, -1].item() == 1 else 0.0

class StageLevel2Reward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward for being in Stage 2 - hygiene + masks required, limited gatherings
        return 1.0 if obs.stage[-1, -1].item() == 2 else 0.0

class StageLevel3Reward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward for being in Stage 3 - stricter measures, gatherings banned
        return 1.0 if obs.stage[-1, -1].item() == 3 else 0.0

class StageLevel4Reward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Reward for being in Stage 4 - most severe restrictions
        return 1.0 if obs.stage[-1, -1].item() == 4 else 0.0

# Reward function for infection control
class InfectionControlReward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Calculate reward based on change in infection rates
        infected_idx = sorted_infection_summary.index(InfectionSummary.INFECTED)
        critical_idx = sorted_infection_summary.index(InfectionSummary.CRITICAL)

        prev_infected = prev_obs.global_infection_summary[-1, -1, infected_idx]
        prev_critical = prev_obs.global_infection_summary[-1, -1, critical_idx]

        current_infected = obs.global_infection_summary[-1, -1, infected_idx]
        current_critical = obs.global_infection_summary[-1, -1, critical_idx]

        # Reward is positive if the sum of INFECTED and CRITICAL decreases
        prev_total_cases = prev_infected + prev_critical
        current_total_cases = current_infected + current_critical

        # Calculate reward based on the change in total cases
        reward = prev_total_cases - current_total_cases

        # Prevent division by zero; no need for arbitrary scaling
        return reward

# Reward function for minimizing deaths
class MinimizeDeathsReward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Calculate reward based on change in death rates
        dead_idx = sorted_infection_summary.index(InfectionSummary.DEAD)

        prev_dead = prev_obs.global_infection_summary[-1, -1, dead_idx]
        current_dead = obs.global_infection_summary[-1, -1, dead_idx]

        # Penalize increase in deaths; reward is negative for increasing deaths
        reward = -(current_dead - prev_dead)

        # Continuous penalty based on the change in deaths
        return reward

# Reward function for recovery rates
class RecoveryRateReward(RewardFunction):
    def calculate_reward(self, prev_obs, action, obs):
        # Calculate reward based on change in recovery rates
        recovered_idx = sorted_infection_summary.index(InfectionSummary.RECOVERED)

        prev_recovered = prev_obs.global_infection_summary[-1, -1, recovered_idx]
        current_recovered = obs.global_infection_summary[-1, -1, recovered_idx]

        # Reward increase in recoveries
        reward = current_recovered - prev_recovered

        return reward