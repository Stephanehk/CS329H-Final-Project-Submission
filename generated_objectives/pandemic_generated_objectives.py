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

# Objective 0: Track sum of infected and critical cases
class SumInfectedAndCriticalReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        infected_i = sorted_infection_summary.index(InfectionSummary.INFECTED)
        critical_i = sorted_infection_summary.index(InfectionSummary.CRITICAL)

        # Calculate the sum over time of infected and critical cases
        current_sum = np.sum(obs.global_infection_summary[:, :, infected_i]) + np.sum(obs.global_infection_summary[:, :, critical_i])
        previous_sum = np.sum(prev_obs.global_infection_summary[:, :, infected_i]) + np.sum(prev_obs.global_infection_summary[:, :, critical_i])

        # Reward is the negative change in the sum, encouraging reduction
        return previous_sum - current_sum

# Objective 1: Track sum of critical cases (hospitalizations)
class SumCriticalCasesReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        critical_i = sorted_infection_summary.index(InfectionSummary.CRITICAL)

        # Calculate the sum over time of critical cases
        current_critical = np.sum(obs.global_infection_summary[:, :, critical_i])
        previous_critical = np.sum(prev_obs.global_infection_summary[:, :, critical_i])

        # Reward is the negative change in critical cases, encouraging reduction
        return previous_critical - current_critical

# Objective 2: Track sum of deaths due to COVID-19
class SumDeathsReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        dead_i = sorted_infection_summary.index(InfectionSummary.DEAD)

        # Calculate the sum over time of deaths
        current_deaths = np.sum(obs.global_infection_summary[:, :, dead_i])
        previous_deaths = np.sum(prev_obs.global_infection_summary[:, :, dead_i])

        # Reward is the negative change in deaths, encouraging reduction
        return previous_deaths - current_deaths

# Objective 3: Track sum of critical cases for ICU needs
class SumICUCriticalCasesReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        critical_i = sorted_infection_summary.index(InfectionSummary.CRITICAL)

        # Sum over time of critical cases to assess ICU needs
        current_critical = np.sum(obs.global_infection_summary[:, :, critical_i])
        previous_critical = np.sum(prev_obs.global_infection_summary[:, :, critical_i])

        # Reward is the negative change in critical cases
        return previous_critical - current_critical

# Objective 4: Track sum of recoveries
class SumRecoveriesReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        recovered_i = sorted_infection_summary.index(InfectionSummary.RECOVERED)

        # Calculate the sum over time of recoveries
        current_recovered = np.sum(obs.global_infection_summary[:, :, recovered_i])
        previous_recovered = np.sum(prev_obs.global_infection_summary[:, :, recovered_i])

        # Reward is the increase in recoveries
        return current_recovered - previous_recovered

# Objective 5: One-hot encoded stage rewards
class Stage0Reward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # Reward for being in stage 0
        return 1.0 if obs.stage[-1][-1].item() == 0 else 0.0

class Stage1Reward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # Reward for being in stage 1
        return 1.0 if obs.stage[-1][-1].item() == 1 else 0.0

class Stage2Reward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # Reward for being in stage 2
        return 1.0 if obs.stage[-1][-1].item() == 2 else 0.0

class Stage3Reward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # Reward for being in stage 3
        return 1.0 if obs.stage[-1][-1].item() == 3 else 0.0

class Stage4Reward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # Reward for being in stage 4
        return 1.0 if obs.stage[-1][-1].item() == 4 else 0.0

# Objective 6: Track infection above threshold compliance
class InfectionAboveThresholdReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # The infection_above_threshold variable is used to determine enforcement needs
        current_enforcement = np.sum(obs.infection_above_threshold)
        previous_enforcement = np.sum(prev_obs.infection_above_threshold)

        # Reward is the negative change in enforcement actions
        return previous_enforcement - current_enforcement

# Objective 7: Max hospitalizations (critical cases)
class MaxCriticalCasesReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        critical_i = sorted_infection_summary.index(InfectionSummary.CRITICAL)

        # Find max critical cases across time
        max_critical = np.max(obs.global_infection_summary[:, :, critical_i])

        # Reward is the negative max critical cases, encouraging reduction
        return -max_critical

# Objective 8: Min recovery levels
class MinRecoveryReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        recovered_i = sorted_infection_summary.index(InfectionSummary.RECOVERED)

        # Find min recovery level across time
        min_recovered = np.min(obs.global_infection_summary[:, :, recovered_i])

        # Reward is the min recovery level, encouraging increase
        return min_recovered

# Objective 9: Variance in infections
class InfectionVarianceReward(RewardFunction):
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        infected_i = sorted_infection_summary.index(InfectionSummary.INFECTED)

        # Calculate the variance of infections over time
        infection_variance = np.var(obs.global_infection_summary[:, :, infected_i])

        # Reward is the negative variance, encouraging stability
        return -infection_variance