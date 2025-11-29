from abc import ABCMeta, abstractmethod
from typing import Any
import numpy as np
from pandemic.python.pandemic_simulator.environment.interfaces import (
    PandemicObservation,
    InfectionSummary,
    sorted_infection_summary
)

#Email authors to ask why they included the political cost component
#Missing reward component that takes into account infection threshold
#possible gt-reward fn = # of deaths + # of critical patients + lockdown-stage as 5 different reward features

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: float, obs: PandemicObservation
    ) -> float:
        pass


class LowerStagePreferenceReward(RewardFunction):
    """Rewards lower regulation stages."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        _num_stages = 4
        stage_rewards = np.arange(0, _num_stages) ** 1.5
        stage_rewards = stage_rewards / np.max(stage_rewards)
        return -0.1 * float(stage_rewards[action])

class SmoothStageChangesPreferenceReward(RewardFunction):
    """Penalizes large changes in regulation stages."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        return -0.02 * float(np.abs(obs.stage - prev_obs.stage).mean())

class CriticalCasesAbsoluteReward(RewardFunction):
    """Returns a negative reward proportional to the absolute number of critical cases."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        # print (sorted_infection_summary.index(InfectionSummary.CRITICAL))
        # print (InfectionSummary.CRITICAL)
        # assert False
        _critical_idx = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        return -10.0 * np.mean(obs.global_infection_summary[..., _critical_idx])

class PoliticalStageReward(RewardFunction):
    """Penalizes raising the stage when infection rates are low."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        
        infection_rate = (
            np.mean(prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.INFECTED)])
            + np.mean(prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.CRITICAL)])
            + np.mean(prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.DEAD)])
        )
        stage = int(obs.stage[-1][-1].item() - prev_obs.stage[-1][-1].item())
        assert stage in [-1, 0, 1]
        raise_stage_penalty = min(infection_rate - 0.005, 0) / 0.005
        return -10.0 * ((stage == 1) * raise_stage_penalty**2)


class PanEtAlTruePandemicRewardFunction(RewardFunction):
    """The true reward function used by the pandemic simulator environment."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        critical_reward = CriticalCasesAbsoluteReward().calculate_reward(prev_obs, action, obs)
        political_reward = PoliticalStageReward().calculate_reward(prev_obs, action, obs)
        lower_stage_reward = LowerStagePreferenceReward().calculate_reward(prev_obs, action, obs)
        smooth_changes_reward = SmoothStageChangesPreferenceReward().calculate_reward(prev_obs, action, obs)
        
        return critical_reward + political_reward + lower_stage_reward + smooth_changes_reward

class ProxyPandemicRewardFunction(RewardFunction):
    """The true reward function used by the pandemic simulator environment."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        critical_reward = CriticalCasesAbsoluteReward().calculate_reward(prev_obs, action, obs)
        # political_reward = PoliticalStageReward().calculate_reward(prev_obs, action, obs)
        lower_stage_reward = LowerStagePreferenceReward().calculate_reward(prev_obs, action, obs)
        smooth_changes_reward = SmoothStageChangesPreferenceReward().calculate_reward(prev_obs, action, obs)
        
        return critical_reward + lower_stage_reward + smooth_changes_reward


class TruePandemicRewardFunction(RewardFunction):
    """The true reward function used by the pandemic simulator environment."""

    def __init__(self, town_size="tiny", n=2):
        np.random.seed(42)

        healthy_extreme = [-100, -100, 0, 0, 0, 0, 0]#minimize deaths and critical cases, with no care for lockdown regulations
        prevent_deaths = [0, -100, 0, 0, 0, 0,0]#minimize deaths only
        prevent_lockdown = [0,0, 1, 0, 0, 0, 0]#having any lockdowns is bad



        #these weights prioritize minimizing deaths and critical cases by varying degrees over minimizing lockdown stages
        minimize_infections_and_lockdown_1 = [-100, -100, 0, -1, -1, -1, -1]
        minimize_infections_and_lockdown_2 = [-200, -200, 0, -1, -1, -1, -1]
        minimize_infections_and_lockdown_3 = [-400, -400, 0, -1, -1, -1, -1]

        #these weights prioritize minimizing deaths by varying degrees over minimizing deaths and lockdown stages
        prioritize_infections = []
        for w in [minimize_infections_and_lockdown_1, minimize_infections_and_lockdown_2, minimize_infections_and_lockdown_3]:
            w[0] = 0
            prioritize_infections.append(w)
        prioritize_infections.extend([minimize_infections_and_lockdown_1, minimize_infections_and_lockdown_2, minimize_infections_and_lockdown_3])

        #these weights prioritize minimizing lockdown stages by varying degrees over minimizing deaths and critical cases
        prioritize_lockdowns = []
        for w in prioritize_infections:
            for _ in range(n):
                stage4_penalty = np.random.randint(low=-10,high=-2 +1)
                stage3_penalty = np.random.randint(low=stage4_penalty,high=1)
                stage2_penalty = np.random.randint(low=stage3_penalty,high=1)
                stage1_penalty = np.random.randint(low=stage2_penalty,high=1)
                stage0_penalty = np.random.randint(low=stage1_penalty,high=1)
                w[6] = stage4_penalty
                w[5] = stage3_penalty
                w[4] = stage2_penalty
                w[3] = stage1_penalty
                w[2] = stage0_penalty
                prioritize_lockdowns.append(w)


        self.all_linear_w = [healthy_extreme,prevent_deaths,prevent_lockdown]+prioritize_infections + prioritize_lockdowns
        self.all_linear_w = np.array(self.all_linear_w)

        # print (self.all_linear_w[23])
        # print (self.all_linear_w[24])

        #these are our default values set in the pandemic config file; if you change the pandemic simulator config remember to update these values
        #for tiny town
        if town_size == "tiny":
            max_hospital_capacity = 5
            num_persons = 500
        elif town_size == "medium":
            #for medium town
            max_hospital_capacity = 10
            num_persons = 2000
        else:
            raise NotImplementedError("Only tiny and medium town sizes are supported currently")

        self.max_stage = 4
        self._threshold = 3 * max_hospital_capacity / num_persons

        #(0) weights used by original paper (1) only penlize when critical capacity is reached (2-n) balancing
        self.all_paper_w = [[-0.4, -0.1], [-1, 0], [-0.2, -0.1], [-0.1, -0.1]]
        self.all_paper_w = np.array(self.all_paper_w)

        self.use_specifc_rew=False

        print ("Initialized TruePandemicRewardFunction with town_size:", town_size)

    def get_all_weights(self):
        return list(self.all_linear_w) + list(self.all_paper_w)
    
    def set_specific_reward(self, i, flip_sign=False):
        
        #support flipping the sign of the reward function if i >= 25
        if i >= len(self.all_linear_w) + len(self.all_paper_w):
            i = i - (len(self.all_linear_w) + len(self.all_paper_w))
            flip_sign = True

        assert i >=0 and i < len(self.all_linear_w) + len(self.all_paper_w)
        self.use_specifc_rew=True
        self.reward_i = i
        self.flip_sign = flip_sign

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        # Implement the reward calculation logic here
        # print ("self._threshold:", self._threshold)
        # print ("Critical cases mean: ", np.mean(prev_obs.global_infection_summary[..., sorted_infection_summary.index(InfectionSummary.CRITICAL)]))
        # print ("Infected cases mean: ", np.mean(prev_obs.global_infection_summary[..., sorted_infection_summary.index(InfectionSummary.INFECTED)]))
        # print ("Dead cases mean: ", np.mean(prev_obs.global_infection_summary[..., sorted_infection_summary.index(InfectionSummary.DEAD)]))
        # print ("Stage: ", obs.stage[-1, -1].item())
        # print ("max critical reward:", self.calculate_paper_reward(prev_obs, action, obs, self.all_paper_w[1]))
        if self.use_specifc_rew:
            if self.reward_i < len(self.all_linear_w):
                return self.calculate_linear_reward(prev_obs, action, obs, self.all_linear_w[self.reward_i])*(-1 if self.flip_sign else 1)
            else:
                return self.calculate_paper_reward(prev_obs, action, obs, self.all_paper_w[self.reward_i - len(self.all_linear_w)])*(-1 if self.flip_sign else 1)
        return self.sample_calc_reward(prev_obs, action, obs)

    def sample_calc_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        rewards = []

        for flip_sign in [1,-1]:
            for weights in self.all_linear_w:
                rew = self.calculate_linear_reward(prev_obs, action, obs, weights)*flip_sign
                rewards.append(rew)
            
            _index = sorted_infection_summary.index(InfectionSummary.CRITICAL)
            for weights in self.all_paper_w:
                rew = self.calculate_paper_reward(prev_obs, action, obs, weights)*flip_sign
                rewards.append(rew)

                # if weights[1] == 0:
                #     critical_rew = max(
                #         np.mean(
                #             prev_obs.global_infection_summary[..., _index] - self._threshold
                #         )
                #         / self._threshold,
                #         0,
                #     )
                #     print ("Critical reward component: ", critical_rew)
                #     print ("prev_obs.global_infection_summary[..., _index]: ", np.mean(prev_obs.global_infection_summary[..., _index]))
                #     print ("critical feat:", prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.CRITICAL)])
                #     print ("death feat:", prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.DEAD)])
                #     print ("self._threshold: ", self._threshold)
                #     print ("-------\n")

        assert len(rewards) == 50
        return rewards
        
        # rew = self.calculate_linear_reward(prev_obs, action, obs, self.all_linear_w[0])
        # return [rew]

    def calculate_paper_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation, weights) -> float:
        # Implement the paper reward calculation logic here
        _index = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        
        critical_rew = max(
            np.mean(
                prev_obs.global_infection_summary[..., _index] - self._threshold
            )
            / self._threshold,
            0,
        )

        lockdown_rew = np.power(obs.stage[-1, -1].item(), 1.5)/np.power(self.max_stage, 1.5)

       
        return weights[0]*critical_rew + weights[1]*lockdown_rew

    # def feature2_reward(self, features, rm_i):
    #     if rm_i < len(self.all_linear_w):
    #         weights = self.all_linear_w[rm_i]
    #         return np.dot(weights, features)
    #     else:
    #         weights = self.all_paper_w[rm_i - len(self.all_linear_w)]
    #         return weights[0]*features[0] + weights[1]*features[1]

    def calculate_linear_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation, weights
    ) -> float:

        # dead_i = sorted_infection_summary.index(InfectionSummary.DEAD)
        # prev_dead = prev_obs.global_infection_summary[-1, -1, dead_i]
        # curr_dead = obs.global_infection_summary[-1, -1, dead_i]
        # daily_increase = curr_dead - prev_dead
        # # penalize increase

        # crit_i = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        # prev_crit = prev_obs.global_infection_summary[-1, -1, crit_i]
        # curr_crit = obs.global_infection_summary[-1, -1, crit_i]
        # reward = relative reduction in critical cases


        prev_critical = prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.CRITICAL)]
        prev_dead = prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.DEAD)]

        critical = obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.CRITICAL)]
        dead = obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.DEAD)]

        #Note: critical-prev_critical can be negative (if critical cases increase); the lockdown regulation action actually can influence this recovery rate; recovery is more likely if the ICU is not at capacity
        lockdown_vec = [0,0,0,0,0]
        lockdown_vec[int(obs.stage[-1, -1].item())] = 1
        feats = np.array([critical-prev_critical,dead-prev_dead]+lockdown_vec)
        # feats = np.array([critical-prev_critical,dead-prev_dead]+lockdown_vec)

        reward = np.dot(weights, feats)
        # -daily_increase + -(curr_crit-prev_crit)
        #np.dot(weights, feats)
        # print (feats)
        # print (reward)
        # print (weights)
        # assert False

        return reward

# TruePandemicRewardFunction()