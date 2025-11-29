from abc import ABCMeta, abstractmethod
import numpy as np
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: GlucoseObservation, action: float, obs: GlucoseObservation
    ) -> float:
        pass

# -------------------- OBJECTIVES --------------------

class BgStdReward(RewardFunction):
    """
    Reward = -std(bg)
    Encourages stability by penalizing variability in blood glucose.
    """
    def calculate_reward(self, prev_obs, action, obs):
        bg = obs.bg[obs.bg >= 0]
        if bg.size == 0:
            return 0.0
        return -float(np.std(bg))


class BgTimeInRangeReward(RewardFunction):
    """
    Reward = fraction of bg readings within [70,180]
    """
    def calculate_reward(self, prev_obs, action, obs):
        bg = obs.bg[obs.bg >= 0]
        if bg.size == 0:
            return 0.0
        in_range = ((bg >= 70) & (bg <= 180)).sum()
        return float(in_range) / bg.size


class HypoCountReward(RewardFunction):
    """
    Penalty = count of bg < 70
    """
    def calculate_reward(self, prev_obs, action, obs):
        count = np.sum(obs.bg[obs.bg >= 0] < 70)
        return -float(count)


class HyperCountReward(RewardFunction):
    """
    Penalty = count of bg > 180
    """
    def calculate_reward(self, prev_obs, action, obs):
        count = np.sum(obs.bg[obs.bg >= 0] > 180)
        return -float(count)


class SevereHypoCountReward(RewardFunction):
    """
    Penalty = count of bg < 54 (severe hypo proxy)
    """
    def calculate_reward(self, prev_obs, action, obs):
        count = np.sum(obs.bg[obs.bg >= 0] < 54)
        return -float(count)


class CostReward(RewardFunction):
    """
    Reward = -cost (obs.cost is negative expected cost)
    """
    def calculate_reward(self, prev_obs, action, obs):
        return -float(obs.cost)


class InsulinStdReward(RewardFunction):
    """
    Reward = -std(insulin)
    Encourages consistent insulin dosing.
    """
    def calculate_reward(self, prev_obs, action, obs):
        ins = obs.insulin[obs.insulin >= 0]
        if ins.size == 0:
            return 0.0
        return -float(np.std(ins))


class BgDropPerInsReward(RewardFunction):
    """
    Reward = (mean(prev_bg) - mean(curr_bg)) / (action + eps)
    Measures BG decrease per unit insulin.
    """
    def calculate_reward(self, prev_obs, action, obs):
        eps = 1e-6
        prev_bg = prev_obs.bg[prev_obs.bg >= 0]
        curr_bg = obs.bg[obs.bg >= 0]
        if prev_bg.size == 0 or curr_bg.size == 0:
            return 0.0
        delta = np.mean(prev_bg) - np.mean(curr_bg)
        return float(delta) / (action + eps)


class MealInsulinTimeDiffReward(RewardFunction):
    """
    Reward = -|idx_insulin - idx_meal| / history_length
    Encourages timely insulin relative to meals.
    """
    def calculate_reward(self, prev_obs, action, obs):
        cho = obs.cho
        ins = obs.insulin
        meal_idxs = np.where(cho > 0)[0]
        ins_idxs = np.where(ins > 0)[0]
        if meal_idxs.size == 0 or ins_idxs.size == 0:
            return 0.0
        dt = abs(ins_idxs[-1] - meal_idxs[-1])
        length = len(ins)
        return -float(dt) / (length + 1e-6)


class BgCheckProportionReward(RewardFunction):
    """
    Reward = proportion of valid bg readings
    """
    def calculate_reward(self, prev_obs, action, obs):
        total = len(obs.bg)
        valid = np.sum(obs.bg >= 0)
        return float(valid) / (total + 1e-6)


class DataValidityProportionReward(RewardFunction):
    """
    Reward = average proportion of valid insulin and cho entries
    """
    def calculate_reward(self, prev_obs, action, obs):
        total = len(obs.insulin)
        valid_ins = np.sum(obs.insulin >= 0)
        valid_cho = np.sum(obs.cho >= 0)
        return float(valid_ins + valid_cho) / (2 * total + 1e-6)


class InsulinAdjustmentSlopeReward(RewardFunction):
    """
    Reward = action * (mean(prev_bg) - 100) / 100
    Encourages insulin adjustments when bg deviates from 100.
    """
    def calculate_reward(self, prev_obs, action, obs):
        prev_bg = prev_obs.bg[prev_obs.bg >= 0]
        if prev_bg.size == 0:
            return 0.0
        dev = (np.mean(prev_bg) - 100.0) / 100.0
        return float(action) * dev


class BgMeanProximityReward(RewardFunction):
    """
    Reward = -|mean(bg) - 100|
    Encourages mean bg near 100.
    """
    def calculate_reward(self, prev_obs, action, obs):
        bg = obs.bg[obs.bg >= 0]
        if bg.size == 0:
            return 0.0
        return -abs(float(np.mean(bg)) - 100.0)


class ChoStdReward(RewardFunction):
    """
    Reward = -std(cho)
    Encourages consistent carbohydrate intake.
    """
    def calculate_reward(self, prev_obs, action, obs):
        cho = obs.cho[obs.cho >= 0]
        if cho.size == 0:
            return 0.0
        return -float(np.std(cho))


class PostMealBgDiffReward(RewardFunction):
    """
    Reward = -|mean(post_bg) - mean(pre_bg)|
    Evaluates post-meal BG change.
    """
    def calculate_reward(self, prev_obs, action, obs):
        bg = obs.bg
        cho = obs.cho
        meal_idxs = np.where(cho > 0)[0]
        if meal_idxs.size == 0:
            return 0.0
        idx = meal_idxs[-1]
        pre = bg[max(0, idx-12):idx][bg[max(0, idx-12):idx] >= 0]
        post = bg[idx+1:idx+13][bg[idx+1:idx+13] >= 0]
        if pre.size == 0 or post.size == 0:
            return 0.0
        return -abs(float(np.mean(post)) - np.mean(pre))


class TimeInRangeCostRatioReward(RewardFunction):
    """
    Reward = (time_in_range_fraction) / (abs(cost) + eps)
    Balances BG control vs cost.
    """
    def calculate_reward(self, prev_obs, action, obs):
        eps = 1e-6
        bg = obs.bg[obs.bg >= 0]
        if bg.size == 0:
            return 0.0
        in_range = ((bg >= 70) & (bg <= 180)).sum()
        frac = float(in_range) / bg.size
        return frac / (abs(obs.cost) + eps)


# ------------------ UNDESIRABLE PATTERNS ------------------

class ConcurrentBgInsulinIncPenalty(RewardFunction):
    """
    Penalty = delta_bg * delta_insulin when both increase.
    """
    def calculate_reward(self, prev_obs, action, obs):
        prev_bg = prev_obs.bg[prev_obs.bg >= 0]
        curr_bg = obs.bg[obs.bg >= 0]
        prev_ins = prev_obs.insulin[prev_obs.insulin >= 0]
        curr_ins = obs.insulin[obs.insulin >= 0]
        if prev_bg.size == 0 or curr_bg.size == 0 or prev_ins.size == 0 or curr_ins.size == 0:
            return 0.0
        d_bg = np.mean(curr_bg) - np.mean(prev_bg)
        d_ins = np.mean(curr_ins) - np.mean(prev_ins)
        return - (d_bg * d_ins) if d_bg > 0 and d_ins > 0 else 0.0


class ChoDropBgHighPenalty(RewardFunction):
    """
    Penalty = (mean(prev_cho)-mean(curr_cho)) * (mean(curr_bg)-180) when both positive.
    """
    def calculate_reward(self, prev_obs, action, obs):
        prev_cho = prev_obs.cho[prev_obs.cho >= 0]
        curr_cho = obs.cho[obs.cho >= 0]
        curr_bg = obs.bg[obs.bg >= 0]
        if prev_cho.size == 0 or curr_cho.size == 0 or curr_bg.size == 0:
            return 0.0
        drop = np.mean(prev_cho) - np.mean(curr_cho)
        high = np.mean(curr_bg) - 180.0
        return - (drop * high) if drop > 0 and high > 0 else 0.0


class RapidBgRisePenalty(RewardFunction):
    """
    Penalty = max(bg_last - bg_prev_last, 0)
    """
    def calculate_reward(self, prev_obs, action, obs):
        if prev_obs.bg.size == 0 or obs.bg.size == 0:
            return 0.0
        d = obs.bg[-1] - prev_obs.bg[-1]
        return -float(max(d, 0.0))


class BgStdPenalty(RewardFunction):
    """
    Penalty = std(bg)
    Rapid fluctuations penalty.
    """
    def calculate_reward(self, prev_obs, action, obs):
        bg = obs.bg[obs.bg >= 0]
        if bg.size == 0:
            return 0.0
        return -float(np.std(bg))


class InsulinIncCostDecPenalty(RewardFunction):
    """
    Penalty = (mean(curr_ins)-mean(prev_ins)) * (prev_cost - curr_cost)
    when both positive.
    """
    def calculate_reward(self, prev_obs, action, obs):
        prev_ins = prev_obs.insulin[prev_obs.insulin >= 0]
        curr_ins = obs.insulin[obs.insulin >= 0]
        if prev_ins.size == 0 or curr_ins.size == 0:
            return 0.0
        d_ins = np.mean(curr_ins) - np.mean(prev_ins)
        d_cost = prev_obs.cost - obs.cost
        return - (d_ins * d_cost) if d_ins > 0 and d_cost > 0 else 0.0


class InsulinDropBgHighPenalty(RewardFunction):
    """
    Penalty = (mean(prev_ins)-mean(curr_ins)) * (mean(curr_bg)-100)
    when both positive.
    """
    def calculate_reward(self, prev_obs, action, obs):
        prev_ins = prev_obs.insulin[prev_obs.insulin >= 0]
        curr_ins = obs.insulin[obs.insulin >= 0]
        curr_bg = obs.bg[obs.bg >= 0]
        if prev_ins.size == 0 or curr_ins.size == 0 or curr_bg.size == 0:
            return 0.0
        drop_ins = np.mean(prev_ins) - np.mean(curr_ins)
        high = np.mean(curr_bg) - 100.0
        return - (drop_ins * high) if drop_ins > 0 and high > 0 else 0.0


class ChoDropBgRisePenalty(RewardFunction):
    """
    Penalty = (mean(prev_cho)-mean(curr_cho)) * (mean(curr_bg)-mean(prev_bg))
    when both positive.
    """
    def calculate_reward(self, prev_obs, action, obs):
        prev_cho = prev_obs.cho[prev_obs.cho >= 0]
        curr_cho = obs.cho[obs.cho >= 0]
        prev_bg = prev_obs.bg[prev_obs.bg >= 0]
        curr_bg = obs.bg[obs.bg >= 0]
        if prev_cho.size == 0 or curr_cho.size == 0 or prev_bg.size == 0 or curr_bg.size == 0:
            return 0.0
        drop = np.mean(prev_cho) - np.mean(curr_cho)
        rise = np.mean(curr_bg) - np.mean(prev_bg)
        return - (drop * rise) if drop > 0 and rise > 0 else 0.0


class CostSpikeNoChangePenalty(RewardFunction):
    """
    Penalty = max(delta_cost,0) when delta_ins and delta_bg ~ 0
    """
    def calculate_reward(self, prev_obs, action, obs):
        prev_bg = prev_obs.bg[prev_obs.bg >= 0]
        curr_bg = obs.bg[obs.bg >= 0]
        prev_ins = prev_obs.insulin[prev_obs.insulin >= 0]
        curr_ins = obs.insulin[obs.insulin >= 0]
        if prev_bg.size == 0 or curr_bg.size == 0 or prev_ins.size == 0 or curr_ins.size == 0:
            return 0.0
        d_bg = abs(np.mean(curr_bg) - np.mean(prev_bg))
        d_ins = abs(np.mean(curr_ins) - np.mean(prev_ins))
        d_cost = obs.cost - prev_obs.cost
        # cost is negative expected cost; spike => d_cost > 0
        if d_cost > 0 and d_bg < 1e-3 and d_ins < 1e-3:
            return -float(d_cost)
        return 0.0

# ------------------- NEW FEATURE-EXPANSIONS -------------------

# class BgPolynomialExpansionReward(RewardFunction):
#     """
#     Reward = -mean(bg^2) to penalize high glucose levels.
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         bg = obs.bg[obs.bg >= 0]
#         if bg.size == 0:
#             return 0.0
#         return -float(np.mean(bg ** 2))


# class BgInsulinInteractionReward(RewardFunction):
#     """
#     Reward = mean(bg * insulin) to capture interaction effects.
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         bg = obs.bg[obs.bg >= 0]
#         insulin = obs.insulin[obs.insulin >= 0]
#         if bg.size == 0 or insulin.size == 0:
#             return 0.0
#         return float(np.mean(bg * insulin))


# class BgRatioReward(RewardFunction):
#     """
#     Reward = -mean(bg / (insulin + eps)) to penalize high bg relative to insulin.
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         bg = obs.bg[obs.bg >= 0]
#         insulin = obs.insulin[obs.insulin >= 0]
#         eps = 1e-6
#         if bg.size == 0 or insulin.size == 0:
#             return 0.0
#         return -float(np.mean(bg / (insulin + eps)))


# class BgLogTransformReward(RewardFunction):
#     """
#     Reward = -sum(log(bg + 1)) to penalize high bg levels.
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         bg = obs.bg[obs.bg >= 0]
#         if bg.size == 0:
#             return 0.0
#         return -float(np.sum(np.log(bg + 1)))  # log transformation


# class BgRollingMeanReward(RewardFunction):
#     """
#     Reward = mean of the last 5 bg values (rolling mean)
#     """
#     def __init__(self):
#         self.bg_history = []

#     def calculate_reward(self, prev_obs, action, obs):
#         bg = obs.bg[obs.bg >= 0]
#         self.bg_history.extend(bg.tolist())
#         if len(self.bg_history) > 5:
#             self.bg_history = self.bg_history[-5:]  # keep only last 5 values
#         if len(self.bg_history) == 0:
#             return 0.0
#         return float(np.mean(self.bg_history))


# class BgDerivativeReward(RewardFunction):
#     """
#     Reward = Δbg (difference between last two bg readings)
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         bg = obs.bg[obs.bg >= 0]
#         if bg.size < 2:
#             return 0.0
#         return float(bg[-1] - bg[-2])


# class InsulinRollingMeanReward(RewardFunction):
#     """
#     Reward = mean of the last 5 insulin values (rolling mean)
#     """
#     def __init__(self):
#         self.insulin_history = []

#     def calculate_reward(self, prev_obs, action, obs):
#         ins = obs.insulin[obs.insulin >= 0]
#         self.insulin_history.extend(ins.tolist())
#         if len(self.insulin_history) > 5:
#             self.insulin_history = self.insulin_history[-5:]  # keep only last 5 values
#         if len(self.insulin_history) == 0:
#             return 0.0
#         return float(np.mean(self.insulin_history))


# class InsulinDerivativeReward(RewardFunction):
#     """
#     Reward = Δinsulin (difference between last two insulin readings)
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         ins = obs.insulin[obs.insulin >= 0]
#         if ins.size < 2:
#             return 0.0
#         return float(ins[-1] - ins[-2])


# class BgInsulinAbsDiffReward(RewardFunction):
#     """
#     Reward = -|bg - insulin| to penalize large discrepancies.
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         bg = obs.bg[obs.bg >= 0]
#         insulin = obs.insulin[obs.insulin >= 0]
#         if bg.size == 0 or insulin.size == 0:
#             return 0.0
#         return -float(np.abs(np.mean(bg) - np.mean(insulin)))


# class LogCostPenalty(RewardFunction):
#     """
#     Penalty = -log(-cost + 1) to penalize high costs.
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         if obs.cost >= 0:  # only penalize if cost is negative (expected cost)
#             return 0.0
#         return -float(np.log(-obs.cost + 1))


# class ChoAbsDiffReward(RewardFunction):
#     """
#     Reward = -|mean(cho) - target| to encourage cho intake near a target.
#     """
#     def calculate_reward(self, prev_obs, action, obs):
#         cho = obs.cho[obs.cho >= 0]
#         target = 45.0  # Example target for carbohydrate intake
#         if cho.size == 0:
#             return 0.0
#         return -float(np.abs(np.mean(cho) - target))

# class ChoRollingMeanReward(RewardFunction):
#     """
#     Reward = mean of the last 5 cho values (rolling mean)
#     """
#     def __init__(self):
#         self.cho_history = []

#     def calculate_reward(self, prev_obs, action, obs):
#         cho = obs.cho[obs.cho >= 0]
#         self.cho_history.extend(cho.tolist())
#         if len(self.cho_history) > 5:
#             self.cho_history = self.cho_history[-5:]  # keep only last 5 values
#         if len(self.cho_history) == 0:
#             return 0.0
#         return float(np.mean(self.cho_history))