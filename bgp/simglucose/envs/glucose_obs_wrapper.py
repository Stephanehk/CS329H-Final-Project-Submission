from dataclasses import dataclass
import numpy as np
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv

@dataclass
class GlucoseObservation:
    """Dataclass that updates numpy arrays with information from PandemicSimState. Typically, this observation is
    used by the reinforcement learning interface."""
    
    bg: np.ndarray = None
    insulin: np.ndarray = None
    cho: np.ndarray = None
    cost: float = None


    def update_obs_with_sim_state(
        self,
        env: SimglucoseEnv,
    ) -> None:
        """
        Update the GlucoseObservation with the information from the simulation environment.

        Observations are stochastic, and contain the continuous glucose monitor (CGM) observations and insulin administered
        To provide temporal context, we augment our observed state space to include the previous 4 hours of CGM and insulin data at 5-minute resolution

        Actions are real positive numbers, denoting the size of the insulin bolus in medication units.

        Args:
            env: SimglucoseEnv instance containing the current simulation state
            hist_index: history time index (default: 0)
        """
        # Get the history of CGM readings (blood glucose)
        self.bg = env.env.CGM_hist[-env.state_hist:]
        
        # Get the history of insulin administration
        self.insulin = env.env.insulin_hist[-env.state_hist:]
        
        # Get the history of carbohydrate (meal) intake
        self.cho = env.env.CHO_hist[-env.state_hist:]
        
        # Calculate the expected patient cost
        self.cost = self.get_expected_patient_cost(env.env.insulin_hist, env.env.CGM_hist)
        
        # Pad with -1 if history is not long enough
        if len(self.bg) < env.state_hist:
            self.bg = np.concatenate((np.full(env.state_hist - len(self.bg), -1), self.bg))
        if len(self.insulin) < env.state_hist:
            self.insulin = np.concatenate(
                (np.full(env.state_hist - len(self.insulin), -1), self.insulin)
            )

    def get_expected_patient_cost(self,insulin_hist,bg_hist):
        expected_cost = 0.32 * np.mean(insulin_hist[-1])  # Cost of the insulin.
        if bg_hist[-1] < 70:
            # Patient is hypoglycemic, so add potential cost of hospital visit.
            expected_cost += 10 * 1350 / (12 * 24 * 365)
        return -expected_cost
    
    def flatten(self,):
        return np.concatenate((self.bg,self.insulin, self.cho, [self.cost])).flatten()
