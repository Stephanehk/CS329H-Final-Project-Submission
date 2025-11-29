from abc import ABCMeta, abstractmethod
from typing import Any, Sequence
import numpy as np

from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation


class RewardFunction(metaclass=ABCMeta):
    """Minimal interface expected by the rest of the code base."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """Create the reward object (sub-classes decide what to store)."""
        pass

    @abstractmethod
    def calculate_reward(
        self,
        prev_obs: TrafficObservation,
        action: Sequence[float],
        obs: TrafficObservation,
    ) -> float:
        """Return the scalar reward produced by `action` in `prev_obs → obs`."""
        ...

class PanEtAlTrueTrafficRewardFunction(RewardFunction):
    """
    Re-implementation of the Flow `compute_reward` logic, but expressed in
    observation-space so it can be reused outside the simulator loop.

    • **Term 1 (`cost1`)** – system-level desired-velocity reward  
      Matches `rewards.desired_velocity`: 1 when all vehicles drive exactly
      at `target_velocity`, 0 when they are far from it.

    • **Term 2 (`cost2`)** – headway penalty for each RL vehicle  
      Linear penalty when time-headway < `t_min` seconds.

    • **Term 3 (`cost3`)** – acceleration penalty  
      Linear penalty on the mean |accel| sent to RL vehicles.

    The final reward is `max(η₁·cost1 + η₂·cost2 + η₃·cost3, 0)`, unless
    a failure/ collision occurred, in which case it is **0**.
    """

    def __init__(
        self,
        eta1: float = 1.0,
        eta2: float = 0.10,
        eta3: float = 1.0,
        t_min: float = 1.0,
        accel_threshold: float = 0.0,
        evaluate: bool = False,
    ) -> None:
        """
        Args
        ----
        eta1 / eta2 / eta3 : weights for the three cost terms (see above)
        t_min              : minimum acceptable time-headway (s)
        accel_threshold    : |a| below which no acceleration penalty applies
        evaluate           : if *True*, reward is simply average speed of
                             **all** vehicles ( Flow’s evaluation mode )
        """
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.t_min = t_min
        self.accel_threshold = accel_threshold
        self.evaluate = evaluate
        self._eps = np.finfo(np.float32).eps

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def calculate_reward(
        self,
        prev_obs: TrafficObservation,
        action: Sequence[float],
        obs: TrafficObservation,
    ) -> float:
        """
        Compute the ground-truth reward given *prev_obs* → *obs* transition.

        Parameters
        ----------
        prev_obs : TrafficObservation
            Observation before the action (unused here, but part of the API).
        action : Sequence[float]
            Vector of bounded accelerations (one per RL vehicle).
        obs : TrafficObservation
            Observation **after** the environment step.

        Returns
        -------
        float
            Non-negative reward value.
        """
        # ------------------------------------------------------------------ #
        # 0. Early exit – evaluation mode or collision                       #
        # ------------------------------------------------------------------ #
        if self.evaluate:
            # mean speed of *all* vehicles (m/s)
            return float(np.mean(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0

        if getattr(obs, "fail", False):
            # print (".    fail=True")
            return 0.0

        # ------------------------------------------------------------------ #
        # 1. Desired-velocity term (system level)                            #
        # ------------------------------------------------------------------ #
        vel = obs.all_vehicle_speeds
        if vel.size == 0:
            cost1 = 0.0
        else:
            target = obs.target_velocity
            max_cost = np.linalg.norm(np.full_like(vel, target))
            cost = np.linalg.norm(vel - target)
            cost1 = max(max_cost - cost, 0.0) / (max_cost + self._eps)  # ∈ [0, 1]
            # print (".  cost1:", cost1)
        # ------------------------------------------------------------------ #
        # 2. Headway penalty (RL vehicles only)                              #
        # ------------------------------------------------------------------ #
        cost2 = 0.0
        for speed, headway in zip(obs.ego_speeds, obs.leader_headways):
            if speed > 0:  # avoid divide-by-zero
                t_headway = max(headway / speed, 0.0)
                cost2 += min((t_headway - self.t_min) / self.t_min, 0.0)  # ≤ 0
        # print (".  cost2:", cost2)

        # ------------------------------------------------------------------ #
        # 3. Acceleration penalty                                            #
        # ------------------------------------------------------------------ #
        if action is None or len(action) == 0:
            cost3 = 0.0
        else:
            mean_abs_accel = float(np.mean(np.abs(action)))
            cost3 = (
                self.accel_threshold - mean_abs_accel
                if mean_abs_accel > self.accel_threshold
                else 0.0
            )  # ≤ 0
        # print (".  cost3:", cost3)
        # ------------------------------------------------------------------ #
        # 4. Weighted sum, clamped to be non-negative                        #
        # ------------------------------------------------------------------ #
        reward = (
            self.eta1 * cost1 + self.eta2 * cost2 + self.eta3 * cost3
        )
        return reward


class TrueTrafficRewardFunction(RewardFunction):
    """The true reward function used by the traffic simulator environment."""

    def __init__(self):
        np.random.seed(42)
        max_speed = 30 #m/s, set by traffic env but hardcoded here for now
        #a reasonable reward function is likely prioritizes:
        #(1) crashes
        #(2) mean speed for all vehicles
        #(3) min_vehicle_speed to prevent deadlocks
        #(4) squared speed above max (to prevent speeding) 
        #make note about why squared speed 

        #and then varies in how much the following are weighted:
        #(5) fuel consumption (via VSP proxy) --seems to range from 0-15; we shoyld be careful about scaling this too high
        #make note that we are assuming ICE vehicles here
        #(6) braking (via brake proxy)
        #(7) headway penalty (to promote politeness)
    
        #abids by (1-4) but not (5-7)
        w_agressive = np.load("data/gt_rew_fn_data/traffic_feasible_w_aggressive_stephane.npy")
        # [self.compute_crash_penalty(max_speed), 1, 1, 0, 0, 0, 0 ]

        #balance between all reward components
        # w_balanced = [self.compute_crash_penalty(max_speed), 1, 1, -1, -0.1, -1, 1 ]
        w_balanced = np.load("data/gt_rew_fn_data/traffic_feasible_w_for_stephane.npy")

        #costly equitable
        w_equitable= np.load("data/gt_rew_fn_data/traffic_feasible_w_egaleterian_stephane.npy")
    
        self.weights = [w_agressive, w_balanced, w_equitable]
        self.weights.extend(self.interpolate_weights(w_agressive, w_balanced, w_equitable, num_samples=22))

        self.crash_weights = [self.compute_crash_penalty(max_speed), 0, 0, 0, 0, 0, 0]

        # for w in self.weights:
        #     print (w)
        assert len(self.weights) == 25

    def get_all_weights(self):
        return self.weights

    def set_specific_reward(self, i, flip_sign=False):
        self.use_crash_rew = False
        if i == 50:
            self.use_crash_rew = True
        #support flipping the sign of the reward function if i >= 25
        if i >= len(self.weights):
            i = i - len(self.weights)
            flip_sign = True

        self.use_specific_rew=True
        self.reward_i = i
        self.flip_sign = flip_sign
        # print (len(self.weights))
        # print ("Using specific reward function index:", i, " with flip_sign=", flip_sign)

    def interpolate_weights(self, w_aggressive, w_balanced, w_equitable, num_samples=22):
        """
        Interpolates between three weight vectors (aggressive, balanced, equitable)
        while keeping the first element fixed.
        
        Args:
            w_aggressive, w_balanced, w_equitable (np.ndarray): Weight vectors of the same length.
            num_samples (int): Number of interpolated weight vectors to generate.
        
        Returns:
            np.ndarray: Array of shape (num_samples, len(w_aggressive)) with interpolated weights.
        """
        assert len(w_aggressive) == len(w_balanced) == len(w_equitable), "All vectors must be same length"

        w_aggressive = np.array(w_aggressive)
        w_balanced = np.array(w_balanced)
        w_equitable = np.array(w_equitable)
        
        # Interpolation weights across a 2D simplex between the three points
        samples = []
        for i in np.linspace(0, 1, num_samples):
            for j in np.linspace(0, 1 - i, 1):  # optional: could randomize barycentric weights
                k = 1 - i - j
                w_interp = i * w_aggressive + j * w_balanced + k * w_equitable
                w_interp[0] = w_aggressive[0]  # keep the 0th element fixed
                samples.append(w_interp)
        
        # If we have more than num_samples, pick evenly spaced ones
        samples = np.array(samples)
        if len(samples) > num_samples:
            indices = np.linspace(0, len(samples) - 1, num_samples, dtype=int)
            samples = samples[indices]

        return samples

    def feature2_reward(self, features, rm_i):
        if rm_i < len(self.weights):
            weights = self.weights[rm_i]
            return np.dot(weights, features)
        else:
            raise ValueError(f"rm_i {rm_i} is out of bounds for TrueTrafficRewardFunction with {len(self.weights)} reward models!")

    def compute_crash_penalty(self, max_speed, horizon=300):
        #the following computes a crash penalty via the method proposed in https://arxiv.org/pdf/2104.13906

        #we assume that, if ADs are idle, all reward components are 0; this implies that all human drivers are blocked
        g_idle = 0
        #we assume that the highest return for any trajectory is <= max_speed*horizon;
        #this implies an idealized setting where we have no penalites from crashes, speeding, fuel, braking, or headway violations
        #note this is likely a significant over-estimate, in practice the mean speed without colisions is ~5-10 m/s not the max of 30m/s
        g_success = max_speed*horizon
        #total successful path length = 373m, and 50-60 year olds drive ~ <1000000000m per collision 
        #using these two facts we work backwards to compute p via the formula: 1000000000 = (p/(1-p) + 0.5)*225
        p=0.9999996270000696
        crash_penalty = (g_idle-p*g_success)/(1-p)
        return crash_penalty

    def sample_linear_reward(self, prev_obs: TrafficObservation, action, obs: TrafficObservation, weights) -> float:
        """
        A sample linear reward function that weights different components.
        """
        # weights for cost1, cost2, and cost3, respectively
        eta1, eta2, eta3, eta4, eta5, eta6, eta7 = weights

        cost1 = 0 if not self.did_crash(obs) else 1 
        cost2 = self.mean_velocity(obs) 
        cost3 = self.min_vehicle_speed(obs)
        cost4 = self.squared_speed_above_max(obs, obs.target_velocity)
        cost5 = self.fuel_proxy_vsp(action, obs.ego_speeds)
        cost6 = self.brake_proxy_energy(action, obs.ego_speeds)
        cost7 = self.headway_penalty(obs)

        self.cost1 = cost1
        self.cost2 = cost2
        self.cost3 = cost3
        self.cost4 = cost4
        self.cost5 = cost5
        self.cost6 = cost6
        self.cost7 = cost7

        # print ("did_crash:", cost1)
        # print ("mean_velocity:", cost2)
        # print ("min_vehicle_speed:", cost3)
        # print ("squared_speed_above_max:", cost4)
        # print ("fuel_proxy_vsp:", cost5)
        # print ("brake_proxy_energy:", cost6)
        # print ("headway_penalty:", cost7)
        # print ("------------------------------")

        return eta1 * cost1 + eta2 * cost2 + eta3 * cost3 + eta4 * cost4 + eta5 * cost5 + eta6 * cost6 + eta7 * cost7, self.did_crash(obs)

    def sample_calc_reward(self, prev_obs: TrafficObservation, action: int, obs: TrafficObservation) -> float:
        rewards = []
        for flip_sign in [1,-1]:
            for w in self.weights:
                r,crash = self.sample_linear_reward(prev_obs, action, obs, w)
                if not crash:
                    r = r * flip_sign
                rewards.append(r)
        return rewards

    def calculate_reward(self, prev_obs: TrafficObservation, action: int, obs: TrafficObservation) -> float:

        if self.use_specific_rew:
            if self.use_crash_rew:
                r,_ = self.sample_linear_reward(prev_obs, action, obs, self.crash_weights)
            else:
                r, crash = self.sample_linear_reward(prev_obs, action, obs, self.weights[self.reward_i])
                if not crash:
                    r *= (-1 if self.flip_sign else 1)
            # print ("Computed reward with specific reward function index", self.reward_i, " flip_sign=", self.flip_sign, ", use_crash_rew:", self.use_crash_rew, ", reward:", r)
            return r
            
        return self.sample_calc_reward(prev_obs, action, obs)

    def mean_velocity(self, obs: TrafficObservation):
        return float(np.mean(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0
    
    def squared_speed_above_max(self, obs: TrafficObservation, target):
        speed = obs.all_vehicle_speeds
        if speed.size == 0:
            return 0.0
        else:
            excess = np.maximum(np.abs(speed) - target, 0.0)  # overshoot per component
            penalty = np.sum(excess ** 2)  # squared penalty
            return penalty

    def min_vehicle_speed(self, obs: TrafficObservation):
        speed = np.abs(obs.all_vehicle_speeds)
        if speed.size == 0:
            return 0.0
        else:
            return np.min(speed)

    def did_crash(self, obs: TrafficObservation):
        return obs.fail

    def fuel_proxy_vsp(self, accels, speeds,
                            *,
                            # ---- Physical constants / fleet-average parameters ----
                            g=9.80665,         # [m/s^2] standard gravity (CODATA/NIST)
                            rho=1.225,         # [kg/m^3] standard air density at 15°C, sea level
                            C_r=0.010,         # [-] rolling resistance coeff. for passenger cars on asphalt
                            CdA_over_m=4.7e-4, # [m^2/kg] fleet-avg. drag area per mass ≈ (Cd*A)/m
                            sin_theta=0.0      # [-] road grade ≈ sin(theta); use 0.05 for ~5% uphill if known
                            ):
        """
        Per-mass tractive power proxy (≈ Vehicle-Specific Power, VSP) to penalize fuel/energy use.
        Sums across vehicles at the current timestep.

        VSP is *tractive power per unit mass*. Expanded (per EPA MOVES / Jiménez-Palacios):
            VSP = a*v                   (inertia/kinetic energy term)
                + g*sin(theta)*v        (grade/climb term)
                + C_r*g*v               (rolling resistance term)
                + 0.5*rho*(Cd*A/m)*v^3  (aerodynamic drag term)

        We include all terms except grade by default; set sin_theta if you have slope.

        Parameters
        ----------
        accels : array-like (n_vehicles,)
            Longitudinal accelerations [m/s^2]; positive => throttle.
        speeds : array-like (n_vehicles,)
            Corresponding speeds [m/s].
        g : float
            Standard gravity [m/s^2].
        rho : float
            Air density [kg/m^3]. 1.225 is standard atmosphere at 15°C; adjust if sim uses a different altitude/temp.
        C_r : float
            Rolling resistance coefficient (dimensionless). Passenger-car range ≈ 0.008–0.015; 0.010 is a robust midrange.
        CdA_over_m : float
            Drag area per mass [m^2/kg]. For a typical car: Cd≈0.3, A≈2.2–2.5 m^2 => Cd*A≈0.66–0.75 m^2; m≈1500 kg
            ⇒ (Cd*A)/m ≈ 0.44–0.50×10^-3 m^2/kg. We use 4.7e-4 as a representative fleet-average.
        sin_theta : float
            Road grade as sin(theta). If unknown, leave 0.0; if you have % grade, sin(theta)≈grade_fraction (e.g., 0.05 for 5%).

        Returns
        -------
        float
            Sum over vehicles of per-mass tractive power proxy [W/kg up to constant factors].
            Use directly as a fuel/energy penalty term (to be weighted in your RL reward).
        """
        a = np.asarray(accels, dtype=float)
        v = np.asarray(speeds, dtype=float)

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


    def brake_proxy_energy(self, accels, speeds):
        """
        Brake wear / braking energy-rate proxy per unit mass.

        From kinetic energy E_k = 0.5*m*v^2, dE_k/dt = m*v*dv/dt = m*v*a.
        During braking a<0, so brake power (heat dissipation rate) ≈ -m*v*a.
        Per unit mass that is simply:  -v*a. We aggregate only the negative-a part:

            Proxy = sum_i v_i * max(-a_i, 0)

        Parameters
        ----------
        accels : array-like (n_vehicles,)
            Longitudinal accelerations [m/s^2]; negative => braking.
        speeds : array-like (n_vehicles,)
            Corresponding speeds [m/s].

        Returns
        -------
        float
            Sum over vehicles of per-mass braking power proxy [m^2/s^3], suitable as a brake-wear penalty.
        """
        a = np.asarray(accels, dtype=float)
        v = np.asarray(speeds, dtype=float)
        neg = np.clip(-a, 0.0, None)  # take only braking (make positive)
        return float(np.sum(neg * v))

    def headway_penalty(self, obs):
        # ------------------------------------------------------------------ #
        # 2. Headway penalty (RL vehicles only)                              #
        # ------------------------------------------------------------------ #
        t_min = 1  # smallest acceptable time headway
        cost = 0.0
        for speed, headway in zip(obs.ego_speeds, obs.leader_headways):
            if speed > 0:  # avoid divide-by-zero
                t_headway = max(headway / speed, 0.0)
                cost += min((t_headway - t_min) / t_min, 0.0)
        return cost        

    


def desired_velocity(env, fail=False, edge_list=None):
    r"""Encourage proximity to a desired velocity.

    This function measures the deviation of a system of vehicles from a
    user-specified desired velocity peaking when all vehicles in the ring
    are set to this desired velocity. Moreover, in order to ensure that the
    reward function naturally punishing the early termination of rollouts due
    to collisions or other failures, the function is formulated as a mapping
    :math:`r: \\mathcal{S} \\times \\mathcal{A}
    \\rightarrow \\mathbb{R}_{\\geq 0}`.
    This is done by subtracting the deviation of the system from the
    desired velocity from the peak allowable deviation from the desired
    velocity. Additionally, since the velocity of vehicles are
    unbounded above, the reward is bounded below by zero,
    to ensure nonnegativity.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system
    edge_list : list  of str, optional
        list of edges the reward is computed over. If no edge_list is defined,
        the reward is computed over all edges

    Returns
    -------
    float
        reward value
    """
    if edge_list is None:
        veh_ids = env.k.vehicle.get_ids()
    else:
        veh_ids = env.k.vehicle.get_ids_by_edge(edge_list)

    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.0


    target_vel = env.env_params.additional_params["target_velocity"]
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    return max(max_cost - cost, 0) / (max_cost + eps)    


def commute_time(env, rl_actions):
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    if any(vel < -100):
        return -10000.0
    if len(vel) == 0:
        return -10000.0

    commute = np.array([(v + 0.001) ** -1 for v in vel])
    commute = commute[commute > 0]
    return -np.mean(commute)

def merge_true_reward_fn(env, rl_actions):
    """See class definition."""
    # print ("I am computing reward here!!")

    return np.mean(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    if env.env_params.evaluate:
        return np.mean(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))
    else:
        # return a reward of 0 if a collision occurred
        # if kwargs["fail"]:
        #     return 0
        # reward high system-level velocities
        # if self.local_reward == 'local':
        # 	cost1 = rewards.local_desired_velocity(self, self.rl_veh, fail=kwargs["fail"])
        # elif self.local_reward == 'partial_first':
        # 	cost1 = rewards.local_desired_velocity(self, self.rl_veh[:3], fail=kwargs["fail"])
        # elif self.local_reward == 'partial_last':
        # 	cost1 = rewards.local_desired_velocity(self, self.rl_veh[-3:], fail=kwargs["fail"])
        # else:
        cost1 = desired_velocity(env, fail=False)


        # penalize small time headways
        cost2 = 0
        t_min = 1  # smallest acceptable time headway
        for rl_id in env.rl_veh:
            lead_id = env.k.vehicle.get_leader(rl_id)
            if lead_id not in ["", None] and env.k.vehicle.get_speed(rl_id) > 0:
                t_headway = max(
                    env.k.vehicle.get_headway(rl_id)
                    / env.k.vehicle.get_speed(rl_id),
                    0,
                )
                cost2 += min((t_headway - t_min) / t_min, 0)


        cost3 = 0
        mean_actions = np.mean(np.abs(np.array(rl_actions)))
        accel_threshold = 0

        if mean_actions > accel_threshold:
            cost3 += accel_threshold - mean_actions

        # weights for cost1, cost2, and cost3, respectively
        eta1, eta2, eta3 = 1.00, 0.10, 1

        return max(eta1 * cost1 + eta2 * cost2 + eta3 * cost3, 0)

def penalize_accel(env, rl_actions):
    if rl_actions is None:
        return 0
    mean_actions = np.mean(np.abs(np.array(rl_actions)))
    accel_threshold = 0
    return min(0, accel_threshold - mean_actions)

def penalize_headway(env, rl_actions):
    cost = 0
    t_min = 1  # smallest acceptable time headway
    for rl_id in env.rl_veh:
        lead_id = env.k.vehicle.get_leader(rl_id)
        if lead_id not in ["", None] and env.k.vehicle.get_speed(rl_id) > 0:
            t_headway = max(
                env.k.vehicle.get_headway(rl_id) / env.k.vehicle.get_speed(rl_id), 0
            )
            cost += min((t_headway - t_min) / t_min, 0)
    return cost
