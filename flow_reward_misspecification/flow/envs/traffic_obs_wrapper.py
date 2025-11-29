from dataclasses import dataclass
import numpy as np
from flow.envs.merge import MergePOEnv
from dataclasses import dataclass, field
from typing import List
import numpy as np
from flow.envs.merge import MergePOEnv

@dataclass
class TrafficObservation:
    """Dataclass that updates numpy arrays with information from MergePOEnv. This observation is
    used by the reinforcement learning (RL) interface for traffic control. RL vehicles are those
    controlled by the RL algorithm, while non-RL vehicles are other drivers on the road."""
    # ── Per-RL-vehicle arrays ────────────────────────────────────────────────
    ego_speeds: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Un-normalised speed (m/s) of each RL-controlled vehicle.**

    leader_headways: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Bumper-to-bumper distance (m) from every RL vehicle to its current leader.**

    leader_speed_diffs: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **(Leader speed − ego speed) in m/s for each RL vehicle.**

    follower_headways: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Distance (m) between each RL vehicle and its follower (if any).**

    follower_speed_diffs: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **(Ego speed − follower speed) in m/s.**


    # ── Global variables ─────────────────────────────────────────────────────
    all_vehicle_speeds: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Vector of raw speeds (m/s) for *every* vehicle in the network.**

    ego_vehicle_accels: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Vector of raw accelerations (m/s^2) for *every* RL-controlled vehicle.**

    target_velocity: float = (
        0.0  # **The velocity (m/s) that all vehicles should be traveling at. Its against the law to travel faster than this speed.**
    )

    fail: bool = (
        False  # **True if a collision / failure occurred in the current rollout.**
    )

    # ── Helper variables ─────────────────────────────────────────────────────
    rl_ids: List[str] = field(
        default_factory=list
    )  # **String IDs of the RL-controlled vehicles present at this timestep.**
      # Maintains a stable mapping from array index → vehicle id.

    # Normalisation constants kept for convenient re-scaling in policy code
    max_speed: float = (
        0.0  # **Network’s absolute speed limit (m/s).** Used for normalising features.
    )

    max_length: float = (
        0.0  # **Approx. longest possible bumper-to-bumper gap (m) on the road.**
        # Used to normalise headway distances.
    )
    # ──────────────────────────────────────────────────────────────────────────
    # Update from env ─────────────────────────────────────────────────────────
    def update_obs_with_sim_state(self, env: MergePOEnv, action: np.ndarray, infos) -> None:
        """Populate all fields from the current simulator state."""
        # ---------- global -----------
        self.rl_ids          = env.rl_veh.copy()
        self.target_velocity = env.env_params.additional_params["target_velocity"]
        self.max_speed       = env.k.network.max_speed()
        self.max_length      = env.k.network.length()

        veh_ids              = env.k.vehicle.get_ids()
        self.all_vehicle_speeds = np.array(env.k.vehicle.get_speed(veh_ids), dtype=np.float32)
        self.ego_vehicle_accels = np.array(action, dtype=np.float32)

        # Simple collision indicator used by Flow environments
        self.fail = infos["crash"]

        # ---------- per-RL-vehicle ----------
        num_rl = env.num_rl
        self.ego_speeds           = np.zeros(num_rl, dtype=np.float32)
        self.leader_headways      = np.zeros(num_rl, dtype=np.float32)
        self.leader_speed_diffs   = np.zeros(num_rl, dtype=np.float32)
        self.follower_headways    = np.zeros(num_rl, dtype=np.float32)
        self.follower_speed_diffs = np.zeros(num_rl, dtype=np.float32)

        for i, rl_id in enumerate(self.rl_ids):
            if rl_id not in veh_ids:
                continue  # RL vehicle has left the network

            ego_speed = env.k.vehicle.get_speed(rl_id)
            leader_id = env.k.vehicle.get_leader(rl_id)
            follower_id = env.k.vehicle.get_follower(rl_id)

            # Leader ----------------------------------------------------------
            if leader_id in ("", None):
                leader_speed = self.max_speed
                headway = self.max_length
            else:
                leader_speed = env.k.vehicle.get_speed(leader_id)
                headway = (
                    env.k.vehicle.get_x_by_id(leader_id)
                    - env.k.vehicle.get_x_by_id(rl_id)
                    - env.k.vehicle.get_length(rl_id)
                )

            # Follower --------------------------------------------------------
            if follower_id in ("", None):
                follower_speed = 0.0
                follow_headway = self.max_length
            else:
                follower_speed = env.k.vehicle.get_speed(follower_id)
                follow_headway = env.k.vehicle.get_headway(follower_id)

            self.ego_speeds[i]           = ego_speed
            self.leader_headways[i]      = headway
            self.leader_speed_diffs[i]   = leader_speed - ego_speed
            self.follower_headways[i]    = follow_headway
            self.follower_speed_diffs[i] = ego_speed - follower_speed

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers -----------------------------------------------------------------
    def as_flat_array(self, normalise: bool = False) -> np.ndarray:
        """
        Return a 1-D observation suitable for an RL policy.
        If `normalise` is True, the same normalisation scheme as before is used.
        """
        if normalise:
            # Scale speeds by max_speed and distances by max_length
            scale_v = lambda v: v / (self.max_speed + 1e-8)
            scale_d = lambda d: d / (self.max_length + 1e-8)
            feats = [
                scale_v(self.ego_speeds),
                scale_v(self.leader_speed_diffs),
                scale_d(self.leader_headways),
                scale_v(self.follower_speed_diffs),
                scale_d(self.follower_headways),
            ]
        else:
            feats = [
                self.ego_speeds,
                self.leader_speed_diffs,
                self.leader_headways,
                self.follower_speed_diffs,
                self.follower_headways,
            ]
        return np.concatenate(feats, dtype=np.float32)

    # Alias kept for backwards compatibility with older code
    def flatten(self) -> np.ndarray:
        return self.as_flat_array()