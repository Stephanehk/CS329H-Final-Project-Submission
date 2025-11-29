"""Contains the RLController class."""

from flow.controllers.base_controller import BaseController


class RLController(BaseController):
    """RL Controller.

    Vehicles with this class specified will be stored in the list of the RL IDs
    in the Vehicles class.

    Usage: See base class for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification

    Examples
    --------
    A set of vehicles can be instantiated as RL vehicles as follows:

        >>> from flow.core.params import VehicleParams
        >>> vehicles = VehicleParams()
        >>> vehicles.add(acceleration_controller=(RLController, {}))

    In order to collect the list of all RL vehicles in the next, run:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> rl_ids = env.k.vehicle.get_rl_ids()
    """

    def __init__(
        self,
        veh_id,
        car_following_params,
        acc_controller=None,
        acc_controller_params={},
    ):
        """Instantiate an RL Controller."""
        BaseController.__init__(self, veh_id, car_following_params)

        if acc_controller is not None:
            if isinstance(acc_controller, str):
                from flow import controllers

                acc_controller = getattr(controllers, acc_controller)
            self.acc_controller = acc_controller(
                veh_id=veh_id,
                car_following_params=car_following_params,
                **acc_controller_params
            )

    def get_accel(self, env):
        """Pass, as this is never called; required to override abstractmethod."""
        pass

    def get_controller_accel(self, env):
        accel = self.acc_controller.get_accel(env)

        if self.acc_controller.accel_noise > 0:
            accel += np.sqrt(env.sim_step) * np.random.normal(
                0, self.acc_controller.accel_noise
            )

        for failsafe in self.acc_controller.failsafes:
            accel = failsafe(env, accel)

        return accel
