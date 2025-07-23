"""A simple path tracking controller for an obstacle-avoided cubic hermite spline path.

Usage:
python scripts/sim.py -g -t -config "level1.toml" -controller "spline_tracker.py"

Note:
This controller does not work on config level2.toml, due to adverse path replacement effects

Hot it works:
    It calls the path planner (generates list of points to fly to and avoids obstacles),
    interpolates the path to a desired speed (NUM_POINTS) and then for every tick uses
    the next entry in the path.

Constants:
    NUM_POINTS (int): Number of points in the trajectory. Determines speed (one point per tick)

Author:
    Antonio Steiger

"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

import lsy_drone_racing.utils.path_planner as pp
import lsy_drone_racing.utils.trajectory as trajectory
from lsy_drone_racing.control import Controller

NUM_POINTS = 600


class SplineTracker(Controller):
    """Spline tracking controller class implementing the Controller base class."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initializes all variables and generates an initial path.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: info dict from environment
            config: config dict from environment
        """
        # state machine
        self.finished = False
        self.tick = 0

        # measurements
        self.pos = obs["pos"]
        self.gates_pos = obs["gates_pos"]
        self.obstacles_pos = obs["obstacles_pos"]
        self.gates_quat = obs["gates_quat"]
        self.gates_rpy = []
        self.gates_rpy = np.array([self.quaternion_to_rpy(quat) for quat in self.gates_quat])

        # path planner
        self.pp = pp.PathPlanner(self.pos, self.gates_pos, self.gates_rpy, self.obstacles_pos)
        self.path = self.pp.path

        # settings
        self._interpolation_factor = 1

        self.path = self.interpolate_to_n_points(self.path, NUM_POINTS)
        trajectory.trajectory = self.path

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: info from environment

        Returns:
            A drone state command [x, y, z, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] in
            absolute coordinates.

        How it works:
            Gets the next point on the path for every tick.
        """
        gates_pos = np.array(obs["gates_pos"])
        obstacles_pos = np.array(obs["obstacles_pos"])
        gates_rpy = np.array([self.quaternion_to_rpy(quat) for quat in obs["gates_quat"]])

        self.tick += 1
        if self.tick >= len(self.path) - 1:
            self.finished = True

        if self.is_obs_different(gates_pos, obstacles_pos):
            self.gates_pos = gates_pos
            self.obstacles_pos = obstacles_pos
            self.gates_rpy = gates_rpy

            path = self.pp.plan(gates_pos, gates_rpy, obstacles_pos)
            self.path[self.tick :] = self.interpolate_to_n_points(path, NUM_POINTS)[self.tick :]
            trajectory.trajectory = self.interpolate_to_n_points(self.path, 500)

        position = self.path[self.tick]

        return np.concatenate((position, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Returns True when the controller is finished."""
        return self.finished

    def is_obs_different(
        self, gates_pos: NDArray[np.floating], obstacles_pos: NDArray[np.floating]
    ) -> bool:
        """Determine if the passed observation differs from what is currently known.

        Args:
            gates_pos: numpy array of newly observed gate positions ( obs["gate_pos"] )
            obstacles_pos: numpy array of newly observed obstacle positions ( obs["obstacles_pos"] )

        Returns:
            True if the observation changed, False if not.
        """
        for i in range(len(gates_pos)):
            if not np.array_equal(gates_pos[i], self.gates_pos[i]):
                return True
            elif not np.array_equal(obstacles_pos[i], self.obstacles_pos[i]):
                return True
        return False

    def quaternion_to_rpy(self, quaternion: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert a quaternion (x, y, z, w) to roll, pitch, yaw.

        Parameters:
            quaternion (array-like): [x, y, z, w]

        Returns:
            np.ndarray: [roll, pitch, yaw] in radians
        """
        rotation = R.from_quat(quaternion)
        rpy = rotation.as_euler("xyz", degrees=False)
        return rpy

    def interpolate_to_n_points(
        self, trajectory: NDArray[np.floating], target_num_points: int
    ) -> NDArray[np.floating]:
        """Adjusts the trajectory to have approximately target_num_points by subsampling or interpolating linearly.

        Parameters:
            trajectory (np.ndarray): Original trajectory, shape (N, D)
            target_num_points (int): Desired number of output points

        Returns:
            np.ndarray: Resampled trajectory
        """
        if target_num_points < 2:
            raise ValueError("target_num_points must be >= 2")

        N = trajectory.shape[0]

        if N == target_num_points:
            return trajectory

        result = []

        # Compute uniformly spaced positions in the original trajectory
        step = (N - 1) / (target_num_points - 1)

        for i in range(target_num_points):
            pos = i * step
            idx = int(np.floor(pos))
            alpha = pos - idx

            if idx >= N - 1:
                result.append(trajectory[-1])
            else:
                a = trajectory[idx]
                b = trajectory[idx + 1]
                point = (1 - alpha) * a + alpha * b
                result.append(point)

        return np.array(result)
