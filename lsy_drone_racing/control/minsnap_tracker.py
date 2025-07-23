"""Minimum Snap Trajectory Tracking Controller.

How it Works:
    Generates a minimum snap trajectory for a race track with waypoints (gates) using the minsnap-trajectories
    library and added human intuition for waypoint placement and flight directions.
Author:
    Antonio Steiger
"""

import minsnap_trajectories as ms
import numpy as np
from numpy.typing import NDArray

import lsy_drone_racing.utils.minsnap as minsnap
import lsy_drone_racing.utils.trajectory as trajectory
from lsy_drone_racing.control import Controller


class MinSnapTracker(Controller):
    """Minimum Snap Trajectory Tracking Controller Class implementing base Controller class."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the mimimum snap trajectory controller.

        Arguments:
            obs: initial observations from the environment
            info: info from the environment
            config: config from the environment
        """
        super().__init__(obs, info, config)

        self._tick = 0
        self._finished = False

        # Settings
        self._t_total = 9.0
        self._freq = config.env.freq
        self._interpolation_factor = 2

        # generate trajectory
        self.current_gates_pos = np.copy(obs["gates_pos"])
        self.initial_pos = np.copy(obs["pos"])
        # print(obs)
        trajectory.trajectory = minsnap.generate_trajectory(self.make_refs(), self._t_total)
        print("MINSNAP: Trajectory generated")

        # interpolate trajectory to regulate speed
        trajectory.trajectory = interpolate_trajectory_linear(
            trajectory.trajectory, self._interpolation_factor
        )
        print("MINSNAP: Trajectory interpolated")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute next action for the drone to take.

        Arguments:
            obs: newest observation dict from the environment
            info: info from environment

        Returns:
            Control input for the drone in the form:
            [x, y, z, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        """
        if self._tick >= len(trajectory.trajectory):
            self._finished = True
            return np.zeros(13, dtype=np.float32)
        else:
            position = trajectory.trajectory[self._tick, 0:3]
            # attitude = trajectory.trajectory[self._tick, 3:7]
            # velocity = trajectory.trajectory[self._tick, 7:10]
            # thrust = trajectory.trajectory[self._tick, 10]
            # thrust = np.array([thrust], dtype=np.float32)
            # angular = trajectory.trajectory[self._tick, 11:14]
            # acceleration = trajectory.trajectory[self._tick, 14:17]
            # rpy = quat_to_euler(attitude)
            gates_pos = obs["gates_pos"]

            # Regenerate trajectory if the observation changes
            if self.is_obs_different(gates_pos):
                self.current_gates_pos = np.copy(gates_pos)
                self.regenerate_trajectory()

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
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        # print("Pos: ", obs["pos"])
        # print("Goal: ", trajectory.trajectory[self._tick, :3])
        self._tick += 1

        return self._finished

    def is_obs_different(self, gates_pos: NDArray[np.floating]) -> bool:
        """Determine if the passed observation differs from what is currently known.

        Args:
            gates_pos: numpy array of newly observed gate positions ( obs["gate_pos"] )

        Returns:
            True if the observation changed, False if not.
        """
        for i in range(len(gates_pos)):
            if not np.array_equal(gates_pos[i], self.current_gates_pos[i]):
                return True
        return False

    def regenerate_trajectory(self) -> None:
        """Regenerate trajectory."""
        trajectory.trajectory = minsnap.generate_trajectory(self.make_refs(), self._t_total)
        trajectory.trajectory = interpolate_trajectory_linear(
            trajectory.trajectory, self._interpolation_factor
        )

        return

    def make_refs(self) -> NDArray[np.object_]:
        """Create the references for the minsnap-trajectories library."""
        waypoint1 = self.initial_pos.copy()  # starting point

        waypoint2 = self.current_gates_pos[0].copy()  # first gate

        waypoint3 = self.current_gates_pos[1].copy()  # second gate

        waypoint4 = self.current_gates_pos[2].copy()  # third gate
        waypoint4[1] += 0.25  # increased y to "touch gate"

        waypoint5 = self.current_gates_pos[3].copy()  # fourth gate
        waypoint5[1] -= 0.2  # increased y to meet velocity threshold

        refs = [
            # starting point
            ms.Waypoint(
                time=0.0,
                position=np.array(waypoint1),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                jerk=np.array([0.0, 0.0, 0.0]),
            ),
            # first gate
            ms.Waypoint(time=2.0, position=np.array(waypoint2)),
            # second gate
            ms.Waypoint(time=4.0, position=np.array(waypoint3), velocity=np.array([0.8, 0.8, 0.0])),
            # third gate
            ms.Waypoint(time=6.7, position=np.array(waypoint4), velocity=np.array([0.0, 0.0, 0.0])),
            # fourth gate
            ms.Waypoint(time=self._t_total, position=np.array(waypoint5)),
        ]

        return refs


def quat_to_euler(q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert a quaternion to euler angles roll, pitch and yaw.

    Arguments:
        q: quaternion [q1, q2, q3, w]

    Returns:
        [roll, pitch, yaw]
    """
    x, y, z, w = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def interpolate_trajectory_linear(
    trajectory: NDArray[np.floating], interpolation_factor: int = 2
) -> NDArray[np.floating]:
    """Linearly interpolates between trajectory points.

    Parameters:
        trajectory (np.ndarray): Original trajectory, shape (N, 13)
        interpolation_factor (int): Number of segments per original segment.
                                    Must be >= 1. (1 = no interpolation)

    Returns:
        np.ndarray: Interpolated trajectory
    """
    if interpolation_factor < 1:
        raise ValueError("interpolation_factor must be >= 1")

    N = trajectory.shape[0]
    result = []

    for i in range(N - 1):
        a = trajectory[i]
        b = trajectory[i + 1]

        for k in range(interpolation_factor):
            alpha = k / interpolation_factor
            point = (1 - alpha) * a + alpha * b
            result.append(point)

    result.append(trajectory[-1])  # add the final point
    return np.array(result)
