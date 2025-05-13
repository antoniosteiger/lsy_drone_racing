from lsy_drone_racing.control import Controller
import numpy as np
from numpy.typing import NDArray


class MinSnapTracker(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        self.trajectory = np.loadtxt(config.trajectory_file, delimiter=",")
        print("CONTROLLER: Trajectory points loaded")
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        if self._tick >= len(self.trajectory):
            self._finished = True
            return np.zeros(13, dtype=np.float32)
        else:
            position = self.trajectory[self._tick, 0:3]
            attitude = self.trajectory[self._tick, 3:7]
            velocity = self.trajectory[self._tick, 7:10]
            thrust = self.trajectory[self._tick, 10]
            thrust = np.array([thrust], dtype=np.float32)
            angular = self.trajectory[self._tick, 11:14]
            acceleration = self.trajectory[self._tick, 14:17]
            rpy = quat_to_euler(attitude)

            return np.concatenate((position, velocity, acceleration, np.zeros(1), angular), dtype=np.float32)
            #return np.concatenate((thrust, rpy), dtype=np.float32)

    
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
        self._tick += 1
        return self._finished
    
def quat_to_euler(q):
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