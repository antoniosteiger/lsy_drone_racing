# Time-Optimal Trajectory Controller

from lsy_drone_racing.control import Controller
import numpy as np
from numpy.typing import NDArray

class TrajectoryController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        self.trajectory = np.loadtxt(config.trajectory_file, delimiter=",", usecols=(0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16))
        print("CONTROLLER: Trajectory points loaded")
        self._tick = 0
        self._step = 0
        self._freq = config.env.freq
        self._finished = False
        self._action = np.concatenate((self.trajectory[0, 1:4], np.zeros(10)), dtype=np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        if self._step >= len(self.trajectory):
            self._finished = True
            return np.zeros(13, dtype=np.float32)
        elif (self._tick * (1.0 / self._freq))/2 >= self.trajectory[self._step, 0]:
            position = self.trajectory[self._step, 1:4]
            velocity = self.trajectory[self._step, 4:7]
            angular = self.trajectory[self._step, 7:10]
            acceleration = self.trajectory[self._step, 10:13]
            self._step += 1
            self._action = np.concatenate((position, np.zeros(10)), dtype=np.float32)
            #print(f"CONTROLLER: Step {self._step} - Position: {position}, Velocity: {velocity}, Angular: {angular}, Acceleration: {acceleration}")
            return self._action
        else:
            return self._action
    
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