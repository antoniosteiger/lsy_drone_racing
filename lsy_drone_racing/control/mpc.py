from lsy_drone_racing.control import Controller
import numpy as np
import casadi as cs
from numpy.typing import NDArray
import lsy_drone_racing.utils.path_planner as PathPlanner
import lsy_drone_racing.utils.trajectory as trajectory

# Drone data
mass = 0.028  # kg
gravitational_accel = 9.80665  # m/s^2

J = cs.diag([1.4e-5, 1.4e-5, 2.17e-5])  # kg*m^2
J_inv = cs.diag([1 / 1.4e-5, 1 / 1.4e-5, 1 / 2.17e-5])

thrustCoefficient = 2.88e-8  # N*s^2
dragCoefficient = 7.24e-10  # N*m*s^2
propdist = 0.092

# Track Data
gates_pos = np.array([
    [1.0, 1.5, 0.07],
    [0.45, -0.5, 0.56],
    [1.0, -1.05, 1.11],
    [0.0, 1.0, 0.56],
    [-0.5, 0.0, 1.11]
])

gates_rpy = np.array([
    [0, 0, 0],
    [0.0, 0.0, 2.35],
    [0.0, 0.0, -0.78],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 3.14]

])

obstacles = np.array([
    [1.0, 0.0, 1.4],
    [0.5, -1.0, 1.4],
    [0.0, 1.5, 1.4],
    [-0.5, 0.5, 1.4]
])

class MPC(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        self.finished = False
        self.tick = 0
        self.pp = PathPlanner.PathPlanner()
        self.path = self.pp.plan(gates_pos, gates_rpy, obstacles)
        trajectory.trajectory = self.path
        #pp.plot()

    def compute_control(
            self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
        ) -> NDArray[np.floating]:

        # gates_pos = np.array(obs["gates_pos"])
        # if self.is_obs_different(gates_pos):
        #     self.path = self.pp.plan(gates_pos)
        #     trajectory.trajectory = self.path

        position = self.path[self.tick]
        self.tick += 1
        if(self.tick >= 400):
            self.finished = True
        print(self.path)
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
        return self.finished
    
    def is_obs_different(self, gates_pos):
        for i in range(len(gates_pos)):
            if not np.array_equal(gates_pos[i], self.current_gates_pos[i]):
                return True
        return False