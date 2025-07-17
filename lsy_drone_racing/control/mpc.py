from lsy_drone_racing.control import Controller
import numpy as np
import casadi as cs
from numpy.typing import NDArray
import lsy_drone_racing.utils.path_planner2 as pp
import lsy_drone_racing.utils.trajectory as trajectory
from scipy.spatial.transform import Rotation as R

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
        trajectory.trajectory = self.path
        
        # settings
        self._interpolation_factor = 3
        
        self.path = self.interpolate_trajectory_linear(self.path, self._interpolation_factor)
        
        # self.pp.plot()


    def compute_control(
            self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
        ) -> NDArray[np.floating]:

        pos = obs["pos"]
        gates_pos = np.array(obs["gates_pos"])
        obstacles_pos = np.array(obs["obstacles_pos"])
        gates_rpy = np.array([self.quaternion_to_rpy(quat) for quat in obs["gates_quat"]])
        
        if self.is_obs_different(gates_pos, obstacles_pos):
            self.gates_pos = gates_pos
            self.obstacles_pos = obstacles_pos
            self.gates_rpy = gates_rpy
            
            self.path = self.pp.plan(pos, gates_pos, gates_rpy, obstacles_pos)
            trajectory.trajectory = self.path
            self.path = self.interpolate_trajectory_linear(self.path, self._interpolation_factor)
            # self.path = self.smoo
            

        
        position = self.path[self.tick]
        # print(self.path[self.tick])
        
        self.tick += 1
        if(self.tick >= len(self.path)):
            self.finished = True
        
        
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
    
    def is_obs_different(self, gates_pos, obstacles_pos):
        for i in range(len(gates_pos)):
            if not np.array_equal(gates_pos[i], self.gates_pos[i]):
                return True
            elif not np.array_equal(obstacles_pos[i], self.obstacles_pos[i]):
                return True
        return False
    
    def quaternion_to_rpy(self, quaternion):
        """
        Convert a quaternion (x, y, z, w) to roll, pitch, yaw.
        
        Parameters:
            quaternion (array-like): [x, y, z, w]
        
        Returns:
            np.ndarray: [roll, pitch, yaw] in radians
        """
        
        rotation = R.from_quat(quaternion)
        rpy = rotation.as_euler('xyz', degrees=False)
        return rpy
    
    def clean_path(self, path, threshold=1e-6):
        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        keep = np.insert(diffs > threshold, 0, True)  # Always keep the first point
        return path[keep]
    
    def smooth_trajectory(self, path, min_distance=0.005, max_speed=2.0, dt=0.02):
        """
        Simple trajectory smoother that fixes common issues
        
        Args:
            path: numpy array of 3D points (N, 3)
            min_distance: minimum distance between points
            max_speed: maximum allowed speed (m/s)
            dt: time step for speed calculation
        
        Returns:
            cleaned numpy array of 3D points
        """
        if len(path) < 2:
            return path
        
        # Remove NaN/inf
        valid_mask = ~(np.any(np.isnan(path) | np.isinf(path), axis=1))
        clean_path = path[valid_mask]
        
        if len(clean_path) < 2:
            print("ERROR: All points are NaN/inf!")
            return path
        
        # Remove duplicate/too-close points
        result = [clean_path[0]]
        
        for i in range(1, len(clean_path)):
            dist = np.linalg.norm(clean_path[i] - result[-1])
            
            if dist < min_distance:
                continue
                
            # Limit step size based on max speed
            max_step = max_speed * dt
            if dist > max_step:
                direction = (clean_path[i] - result[-1]) / dist
                new_point = result[-1] + direction * max_step
            else:
                new_point = clean_path[i]
                
            result.append(new_point)
        
        final_path = np.array(result)
        
        # Quick validation
        if np.any(np.isnan(final_path)) or np.any(np.isinf(final_path)):
            print("WARNING: NaN/inf still present after cleaning!")
        
        if len(final_path) > 1:
            speeds = np.linalg.norm(np.diff(final_path, axis=0), axis=1) / dt
            if np.max(speeds) > max_speed * 1.1:  # 10% tolerance
                print(f"WARNING: High speed detected: {np.max(speeds):.2f} m/s")
        
        return final_path
    
    def interpolate_trajectory_linear(self, trajectory, interpolation_factor=2):
        """
        Linearly interpolates between trajectory points.

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