from lsy_drone_racing.control import Controller
import numpy as np
import casadi as cs
from numpy.typing import NDArray
import lsy_drone_racing.utils.path_planner2 as pp
import lsy_drone_racing.utils.trajectory as trajectory
from scipy.spatial.transform import Rotation as R

NUM_POINTS = 600

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
        
        # settings
        self._interpolation_factor = 1
        
        self.path = self.interpolate_to_n_points(self.path, NUM_POINTS)
        trajectory.trajectory = self.path
        
        # self.pp.plot()


    def compute_control(
            self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
        ) -> NDArray[np.floating]:

        pos = obs["pos"]
        gates_pos = np.array(obs["gates_pos"])
        obstacles_pos = np.array(obs["obstacles_pos"])
        gates_rpy = np.array([self.quaternion_to_rpy(quat) for quat in obs["gates_quat"]])

        self.tick += 1
        if(self.tick >= len(self.path)-1):
            self.finished = True
        
        if self.is_obs_different(gates_pos, obstacles_pos):
            self.gates_pos = gates_pos
            self.obstacles_pos = obstacles_pos
            self.gates_rpy = gates_rpy
            
            path = self.pp.plan(pos, gates_pos, gates_rpy, obstacles_pos)
            self.path[self.tick:] = self.interpolate_to_n_points(path, NUM_POINTS)[self.tick:]
            trajectory.trajectory = self.interpolate_to_n_points(self.path, 500)
            

        
        position = self.path[self.tick]
        # print(self.path[self.tick])
        
        
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
    
    def interpolate_to_n_points(self, trajectory, target_num_points):
        """
        Adjusts the trajectory to have approximately target_num_points
        by subsampling or interpolating linearly.

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
    
    # def interpolate_trajectory_linear(self, trajectory, interpolation_factor=2):
    #     """
    #     Linearly interpolates or subsamples the trajectory based on interpolation_factor.

    #     Parameters:
    #         trajectory (np.ndarray): Original trajectory, shape (N, D)
    #         interpolation_factor (float): 
    #             - f > 1.0: add points between original ones
    #             - f < 1.0: remove points (subsample)
    #             - f = 1.0: no change

    #     Returns:
    #         np.ndarray: Adjusted trajectory
    #     """
    #     if interpolation_factor <= 0:
    #         raise ValueError("interpolation_factor must be > 0")

    #     N = trajectory.shape[0]

    #     # Case 1: Subsample (f < 1)
    #     if interpolation_factor < 1.0:
    #         stride = int(round(1 / interpolation_factor))
    #         result = trajectory[::stride]
    #         if not np.all(result[-1] == trajectory[-1]):
    #             result = np.vstack([result, trajectory[-1]])
    #         return result

    #     # Case 2: Interpolate (f >= 1)
    #     result = []
    #     for i in range(N - 1):
    #         a = trajectory[i]
    #         b = trajectory[i + 1]
    #         num_segments = int(interpolation_factor)
    #         for k in range(num_segments):
    #             alpha = k / interpolation_factor
    #             point = (1 - alpha) * a + alpha * b
    #             result.append(point)
    #     result.append(trajectory[-1])
    #     return np.array(result)