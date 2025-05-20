# custom_trajectory_controller.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.snap_trajectory import SnapTrajectory
from numpy.typing import NDArray
from typing import Dict, Any, List, Tuple, Optional


class TrajectoryController(Controller):
    def __init__(
        self,
        obs: Dict[str, NDArray[np.floating]],
        info: Dict[str, Any],
        config: Dict[str, Any],
    ):
        super().__init__(obs, info, config)
        self.snap_traj = SnapTrajectory(degree=3)  # Cubic polynomial
        
        # Use custom waypoints instead of gate positions
        self.custom_waypoints = np.array([
            [1.0, 1.5, 0.05],
            [0.8, 1.0, 0.2],
            [0.55, -0.3, 0.5],
            [0.2, -1.3, 0.65],
            [1.1, -0.85, 1.1],
            [0.2, 0.5, 0.65],
            [0.0, 1.2, 0.525],
            [0.0, 1.2, 1.1],
            [-0.5, 0.0, 1.1],
            [-0.5, -0.5, 1.1],
        ])
        
        # Generate yaw angles based on direction of travel
        self.custom_waypoints_rpy = self._generate_yaw_angles(self.custom_waypoints)
        
        # Fall back to gate positions if available
        self.gate_positions, self.gate_rpys = self._get_gate_positions(config)
        
        self.obstacles = self._get_obstacles(config)
        self.drone_radius = 0.3
        self.waypoint_radius = 0.4  # Radius to consider waypoint reached
        self._freq = config.get("env", {}).get("freq", 50)
        self._tick = 0
        self._finished = False
        self.current_waypoint = 0
        self.T_segment = 2.0
        self.max_v = 2.0  # Reduced max velocity for smoother control
        self.max_a = 4.0
        self.replan_interval = 50
        self.yaw_smoothing = 0.8
        try:
            self.last_yaw = R.from_quat(obs["quat"].flatten()).as_euler('xyz')[2]
        except (KeyError, TypeError, ValueError):
            print("⚠️ Invalid quaternion, initializing yaw to 0")
            self.last_yaw = 0.0
        self.previous_yaw = self.last_yaw
        self.tracking_error_threshold = 0.3
        self.last_replan_time = 0
        self.min_replan_interval = 5
        self.total_time = 0.0
        current_pos = obs.get("pos", np.zeros(3)).flatten()
        self._initialize_trajectory(current_pos, obs)
        self.debug_data = {
            "waypoint_transitions": [],
            "replan_events": [],
            "tracking_errors": [],
            "planned_waypoints": [],
            "waypoint_proximity": [],
            "errors": [],
            "control_outputs": []
        }

    def _generate_yaw_angles(self, waypoints: NDArray) -> NDArray:
        """Generate yaw angles based on direction of travel between waypoints"""
        n_waypoints = len(waypoints)
        yaw_angles = np.zeros((n_waypoints, 3))
        
        for i in range(n_waypoints):
            if i < n_waypoints - 1:
                # Calculate direction vector to next waypoint
                direction = waypoints[i+1] - waypoints[i]
                # Calculate yaw angle from direction vector (xy-plane)
                yaw = np.arctan2(direction[1], direction[0])
            else:
                # For last waypoint, use the same yaw as the previous point
                yaw = yaw_angles[i-1, 2]
            
            yaw_angles[i] = [0.0, 0.0, yaw]
            
        return yaw_angles

    def _get_gate_positions(self, config: Dict) -> Tuple[NDArray, NDArray]:
        try:
            gates = config["env"]["track"]["gates"]
            gate_positions = np.array([gate["pos"] for gate in gates])
            gate_rpys = np.array([gate["rpy"] for gate in gates])
            print(f"Loaded {len(gate_positions)} gates: {gate_positions}")
            return gate_positions, gate_rpys
        except (KeyError, TypeError) as e:
            print(f"Using custom waypoints due to: {e}")
            self.debug_data["errors"].append(f"Gate loading error: {str(e)[:200]}")
            # Return empty arrays as we'll use custom waypoints
            return np.array([]), np.array([])

    def _get_obstacles(self, config: Dict) -> List[Dict]:
        try:
            obstacles = config["env"]["track"]["obstacles"]
            return [{
                "pos": np.array(obs["pos"]) - np.array([0, 0, 0.7]),
                "radius": 0.5
            } for obs in obstacles]
        except (KeyError, TypeError) as e:
            print(f"Using default obstacles due to: {e}")
            self.debug_data["errors"].append(f"Obstacle loading error: {str(e)[:200]}")
            return [{
                "pos": np.array([1.0, 0.0, 0.7]),
                "radius": 0.5
            }, {
                "pos": np.array([0.5, -1.0, 0.7]),
                "radius": 0.5
            }, {
                "pos": np.array([0.0, 1.5, 0.7]),
                "radius": 0.5
            }, {
                "pos": np.array([-0.5, 0.5, 0.7]),
                "radius": 0.5
            }]

    def _initialize_trajectory(self, current_pos: NDArray, obs: Dict):
        if len(self.custom_waypoints) == 0:
            self.debug_data["errors"].append("No waypoints found in initialization")
            print("No waypoints found, using current position")
            return
            
        first_waypoint = self.custom_waypoints[0]
        waypoint_rpy = self.custom_waypoints_rpy[0]
        distance = np.linalg.norm(first_waypoint - current_pos)
        self.T_segment = max(1.5, min(2.5, distance / self.max_v))
        
        waypoints = [
            [current_pos[0], 0, 0, 0, 0],  # x
            [current_pos[1], 0, 0, 0, 0],  # y
            [current_pos[2], 0, 0, 0, 0],  # z
            [0.0, 0, 0],                   # yaw
            [0.0],                         # t0
            [first_waypoint[0], 0, 0, 0, 0],   # x
            [first_waypoint[1], 0, 0, 0, 0],   # y
            [first_waypoint[2], 0, 0, 0, 0],   # z
            [waypoint_rpy[2], 0, 0],           # yaw
            [self.T_segment]               # tf
        ]
        
        try:
            success = self.snap_traj.traj(waypoints)
            if not success:
                raise ValueError("Trajectory generation failed")
            self.total_time = self.T_segment
            print(f"Initial trajectory to Waypoint 0, T_segment={self.T_segment:.2f}s")
            self.debug_data["planned_waypoints"].append(waypoints)
        except Exception as e:
            self.debug_data["errors"].append(f"Trajectory initialization error: {str(e)[:200]}")
            print(f"Trajectory initialization error: {e}")

    def compute_control(self, obs: Dict, info: Any = None) -> NDArray:
        try:
            if obs is None or "pos" not in obs or "quat" not in obs:
                self.debug_data["errors"].append("Invalid observation in compute_control")
                print("⚠️ Invalid observation received")
                return self._hover_command(obs)
                
            current_pos = obs.get("pos", np.zeros(3)).flatten()
            tracking_error = self._get_tracking_error(obs)
            self.debug_data["tracking_errors"].append(tracking_error)
            
            # Check if we've reached the current waypoint
            if self.current_waypoint < len(self.custom_waypoints):
                waypoint_pos = self.custom_waypoints[self.current_waypoint]
                waypoint_distance = np.linalg.norm(current_pos - waypoint_pos)
                
                self.debug_data["waypoint_proximity"].append({
                    "tick": self._tick,
                    "waypoint": self.current_waypoint,
                    "distance": float(waypoint_distance),
                    "tracking_error": float(tracking_error)
                })
                
                print(f"Tick {self._tick}: pos={current_pos}, waypoint_distance={waypoint_distance:.2f}m, tracking_error={tracking_error:.3f}")
                
                # Check if we've reached the current waypoint
                if waypoint_distance < self.waypoint_radius:
                    self.current_waypoint += 1
                    self.debug_data["waypoint_transitions"].append({
                        "tick": self._tick,
                        "waypoint": self.current_waypoint,
                        "pos": current_pos.tolist()
                    })
                    print(f"🏁 Reached waypoint {self.current_waypoint - 1}, targeting waypoint {self.current_waypoint}")
                    
                    # If we've reached the last waypoint, we're done
                    if self.current_waypoint >= len(self.custom_waypoints):
                        self._finished = True
                        print("🎉 Reached final waypoint, mission complete!")
                        return self._hover_command(obs)
            
            # Determine if we should replan
            should_replan = (
                (tracking_error > self.tracking_error_threshold and 
                 self._tick - self.last_replan_time > self.min_replan_interval) or
                (self._tick % self.replan_interval == 0) or
                (self.current_waypoint > 0 and self._tick - self.last_replan_time == 0)  # Just transitioned
            )
            
            if should_replan and self.current_waypoint < len(self.custom_waypoints):
                self.last_replan_time = self._tick
                self.debug_data["replan_events"].append({
                    "tick": self._tick,
                    "reason": "tracking_error" if tracking_error > self.tracking_error_threshold else 
                             "transition" if self._tick - self.last_replan_time == 0 else "interval",
                    "waypoint": self.current_waypoint
                })
                
                next_waypoint = self.custom_waypoints[self.current_waypoint]
                waypoint_rpy = self.custom_waypoints_rpy[self.current_waypoint]
                path_length = np.linalg.norm(next_waypoint - current_pos)
                print(f"Replanning: path_length={path_length:.2f}m")
                
                # Adjust time based on distance
                self.T_segment = max(1.5, min(2.5, path_length / self.max_v))
                print(f"Calculated T_segment={self.T_segment:.2f}s")
                
                waypoints = [
                    [current_pos[0], 0, 0, 0, 0],
                    [current_pos[1], 0, 0, 0, 0],
                    [current_pos[2], 0, 0, 0, 0],
                    [self.last_yaw, 0, 0],
                    [0.0],
                    [next_waypoint[0], 0, 0, 0, 0],
                    [next_waypoint[1], 0, 0, 0, 0],
                    [next_waypoint[2], 0, 0, 0, 0],
                    [waypoint_rpy[2], 0, 0],
                    [self.T_segment]
                ]
                
                try:
                    self.snap_traj = SnapTrajectory(degree=3)
                    success = self.snap_traj.traj(waypoints)
                    if not success:
                        raise ValueError("Trajectory generation failed")
                    self.total_time = self.T_segment
                    print(f"Replanned to waypoint {self.current_waypoint}, T_segment={self.T_segment:.2f}s")
                    self.debug_data["planned_waypoints"].append(waypoints)
                except Exception as e:
                    self.debug_data["errors"].append(f"Trajectory replanning error: {str(e)[:200]}")
                    print(f"Trajectory replanning error: {e}")
                    return self._hover_command(obs)
            
            # Evaluate the trajectory at the current time
            t = self._tick / self._freq
            segment_idx = 0
            for i in range(len(self.snap_traj.timestamps) - 1):
                if self.snap_traj.timestamps[i] <= t <= self.snap_traj.timestamps[i + 1]:
                    segment_idx = i
                    break
            else:
                # We're past the end of the trajectory, hover or replan
                if self.current_waypoint < len(self.custom_waypoints):
                    # Force a replan on the next tick
                    self.last_replan_time = self._tick - self.min_replan_interval - 1
                else:
                    self._finished = True
                self.debug_data["errors"].append(f"Time {t} out of trajectory bounds")
                print(f"Time {t} out of bounds, timestamps: {self.snap_traj.timestamps}")
                return self._hover_command(obs)
            
            result = self.snap_traj.evaluate(t, segment_idx)
            if result[0] is None:
                self.debug_data["errors"].append(f"Trajectory evaluation failed at t={t}, segment={segment_idx}")
                print(f"Trajectory evaluation failed at t={t}, segment={segment_idx}")
                return self._hover_command(obs)
                
            p, v, a, psi = result
            
            # Apply safety limits to velocity and acceleration
            v = np.clip(v, -self.max_v, self.max_v)
            a = np.clip(a, -self.max_a, self.max_a)
            
            try:
                current_yaw = R.from_quat(obs["quat"].flatten()).as_euler('xyz')[2]
            except (KeyError, TypeError, ValueError):
                current_yaw = self.last_yaw
                
            self.previous_yaw = self.last_yaw
            desired_yaw = psi
            
            # Smooth yaw transitions
            angle_diff = ((desired_yaw - self.last_yaw + np.pi) % (2 * np.pi)) - np.pi
            self.last_yaw += angle_diff * (1 - self.yaw_smoothing)
            yaw_rate = (self.last_yaw - self.previous_yaw) * self._freq
            
            # Assemble the control command
            control = np.array([
                p[0], p[1], p[2],          # Position
                v[0], v[1], v[2],          # Velocity
                a[0], a[1], a[2],          # Acceleration
                self.last_yaw,             # Yaw
                0.0, 0.0, yaw_rate         # Yaw rate
            ], dtype=np.float32)
            
            self.debug_data["control_outputs"].append({
                "tick": self._tick,
                "control": control.tolist()
            })
            
            self._tick += 1
            return control
            
        except Exception as e:
            self.debug_data["errors"].append(f"Control computation error: {str(e)[:200]}")
            print(f"Control computation error: {e}")
            return self._hover_command(obs)

    def _get_tracking_error(self, obs: Dict) -> float:
        try:
            t = self._tick / self._freq
            segment_idx = 0
            for i in range(len(self.snap_traj.timestamps) - 1):
                if self.snap_traj.timestamps[i] <= t <= self.snap_traj.timestamps[i + 1]:
                    segment_idx = i
                    break
            else:
                return 0.0
                
            p, _, _, _ = self.snap_traj.evaluate(t, segment_idx)
            if p is None:
                return 0.0
                
            current_pos = obs.get("pos", np.zeros(3)).flatten()
            return np.linalg.norm(current_pos - p)
        except Exception as e:
            self.debug_data["errors"].append(f"Tracking error computation error: {str(e)[:200]}")
            return 0.0

    def _hover_command(self, obs: Dict) -> NDArray:
        try:
            pos = obs.get("pos", np.zeros(3)).flatten()
            try:
                yaw = R.from_quat(obs["quat"].flatten()).as_euler('xyz')[2]
            except (KeyError, TypeError, ValueError):
                yaw = self.last_yaw
                
            return np.array([
                pos[0], pos[1], pos[2],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                yaw, 0.0, 0.0, 0.0
            ], dtype=np.float32)
        except Exception as e:
            self.debug_data["errors"].append(f"Hover command error: {str(e)[:200]}")
            return np.zeros(13, dtype=np.float32)

    def _estimate_path_length(self, waypoints: NDArray) -> float:
        try:
            if len(waypoints) < 2:
                return 0.0
            return np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        except Exception as e:
            self.debug_data["errors"].append(f"Path length estimation error: {str(e)[:200]}")
            return 0.0

    def step_callback(self, *args) -> bool:
        return self._finished