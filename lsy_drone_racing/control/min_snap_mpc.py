from lsy_drone_racing.control import Controller
import numpy as np
from numpy.typing import NDArray
import lsy_drone_racing.utils.minsnap as minsnap
import lsy_drone_racing.utils.trajectory as trajectory
from scipy.spatial.transform import Rotation as R
import minsnap_trajectories as ms
import casadi as ca
from casadi import DM

class MinSnapTracker(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        print("OBSERVATION KEYS at init:", list(obs.keys()))
        self._tick = 0
        self._finished = False

        # Settings
        self._t_total = 9.0
        self._freq = config.env.freq
        self._interpolation_factor = 3
        self.dt = 1 / self._freq
        self.horizon = 15
        self.mass = 0.027

        # MPC parameters - TRAJECTORY FOLLOWING PRIORITIZED
        self.Q_pos = 200      # High trajectory tracking weight
        self.Q_vel = 10       # High velocity tracking weight
        self.R_acc = 0.1        # Control effort
        self.U_max = 8
        self.vel_max = 2

        # Gate navigation parameters
        self.gate_approach_distance = 0.4  # Start considering gate orientation at this distance
        self.gate_passage_offset = 0.3      # How far ahead/behind gate center to aim for
        self.gate_alignment_penalty = 5000  # Strong penalty for not aligning with gate orientation
        self.gate_opening_radius = 0.2      # Safe passage radius
        
        # Obstacle avoidance - ONLY for obstacles in trajectory path
        self.obstacle_penalty = 2000
        self.obs_radius = 0.3
        self.safe_distance = 0.05
        self.trajectory_look_ahead = 3      # Look further ahead on trajectory
        self.trajectory_width = 0.3         # Consider obstacles within this distance of trajectory path
        
        # Store initial conditions
        self.current_gates_pos = np.copy(obs["gates_pos"])
        self.original_gates_pos = np.copy(obs["gates_pos"])
        self.initial_pos = np.copy(obs["pos"])
        self.intial_vel = np.zeros(3, dtype=np.float32)
        
        # Use the correct key for gate quaternions
        if "gates_quat" in obs:
            self.gate_quats = obs["gates_quat"]
        else:
            print("WARNING: No gate quaternions found, using default orientations")
            # Default to gates facing along y-axis
            self.gate_quats = np.array([[0, 0, 0, 1]] * len(obs["gates_pos"]))
            
        self.obstacles_pos = np.copy(obs["obstacles_pos"])

        # Control mode flags
        self.use_mpc = False
        self.mpc_reason = ""
        
        # Generate trajectory with gate orientations
        print("MINSNAP: Generating trajectory with gate orientations...")
        trajectory.trajectory = minsnap.generate_trajectory(self.make_refs(), self._t_total)
        print("MINSNAP: Trajectory generated")

        # interpolate trajectory to regulate speed
        trajectory.trajectory = interpolate_trajectory_linear(trajectory.trajectory, self._interpolation_factor)
        print("MINSNAP: Trajectory interpolated")

        self.setup_mpc()
        print("MPC: Solver setup complete")
    
    def setup_mpc(self):
        """Setup MPC with proper gate orientation handling"""
        print("MPC: Setting up trajectory-first solver with gate orientation...")
        opti = ca.Opti()
        nx, nu, N = 6, 3, self.horizon
        g = 9.81
        
        # Decision variables
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        
        # Parameters
        X_ref = opti.parameter(nx, N + 1)
        U_ref = opti.parameter(nu, N)
        X0 = opti.parameter(nx)
        obstacles_param = opti.parameter(3, len(self.obstacles_pos))
        target_gate_pos = opti.parameter(3)
        gate_through_point = opti.parameter(3)  # Point to aim for when passing through gate
        
        # Mode flags
        gate_navigation_mode = opti.parameter(1)  # When approaching/passing through gate
        obstacle_avoidance_mode = opti.parameter(1)  # Only for trajectory-blocking obstacles
        blocking_obstacles = opti.parameter(len(self.obstacles_pos))  # Which obstacles block the path
        
        # Initialize parameters
        opti.set_value(X_ref, np.zeros((nx, N+1)))
        opti.set_value(U_ref, np.zeros((nu, N)))
        opti.set_value(X0, np.zeros(nx))
        opti.set_value(obstacles_param, self.obstacles_pos.T)
        opti.set_value(target_gate_pos, np.zeros(3))
        opti.set_value(gate_through_point, np.zeros(3))
        opti.set_value(gate_navigation_mode, 0)
        opti.set_value(obstacle_avoidance_mode, 0)
        opti.set_value(blocking_obstacles, np.zeros(len(self.obstacles_pos)))
        
        dt = self.dt
        
        # COST FUNCTION - TRAJECTORY FOLLOWING IS PRIMARY
        cost = 0
        for k in range(N):
            state_error = X[:, k] - X_ref[:, k]
            pos_error = state_error[0:3]
            vel_error = state_error[3:6]
            control_error = U[:, k] - U_ref[:, k]
            
            # PRIMARY: Trajectory following (always high priority)
            cost += self.Q_pos * ca.sumsqr(pos_error)
            cost += self.Q_vel * ca.sumsqr(vel_error)
            cost += self.R_acc * ca.sumsqr(control_error)
            
            # SECONDARY: Obstacle avoidance (only for trajectory-blocking obstacles)
            for i in range(len(self.obstacles_pos)):
                obstacle_pos = obstacles_param[:, i]
                dist_to_obstacle = ca.norm_2(X[0:3, k] - obstacle_pos)
                min_safe_dist = self.obs_radius + self.safe_distance
                
                # Only avoid obstacles that are marked as blocking the trajectory
                is_blocking = blocking_obstacles[i]
                obstacle_violation = ca.fmax(min_safe_dist - dist_to_obstacle, 0)
                obstacle_cost = obstacle_avoidance_mode * is_blocking * self.obstacle_penalty * obstacle_violation**2
                cost += obstacle_cost
            
            # TERTIARY: Gate navigation (aim for the correct point through the gate)
            if k < N-2:  # Only apply to earlier horizon steps
                # Instead of aligning to gate center, aim for the through-point
                gate_targeting_error = X[0:3, k] - gate_through_point
                gate_cost = gate_navigation_mode * self.gate_alignment_penalty * ca.sumsqr(gate_targeting_error)
                cost += gate_cost

        # Terminal cost (always favor reaching trajectory endpoints)
        terminal_error = X[:, N] - X_ref[:, N]
        cost += 2.0 * self.Q_pos * ca.sumsqr(terminal_error[0:3])
        cost += 2.0 * self.Q_vel * ca.sumsqr(terminal_error[3:6])
        
        opti.minimize(cost)

        # Dynamics constraints
        for k in range(N):
            a_total = U[:, k] + ca.DM([0, 0, g])
            
            pos_next = X[0:3, k] + dt * X[3:6, k] + 0.5 * dt**2 * a_total
            vel_next = X[3:6, k] + dt * a_total

            opti.subject_to(X[0:3, k + 1] == pos_next)
            opti.subject_to(X[3:6, k + 1] == vel_next)
            
            # Input constraints
            for i in range(nu):
                opti.subject_to(U[i, k] >= -self.U_max)
                opti.subject_to(U[i, k] <= self.U_max)
            
            # Velocity constraints
            for i in range(3):
                opti.subject_to(X[3+i, k] >= -self.vel_max)
                opti.subject_to(X[3+i, k] <= self.vel_max)
            
            # Thrust constraints
            opti.subject_to(U[2, k] >= -g-5)
            opti.subject_to(U[2, k] <= 15)

        # Initial condition constraint
        opti.subject_to(X[:, 0] == X0)
        
        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": 150,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.warm_start_init_point": "yes"
        }
        opti.solver("ipopt", opts)

        # Store solver and variables
        self._mpc_solver = opti
        self.X_var = X
        self.U_var = U
        self.X_ref_param = X_ref
        self.U_ref_param = U_ref
        self._X0_param = X0
        self._obstacles_param = obstacles_param
        self._target_gate_pos_param = target_gate_pos
        self._gate_through_point_param = gate_through_point
        self._gate_navigation_mode_param = gate_navigation_mode
        self._obstacle_avoidance_mode_param = obstacle_avoidance_mode
        self._blocking_obstacles_param = blocking_obstacles
        
        print("MPC: Gate-aware solver setup complete")

    def get_gate_frame_vectors(self, gate_quat):
        """Get gate normal and tangent vectors from quaternion"""
        # Handle different quaternion formats (xyzw vs wxyz)
        if len(gate_quat) == 4:
            # Assume xyzw format, convert to scipy format
            rotation = R.from_quat(gate_quat)  # scipy expects [x, y, z, w]
        else:
            raise ValueError(f"Invalid quaternion format: {gate_quat}")
        
        # Gate frame vectors - these might need adjustment based on your coordinate system
        # Try different orientations if the current one doesn't work
        local_normal = np.array([0, 1, 0])    # Direction to pass through gate (along y-axis)
        local_tangent1 = np.array([1, 0, 0])  # Gate width direction (x-axis)
        local_tangent2 = np.array([0, 0, 1])  # Gate height direction (z-axis)
        
        # Transform to world coordinates
        world_normal = rotation.apply(local_normal)
        world_tangent1 = rotation.apply(local_tangent1)
        world_tangent2 = rotation.apply(local_tangent2)
        
        return world_normal, world_tangent1, world_tangent2

    def get_gate_through_point(self, gate_pos, gate_quat, approach_from_behind=True):
        """Calculate the point to aim for when passing through the gate"""
        gate_normal, _, _ = self.get_gate_frame_vectors(gate_quat)
        
        # Aim for a point slightly ahead of the gate center in the direction of passage
        offset_distance = self.gate_passage_offset
        if approach_from_behind:
            # Aim for a point beyond the gate
            through_point = gate_pos + offset_distance * gate_normal
        else:
            # Aim for a point before the gate
            through_point = gate_pos - offset_distance * gate_normal
            
        return through_point

    def check_gate_navigation_need(self, obs) -> tuple[bool, str, int, np.ndarray]:
        """Check if gate navigation assistance is needed"""
        current_pos = obs["pos"]
        target_gate_idx = obs["target_gate"]
        
        if target_gate_idx >= len(self.current_gates_pos):
            return False, "No target gate", -1, np.zeros(3)
            
        target_gate_pos = self.current_gates_pos[target_gate_idx]
        gate_quat = self.gate_quats[target_gate_idx]
        
        # Distance to target gate
        dist_to_gate = np.linalg.norm(current_pos - target_gate_pos)
        
        # Only use gate navigation when approaching the gate
        if dist_to_gate > self.gate_approach_distance:
            return False, f"Gate {target_gate_idx} too far ({dist_to_gate:.2f}m)", target_gate_idx, np.zeros(3)
        
        # Get gate frame vectors
        gate_normal, _, _ = self.get_gate_frame_vectors(gate_quat)
        
        # Check which side of the gate we're on
        pos_to_gate = target_gate_pos - current_pos
        dot_product = np.dot(pos_to_gate, gate_normal)
        approach_from_behind = dot_product > 0
        
        # Calculate the through-point
        gate_through_point = self.get_gate_through_point(target_gate_pos, gate_quat, approach_from_behind)
        
        return True, f"Gate {target_gate_idx} navigation active (dist: {dist_to_gate:.2f}m)", target_gate_idx, gate_through_point

    def check_trajectory_blocking_obstacles(self, obs) -> tuple[bool, list, str]:
        """Check for obstacles that ACTUALLY block the planned trajectory path"""
        current_pos = obs["pos"]
        
        # Get trajectory points for look-ahead
        traj_len = len(trajectory.trajectory)
        look_ahead_end = min(self._tick + self.trajectory_look_ahead, traj_len - 1)
        
        if self._tick >= traj_len:
            return False, [], "No trajectory remaining"
        
        blocking_obstacles = []
        
        # Check each obstacle
        for i, obstacle_pos in enumerate(self.obstacles_pos):
            is_blocking = False
            
            # Check if obstacle intersects with the planned trajectory path
            for traj_idx in range(self._tick, look_ahead_end + 1):
                traj_point = trajectory.trajectory[traj_idx, 0:3]
                dist_to_obstacle = np.linalg.norm(traj_point - obstacle_pos)
                
                # Obstacle blocks trajectory if it's within the safe corridor
                collision_distance = self.obs_radius + self.safe_distance
                if dist_to_obstacle < collision_distance:
                    is_blocking = True
                    break
            
            # Also check if we're currently too close to this obstacle
            current_dist = np.linalg.norm(current_pos - obstacle_pos)
            if current_dist < self.obs_radius + self.safe_distance + 0.2:
                is_blocking = True
            
            if is_blocking:
                blocking_obstacles.append(i)
        
        if blocking_obstacles:
            reason = f"Obstacles {blocking_obstacles} blocking trajectory path"
            return True, blocking_obstacles, reason
        
        return False, [], "No trajectory-blocking obstacles"

    def should_use_mpc(self, obs) -> tuple[bool, str, bool, bool, list, np.ndarray]:
        """Decide when to use MPC - trajectory-first approach"""
        
        # Check for trajectory-blocking obstacles
        obstacles_blocking, blocking_obstacles, obstacle_reason = self.check_trajectory_blocking_obstacles(obs)
        
        # Check for gate navigation need
        gate_needs_nav, gate_reason, gate_idx, gate_through_point = self.check_gate_navigation_need(obs)
        
        # Use MPC when necessary
        use_mpc = obstacles_blocking or gate_needs_nav
        
        if obstacles_blocking and gate_needs_nav:
            reason = f"OBSTACLE AVOIDANCE + GATE NAVIGATION: {obstacle_reason} + {gate_reason}"
        elif obstacles_blocking:
            reason = f"OBSTACLE AVOIDANCE: {obstacle_reason}"
        elif gate_needs_nav:
            reason = f"GATE NAVIGATION: {gate_reason}"
        else:
            reason = "Following minsnap trajectory"
            
        return use_mpc, reason, gate_needs_nav, obstacles_blocking, blocking_obstacles, gate_through_point

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        if self._tick >= len(trajectory.trajectory):
            print(f"Trajectory finished at tick {self._tick}")
            self._finished = True
            return np.zeros(13, dtype=np.float32)
        
        # Check if MPC is needed
        self.use_mpc, self.mpc_reason, gate_mode, obstacle_mode, blocking_obstacles, gate_through_point = self.should_use_mpc(obs)
        
        print(f"Tick {self._tick}, target_gate: {obs['target_gate']}, MPC: {self.use_mpc}")
        if self.use_mpc:
            print(f"  Modes: Gate_nav={gate_mode}, Obstacle_avoid={obstacle_mode}")
            print(f"  Reason: {self.mpc_reason}")
            if gate_mode:
                print(f"  Gate through-point: {gate_through_point}")
        
        # Update current gate positions
        self.current_gates_pos = np.copy(obs["gates_pos"])
        
        if not self.use_mpc:
            # Follow minsnap trajectory (primary mode)
            if self._tick < len(trajectory.trajectory):
                position = trajectory.trajectory[self._tick, 0:3]
                velocity = trajectory.trajectory[self._tick, 3:6] if trajectory.trajectory.shape[1] > 3 else np.zeros(3)
                print(f"Following minsnap trajectory: pos={position}")
                return np.concatenate((position, np.zeros(10)), dtype=np.float32)
            else:
                return np.zeros(13, dtype=np.float32)
        
        # Use MPC for corrections
        print(f"🎯 Using MPC for corrections: {self.mpc_reason}")
        
        # Current state
        pos = obs["pos"]
        vel = obs["vel"]
        x0 = np.hstack([pos, vel])
        
        # Build reference trajectory
        traj_len = len(trajectory.trajectory)
        slice_start = self._tick
        slice_end = min(self._tick + self.horizon + 1, traj_len)
        
        if slice_start >= traj_len:
            return np.zeros(13, dtype=np.float32)
        
        ref_h = trajectory.trajectory[slice_start:slice_end, :]
        
        # Pad if necessary
        if ref_h.shape[0] < self.horizon + 1:
            last_point = trajectory.trajectory[-1:, :]
            padding_needed = self.horizon + 1 - ref_h.shape[0]
            padding = np.tile(last_point, (padding_needed, 1))
            ref_h = np.vstack([ref_h, padding])
        
        # Extract position and velocity references
        X_r = ref_h[:, 0:6].T
        
        # Compute reference accelerations
        if ref_h.shape[1] > 14:
            U_r = ref_h[0:self.horizon, 14:17].T
        else:
            U_r = np.zeros((3, self.horizon))
            for k in range(self.horizon):
                if k + 1 < ref_h.shape[0]:
                    vel_k = ref_h[k, 3:6]
                    vel_k1 = ref_h[k+1, 3:6]
                    acc_ref = (vel_k1 - vel_k) / self.dt
                    U_r[:, k] = acc_ref - np.array([0, 0, 9.81])
        
        # Get gate information
        target_gate_idx = obs["target_gate"]
        if target_gate_idx < len(self.current_gates_pos):
            target_gate_pos = self.current_gates_pos[target_gate_idx]
        else:
            target_gate_pos = pos
        
        # Create obstacle blocking mask
        obstacle_mask = np.zeros(len(self.obstacles_pos))
        for obs_idx in blocking_obstacles:
            obstacle_mask[obs_idx] = 1.0
        
        # Set MPC parameters
        try:
            self._mpc_solver.set_value(self._X0_param, x0)
            self._mpc_solver.set_value(self.X_ref_param, X_r)
            self._mpc_solver.set_value(self.U_ref_param, U_r)
            self._mpc_solver.set_value(self._obstacles_param, obs["obstacles_pos"].T)
            self._mpc_solver.set_value(self._target_gate_pos_param, target_gate_pos)
            self._mpc_solver.set_value(self._gate_through_point_param, gate_through_point)
            
            # Set mode flags
            self._mpc_solver.set_value(self._gate_navigation_mode_param, 1.0 if gate_mode else 0.0)
            self._mpc_solver.set_value(self._obstacle_avoidance_mode_param, 1.0 if obstacle_mode else 0.0)
            self._mpc_solver.set_value(self._blocking_obstacles_param, obstacle_mask)
            
            # Solve MPC
            sol = self._mpc_solver.solve()
            
            # Extract solution
            X_sol = sol.value(self.X_var)
            U_sol = sol.value(self.U_var)
            
            # Use MPC's recommended position (one step ahead)
            pos_cmd = X_sol[0:3, 1]
            vel_cmd = X_sol[3:6, 1]
            acc_cmd = U_sol[:, 0] + np.array([0, 0, 9.81])
            
            print(f"✅ MPC solved: pos_cmd={pos_cmd}")
            
            return np.concatenate((pos_cmd, vel_cmd, acc_cmd, np.zeros(4)), dtype=np.float32)
            
        except RuntimeError as e:
            print(f"❌ MPC solve failed: {e}")
            print("🔄 Falling back to minsnap trajectory")
            
            # Fallback: use minsnap trajectory
            if self._tick < len(trajectory.trajectory):
                position = trajectory.trajectory[self._tick, 0:3]
                velocity = trajectory.trajectory[self._tick, 3:6] if trajectory.trajectory.shape[1] > 3 else np.zeros(3)
                return np.concatenate((position, velocity, np.zeros(7)), dtype=np.float32)
            else:
                return np.zeros(13, dtype=np.float32)
    
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter."""
        self._tick += 1
        if self._tick % 10 == 0:
            mode_str = "🎯 MPC CORRECTIONS" if self.use_mpc else "📍 MINSNAP TRAJECTORY"
            print(f"Step {self._tick}, reward: {reward:.3f}, Mode: {mode_str}")
            if self.use_mpc:
                print(f"   MPC Reason: {self.mpc_reason}")
        return self._finished
    
    def make_refs(self):
        """Generate waypoints with gate orientation consideration"""
        waypoint1 = self.initial_pos.copy()
        
        # For each gate, calculate a point that considers the gate orientation
        gate_waypoints = []
        for i in range(len(self.current_gates_pos)):
            gate_pos = self.current_gates_pos[i].copy()
            gate_quat = self.gate_quats[i]
            
            # Get gate orientation
            gate_normal, _, _ = self.get_gate_frame_vectors(gate_quat)
            
            # For waypoint generation, aim for a point slightly before the gate
            # This ensures the trajectory approaches from the correct side
            approach_offset = 0.2  # 20cm before gate center
            waypoint_pos = gate_pos - approach_offset * gate_normal
            
            gate_waypoints.append(waypoint_pos)
        
        # Create waypoints
        refs = [
            ms.Waypoint(
                time=0.0,
                position=np.array(waypoint1),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                jerk=np.array([0.0, 0.0, 0.0])
            ),
            ms.Waypoint(
                time=2.0,
                position=np.array(gate_waypoints[0]),
            ),
            ms.Waypoint( 
                time=4.0,
                position=np.array(gate_waypoints[1]),
                velocity=np.array([0.6, 0.6, 0.0])
            ),
            ms.Waypoint(
                time=6.7,
                position=np.array(gate_waypoints[2]),
                velocity=np.array([0.0, 0.0, 0.0]),
            ),
            ms.Waypoint(
                time=self._t_total,
                position=np.array(gate_waypoints[3]),
            ),
        ]
        return refs

# Utility functions remain the same
def quat_to_rotmat(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x**2 + y**2)]
    ])
    return R

def quat_to_euler(q):
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def interpolate_trajectory_linear(trajectory, interpolation_factor=2):
    """Linearly interpolates between trajectory points."""
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

    result.append(trajectory[-1])
    return np.array(result)