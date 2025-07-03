from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import casadi as ca
import minsnap_trajectories as ms
import lsy_drone_racing.utils.minsnap as minsnap
from lsy_drone_racing.control import Controller
import lsy_drone_racing.utils.trajectory as trajectory


if TYPE_CHECKING:
    from numpy.typing import NDArray


class MinSnapMPCController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._tick = 0
        self._finished = False
        self._freq = config.env.freq
        self.dt = 1.0 / self._freq
        print(f"DEBUG: Controller frequency set to {self._freq} Hz, dt = {self.dt:.3f} seconds.")
        
        # MPC parameters
        self._horizon = 10
        self._mpc_solver = None
        self._Q_pos = 15
        self._Q_vel = 2
        self._R_acc = 0.1
        self._u_max = 8.0
        self._X0_param = None
        self._X_ref_param = None
        self._X_var = None
        self._U_var = None
        
        # Trajectory parameters
        self._reference_trajectory = None
        self._t_total = 9.0  # Fixed total time
        
        # Load environment data
        self._gates = self._get_gate_positions(config)
        print(f"DEBUG: Gates loaded: {len(self._gates)} gates found.")
        self._obstacles = self._get_obstacles(config)
        print(f"DEBUG: Obstacles loaded: {len(self._obstacles)} obstacles found.")
        
        # Generate reference trajectory
        self._generate_reference_trajectory(obs)
        print("DEBUG: Reference trajectory generated with total time:", self._t_total)

    def _get_gate_positions(self, config: dict) -> list[dict]:
        track = config.env.track
        raw = getattr(track, "gates", None)
        if raw is None and isinstance(track, dict):
            raw = track.get("gates", [])
        gates = []
        for gate in raw or []:
            pos = np.array(getattr(gate, "pos", gate["pos"]))
            rpy = np.array(getattr(gate, "rpy", gate.get("rpy", [0, 0, 0])))
            gates.append({"pos": pos, "rpy": rpy})
        return gates

    def _get_obstacles(self, config: dict) -> list[dict]:
        track = config.env.track
        raw = getattr(track, "obstacles", None)
        if raw is None and isinstance(track, dict):
            raw = track.get("obstacles", [])
        obstacles = []
        for obs in raw or []:
            pos = np.array(getattr(obs, "pos", obs["pos"]))
            obstacles.append(
                {
                    "pos": pos,
                    "radius": getattr(obs, "radius", obs.get("radius", 0.25)),
                    "height": getattr(obs, "height", obs.get("height", 1.4)),
                }
            )
        return obstacles

    def _generate_reference_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Generate reference trajectory using minsnap optimization"""
        try:
            # Generate trajectory using the custom minsnap function
            polys = minsnap.generate_trajectory(self._make_refs(obs), self._t_total)
            print("DEBUG: Generated trajectory with minsnap optimization.")
            
            # Create time vector for evaluation
            t_eval = np.linspace(0, self._t_total, int(self._t_total * self._freq) + 1)
            n_points = len(t_eval)
            
            # Ensure polys has the right shape for our interpolation
            if polys.shape[0] != n_points:
                # Interpolate polynomials to match our time vector
                t_poly = np.linspace(0, self._t_total, polys.shape[0])
                
                pos_traj = np.zeros((n_points, 3))
                vel_traj = np.zeros((n_points, 3))
                acc_traj = np.zeros((n_points, 3))
                
                for i in range(3):
                    pos_traj[:, i] = np.interp(t_eval, t_poly, polys[:, i])
                    vel_traj[:, i] = np.interp(t_eval, t_poly, polys[:, 7+i])
                    acc_traj[:, i] = np.interp(t_eval, t_poly, polys[:, 14+i])
            else:
                pos_traj = polys[:, 0:3]
                vel_traj = polys[:, 7:10]
                acc_traj = polys[:, 14:17]
            
            # Generate yaw trajectory (simple interpolation between waypoints)
            current_yaw = obs.get("rpy", np.zeros(3))[2]
            yaw_refs = [current_yaw] + [gate["rpy"][2] for gate in self._gates]
            times_yaw = np.linspace(0, self._t_total, len(yaw_refs))
            yaw_traj = np.interp(t_eval, times_yaw, yaw_refs)
            
            # Store the reference trajectory
            self._reference_trajectory = {
                "t": t_eval,
                "pos": pos_traj,
                "vel": vel_traj,  
                "acc": acc_traj,
                "yaw": yaw_traj
            }
            
            # Set trajectory for GUI visualization - store in global trajectory module
            trajectory.trajectory = self._interpolate_trajectory_for_gui(polys)
            
            print(f"DEBUG: Reference trajectory created with {len(t_eval)} points")
            print(f"DEBUG: Position range: {np.min(pos_traj, axis=0)} to {np.max(pos_traj, axis=0)}")
            print(f"DEBUG: Max velocity: {np.max(np.linalg.norm(vel_traj, axis=1)):.2f} m/s")
            
        except Exception as e:
            print(f"ERROR: Trajectory generation failed: {e}")
            # Fallback: create simple linear trajectory
            self._create_fallback_trajectory(obs)

    def _interpolate_trajectory_for_gui(self, polys, interpolation_factor=2):
        """Interpolate trajectory for GUI visualization"""
        if interpolation_factor < 1:
            interpolation_factor = 1
        
        N = polys.shape[0]
        result = []
        
        for i in range(N - 1):
            a = polys[i]
            b = polys[i + 1]
            
            for k in range(interpolation_factor):
                alpha = k / interpolation_factor
                point = (1 - alpha) * a + alpha * b
                result.append(point)
        
        result.append(polys[-1])  # add the final point
        return np.array(result)

    def _create_fallback_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Create a simple fallback trajectory if minsnap fails"""
        print("DEBUG: Creating fallback trajectory")
        current_pos = obs["pos"]
        
        # Simple trajectory to first gate if available
        if self._gates:
            target_pos = self._gates[0]["pos"]
        else:
            target_pos = current_pos + np.array([1.0, 0.0, 0.5])  # Simple forward movement
        
        t_eval = np.linspace(0, self._t_total, int(self._t_total * self._freq) + 1)
        n_points = len(t_eval)
        
        # Linear interpolation from current to target
        pos_traj = np.zeros((n_points, 3))
        vel_traj = np.zeros((n_points, 3))
        acc_traj = np.zeros((n_points, 3))
        
        for i in range(3):
            pos_traj[:, i] = np.linspace(current_pos[i], target_pos[i], n_points)
            vel_traj[:, i] = (target_pos[i] - current_pos[i]) / self._t_total
        
        yaw_traj = np.zeros(n_points)
        
        self._reference_trajectory = {
            "t": t_eval,
            "pos": pos_traj,
            "vel": vel_traj,
            "acc": acc_traj,
            "yaw": yaw_traj
        }
        
        # Set fallback trajectory for GUI
        fallback_polys = np.zeros((n_points, 21))  # Assuming 21 columns for compatibility
        fallback_polys[:, 0:3] = pos_traj
        fallback_polys[:, 7:10] = vel_traj
        fallback_polys[:, 14:17] = acc_traj
        trajectory.trajectory = fallback_polys

    def _make_refs(self, obs):
        """Create waypoint references for minsnap trajectory generation"""
        current_pos = obs["pos"]
        
        # Get gate positions from observation if available
        if "gates_pos" in obs and len(obs["gates_pos"]) > 0:
            gates = obs["gates_pos"]
        else:
            gates = [gate["pos"] for gate in self._gates]
        
        # Ensure we have at least current position
        if not gates:
            print("WARNING: No gates found, using simple forward trajectory")
            gates = [current_pos + np.array([2.0, 0.0, 0.5])]
        
        waypoint1 = np.copy(current_pos)  # starting point
        print(f"DEBUG: Starting position: {waypoint1}")
        
        refs = [
            # Starting point with zero initial conditions
            ms.Waypoint(
                time=0.0,
                position=waypoint1,
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                jerk=np.array([0.0, 0.0, 0.0]),
            ),
        ]
        
        # Add waypoints for each gate
        times = [2.0, 4.0, 6.7, 9.0]  # Fixed times for gates
        
        for i, gate_pos in enumerate(gates[:4]):  # Limit to 4 gates
            waypoint = np.copy(gate_pos)
            
            # Apply small adjustments as in original code
            if i == 2:  # third gate
                waypoint[1] += 0.25
            elif i == 3:  # fourth gate  
                waypoint[1] -= 0.2
            
            # Add velocity constraints for some waypoints
            if i == 1:  # second gate
                refs.append(ms.Waypoint(
                    time=times[i], 
                    position=waypoint,
                    velocity=np.array([0.8, 0.8, 0.0])
                ))
            elif i == 2:  # third gate
                refs.append(ms.Waypoint(
                    time=times[i], 
                    position=waypoint,
                    velocity=np.array([0.0, 0.0, 0.0])
                ))
            else:
                refs.append(ms.Waypoint(time=times[i], position=waypoint))
            
            print(f"DEBUG: Gate {i+1} at {waypoint} at time {times[i]}")
        
        return refs

    def _setup_mpc(self) -> None:
        """Setup MPC optimization problem"""
        opti = ca.Opti()
        nx = 6  # [x, y, z, vx, vy, vz]
        nu = 3  # [ax, ay, az]
        N = self._horizon
        
        # Decision variables
        X = opti.variable(nx, N + 1)  # states
        U = opti.variable(nu, N)      # controls
        
        # Parameters
        X_ref = opti.parameter(nx, N + 1)  # reference trajectory
        X0 = opti.parameter(nx)            # initial state
        
        # Cost matrices
        Q = ca.diag([self._Q_pos, self._Q_pos, self._Q_pos, self._Q_vel, self._Q_vel, self._Q_vel])
        R = ca.diag([self._R_acc, self._R_acc, self._R_acc])
        
        # Objective function
        cost = 0
        for k in range(N):
            state_error = X[:, k] - X_ref[:, k]
            cost += ca.mtimes([state_error.T, Q, state_error])
            cost += ca.mtimes([U[:, k].T, R, U[:, k]])
        
        # Terminal cost
        terminal_error = X[:, N] - X_ref[:, N]
        Q_terminal = Q * 2.0
        cost += ca.mtimes([terminal_error.T, Q_terminal, terminal_error])
        
        opti.minimize(cost)
        
        # System dynamics (double integrator)
        A = ca.DM([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0], 
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        
        B = ca.DM([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0], 
            [self.dt, 0, 0],
            [0, self.dt, 0],
            [0, 0, self.dt]
        ])
        
        # Dynamics constraints
        for k in range(N):
            opti.subject_to(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])
        
        # Initial condition
        opti.subject_to(X[:, 0] == X0)
        
        # Obstacle avoidance constraints
        for obs in self._obstacles:
            center = obs["pos"]
            radius = obs["radius"]
            safety_margin = 0.1
            for k in range(N + 1):
                dist_xy = ca.sqrt(ca.sumsqr(X[0:2, k] - center[0:2]) + 1e-6)
                opti.subject_to(dist_xy >= radius + safety_margin)
        
        # Control and state constraints
        for k in range(N):
            opti.subject_to(U[:, k] >= -self._u_max)
            opti.subject_to(U[:, k] <= self._u_max)
            
            v_max = 5.0
            opti.subject_to(X[3:6, k] >= -v_max)
            opti.subject_to(X[3:6, k] <= v_max)
        
        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)
        
        # Store solver and parameters
        self._mpc_solver = opti
        self._X0_param = X0
        self._X_ref_param = X_ref
        self._X_var = X
        self._U_var = U
        
        print("DEBUG: MPC solver setup complete")

    def _interpolate_reference(self, current_time: float) -> NDArray[np.floating]:
        """Interpolate reference trajectory for MPC horizon"""
        if self._reference_trajectory is None:
            return np.zeros((6, self._horizon + 1))

        current_time = min(current_time, self._t_total)
        t_horizon = np.linspace(
            current_time, current_time + self._horizon * self.dt, self._horizon + 1
        )
        t_horizon = np.clip(t_horizon, 0, self._t_total)
        t_ref = self._reference_trajectory["t"]

        ref_pos = np.zeros((3, len(t_horizon)))
        ref_vel = np.zeros((3, len(t_horizon)))

        for i in range(3):
            ref_pos[i, :] = np.interp(t_horizon, t_ref, self._reference_trajectory["pos"][:, i])
            ref_vel[i, :] = np.interp(t_horizon, t_ref, self._reference_trajectory["vel"][:, i])

        return np.vstack([ref_pos, ref_vel])

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control action using MPC"""
        
        # Setup MPC solver if not done yet
        if self._mpc_solver is None:
            self._setup_mpc()

        current_time = self._tick * self.dt
        
        # Check if trajectory is finished
        if current_time >= self._t_total - 0.1:
            self._finished = True
            if self._reference_trajectory is not None:
                final_pos = self._reference_trajectory["pos"][-1]
                final_yaw = self._reference_trajectory["yaw"][-1]
                return np.concatenate((final_pos, np.zeros(3), np.zeros(3), [final_yaw, 0.0, 0.0, 0.0]), dtype=np.float32)
            else:
                return np.concatenate((obs["pos"], np.zeros(10)), dtype=np.float32)

        # Get current state
        current_pos = obs["pos"]
        current_vel = obs.get("vel", np.zeros(3))
        state_vec = np.concatenate([current_pos, current_vel])
        state_vec = np.clip(state_vec, -100.0, 100.0)
        
        # Get reference trajectory for MPC horizon
        ref_traj = self._interpolate_reference(current_time)
        
        # Solve MPC
        try:
            self._mpc_solver.set_value(self._X0_param, state_vec)
            self._mpc_solver.set_value(self._X_ref_param, ref_traj)
            sol = self._mpc_solver.solve()
            u_opt = sol.value(self._U_var[:, 0])
            
            # Get desired state from reference
            desired_pos = ref_traj[0:3, 0]
            desired_vel = ref_traj[3:6, 0]
            
            # Get yaw command
            yaw_cmd = np.interp(current_time, self._reference_trajectory["t"], self._reference_trajectory["yaw"])
            
            # Debug output
            if self._tick % (self._freq * 1) == 0:  # Every 1 second
                print(f"DEBUG t={current_time:.2f}: pos={current_pos}, des_pos={desired_pos}, u={u_opt}")
            
            return np.concatenate(
                (desired_pos, desired_vel, u_opt, [yaw_cmd, 0.0, 0.0, 0.0]), dtype=np.float32
            )

        except Exception as e:
            print(f"MPC solver failed at t={current_time:.3f}: {e}")
            return self._fallback_control(current_time, current_pos, current_vel)

    def _fallback_control(
        self, current_time: float, current_pos: NDArray, current_vel: NDArray
    ) -> NDArray[np.floating]:
        """Fallback control when MPC fails"""
        if self._reference_trajectory is None:
            return np.concatenate((current_pos, np.zeros(10)), dtype=np.float32)
        
        t_ref = self._reference_trajectory["t"]
        
        # Interpolate reference trajectory at current time
        ref_pos = np.zeros(3)
        ref_vel = np.zeros(3)
        ref_acc = np.zeros(3)
        
        for i in range(3):
            ref_pos[i] = np.interp(current_time, t_ref, self._reference_trajectory["pos"][:, i])
            ref_vel[i] = np.interp(current_time, t_ref, self._reference_trajectory["vel"][:, i])
            ref_acc[i] = np.interp(current_time, t_ref, self._reference_trajectory["acc"][:, i])
        
        # PD control
        pos_error = ref_pos - current_pos
        vel_error = ref_vel - current_vel
        kp = 8.0
        kd = 4.0
        control_acc = kp * pos_error + kd * vel_error + ref_acc
        control_acc = np.clip(control_acc, -self._u_max, self._u_max)
        
        yaw_cmd = np.interp(current_time, t_ref, self._reference_trajectory["yaw"])
        
        return np.concatenate((ref_pos, ref_vel, control_acc, [yaw_cmd, 0.0, 0.0, 0.0]), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        if self._tick % (self._freq * 2) == 0:
            current_time = self._tick * self.dt
            print(f"[DEBUG] Tick: {self._tick}, Time: {current_time:.2f}s, Pos: {obs['pos']}")
        return self._finished

    def reset(self):
        super().reset()
        self._tick = 0
        self._finished = False
        self._mpc_solver = None

    def get_trajectory_info(self) -> dict:
        if self._reference_trajectory is None:
            return {}
        current_time = self._tick * self.dt
        progress = min(current_time / self._t_total, 1.0) if self._t_total > 0 else 0.0
        return {
            "total_time": self._t_total,
            "current_time": current_time,
            "progress": progress,
            "num_gates": len(self._gates),
            "num_obstacles": len(self._obstacles),
            "max_velocity": np.max(np.linalg.norm(self._reference_trajectory["vel"], axis=1)) if self._reference_trajectory else 0,
            "max_acceleration": np.max(np.linalg.norm(self._reference_trajectory["acc"], axis=1)) if self._reference_trajectory else 0,
            "finished": self._finished,
        }