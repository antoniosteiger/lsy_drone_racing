import casadi as ca
import numpy as np
from lsy_drone_racing.control import Controller

class MPCTrajectoryController(Controller):
    """MPC-based trajectory controller for drone racing using CasADi."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Initialize the MPC controller."""
        super().__init__(obs, info, config)
        # Time parameters
        self.dt = 0.02  # 50 Hz
        self.N = 20  # Prediction horizon
        self.nx = 13  # Expanded state: [px, py, pz, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
        self.nu = 13  # Control inputs for all state derivatives
        # System matrices (to be defined based on expanded state)
        self.A, self.B = self._get_system_matrices()
        # Race parameters
        self.gates = config.get('gates', [])
        self.obstacles = config.get('obstacles', [])
        self.num_gates = len(self.gates)
        self.current_gate_idx = 0
        self._finished = False
        self.last_U = np.zeros(self.nu)  # For fallback on solver failure
        self.last_X_sol = None
        self.last_U_sol = None

    def _get_system_matrices(self):
        """Compute discrete-time dynamics matrices for the expanded state."""
        # This is a placeholder; actual dynamics need to be modeled for the full state
        dt = self.dt
        A = np.eye(self.nx)  # Simplified; replace with actual dynamics
        B = np.eye(self.nx) * dt  # Simplified; replace with actual control matrix
        return A, B

    def _is_gate_reached(self, pos: np.ndarray, gate_pos: np.ndarray) -> bool:
        """Check if drone is close to the gate."""
        return np.linalg.norm(pos - gate_pos) < 0.2

    def compute_control(self, obs: dict[str, np.ndarray], info: dict | None = None) -> np.ndarray:
        """Compute next desired state using MPC."""
        if self.current_gate_idx >= self.num_gates:
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        # Extract current state
        state = obs['state']
        x0 = state  # Assuming state is already 13-dimensional

        # Check gate progress
        target_gate = self.gates[self.current_gate_idx]
        if self._is_gate_reached(state[0:3], target_gate):
            self.current_gate_idx += 1
            if self.current_gate_idx >= self.num_gates:
                self._finished = True
                return np.zeros(13, dtype=np.float32)
            target_gate = self.gates[self.current_gate_idx]

        # Create a new Opti instance each time
        opti = ca.Opti()

        # Define variables
        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)

        # Cost function: minimize jerk and track reference
        cost = 0
        w_jerk = 1.0
        w_terminal = 100.0
        w_ref = 10.0

        # Generate a simple reference trajectory (straight line to gate)
        x_ref = np.linspace(x0[0:3], target_gate, self.N + 1).T

        for k in range(self.N):
            cost += w_jerk * ca.sumsqr(U[:, k])  # Minimize jerk
            cost += w_ref * ca.sumsqr(X[0:3, k] - x_ref[:, k])  # Track reference

        # Terminal cost to encourage reaching the gate
        cost += w_terminal * ca.sumsqr(X[0:3, self.N] - target_gate)

        opti.minimize(cost)

        # Dynamics constraints
        for k in range(self.N):
            x_next = self.A @ X[:, k] + self.B @ U[:, k]
            opti.subject_to(X[:, k + 1] == x_next)

        # Initial state constraint
        opti.subject_to(X[:, 0] == x0)

        # Obstacle avoidance for nearby obstacles
        threshold = 5.0  # Only consider obstacles within 5 meters
        for obs_pos in self.obstacles:
            if np.linalg.norm(x0[0:3] - obs_pos[:3]) < threshold:
                for k in range(self.N + 1):
                    dist = ca.norm_2(X[0:3, k] - obs_pos[:3])
                    opti.subject_to(dist >= obs_pos[3] + 0.1)

        # Warm-start if previous solution exists
        if self.last_X_sol is not None:
            X_guess = np.hstack([self.last_X_sol[:, 1:], self.last_X_sol[:, -1].reshape(-1, 1)])
            U_guess = np.hstack([self.last_U_sol[:, 1:], self.last_U_sol[:, -1].reshape(-1, 1)])
            opti.set_initial(X, X_guess)
            opti.set_initial(U, U_guess)

        # Solver options
        opts = {"ipopt.tol": 1e-3, "ipopt.max_iter": 50, "print_time": 0}
        opti.solver("ipopt", opts)

        # Solve the problem
        try:
            sol = opti.solve()
            u_opt = sol.value(U[:, 0])
            self.last_U = u_opt
            self.last_X_sol = sol.value(X)
            self.last_U_sol = sol.value(U)
        except RuntimeError:
            u_opt = self.last_U  # Fallback to last successful control

        # Compute next state
        x_next = self.A @ x0 + self.B @ u_opt
        return x_next.astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        """Check if finished."""
        return self._finished