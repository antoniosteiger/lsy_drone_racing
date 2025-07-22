from lsy_drone_racing.control import Controller
import numpy as np
from numpy.typing import NDArray
import lsy_drone_racing.utils.minsnap as minsnap
import lsy_drone_racing.utils.trajectory as trajectory
import minsnap_trajectories as ms
import casadi as ca

class MinSnapTracker(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        print("OBSERVATION KEYS at init:", list(obs.keys()))
        self._tick = 0
        self._finished = False

        # Settings
        self._t_total = 9.0
        self._freq = config.env.freq
        self._interpolation_factor = 2
        self.dt = 1 / self._freq
        self.horizon = 25


        self.Q_pos = 50
        self.Q_vel = 0
        self.R_acc = 0.1
        self.U_max = 8.0
        self.vel_max = 5.0

        self.gate_size = 0.4



        

        # generate trajectory
        self.current_gates_pos = np.copy(obs["gates_pos"])
        self.initial_pos = np.copy(obs["pos"])
        self.intial_vel = np.zeros(3, dtype=np.float32)
        self.gate_quats = obs["gates_quat"]
        self.obstacles_pos = obs["obstacles_pos"]
        self.obs_radius = 0.25
        self.safe_d = 0.02  # safe distance from obstacles


        #print(obs)
        trajectory.trajectory = minsnap.generate_trajectory(self.make_refs(), self._t_total)
        print("MINSNAP: Trajectory generated")

        # interpolate trajectory to regulate speed
        trajectory.trajectory = interpolate_trajectory_linear(trajectory.trajectory, self._interpolation_factor)
        print("MINSNAP: Trajectory interpolated")

        self.setup_mpc()
        print("MPC: Solver setup complete")
    
    def setup_mpc(self):
        opti = ca.Opti()
        nx, nu, N = 6, 3, self.horizon


        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        X_ref = opti.parameter(nx, N + 1)
        U_ref = opti.parameter(nu, N)
        X0 = opti.parameter(nx)

        A = ca.DM.eye(nx)
        A[0:3, 3:6] = self.dt * ca.DM.eye(3)
        B = ca.DM.zeros((nx, nu))
        B[0:3, :] = 0.5 * self.dt**2 * ca.DM.eye(3)
        B[3:6, :] = self.dt * ca.DM.eye(3)
        

        cost = 0
        for k in range(N):
            e = X[:, k] - X_ref[:, k]
            de = e[0:3]
            print("de: ", de)
            dv = e[3:6]
            print("dv: ", dv)
            u_err = U[:, k] - U_ref[:, k]
            cost += self.Q_pos * ca.sumsqr(de) + self.Q_vel * ca.sumsqr(dv)
            cost += self.R_acc * ca.sumsqr(u_err)

            # Add obstacle avoidance constraint
            for obs_pos in self.obstacles_pos:
                dist = ca.norm_2(X[0:3, k] - obs_pos)
                opti.subject_to(dist >= self.obs_radius + self.safe_d)  # min clearance
        

        eN = X[:, N] - X_ref[:, N]
        cost += self.Q_pos * 2 * ca.sumsqr(eN[0:3])
        opti.minimize(cost)

        for k in range(N):
            opti.subject_to(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])
            opti.subject_to(opti.bounded(-self.U_max, U[:, k], self.U_max))
            opti.subject_to(opti.bounded(-self.vel_max, X[3:6, k], self.vel_max))

        opti.subject_to(X[:, 0] == X0)
        opti.solver("ipopt", {"print_time": False, "ipopt.print_level": 0})

        self._mpc_solver = opti
        self.X_var = X
        self.U_var = U
        self.X_ref_param = X_ref
        self.U_ref_param = U_ref
        self._X0_param = X0

        


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        if self._tick >= len(trajectory.trajectory):
            self._finished = True
            return np.zeros(13, dtype=np.float32)
        else:
            position = trajectory.trajectory[self._tick, 0:3]
            attitude = trajectory.trajectory[self._tick, 3:7]
            velocity = trajectory.trajectory[self._tick, 7:10]
            thrust = trajectory.trajectory[self._tick, 10]
            thrust = np.array([thrust], dtype=np.float32)
            angular = trajectory.trajectory[self._tick, 11:14]
            acceleration = trajectory.trajectory[self._tick, 14:17]
            rpy = quat_to_euler(attitude)
            gates_pos = obs["gates_pos"]
            pos = obs["pos"]
            vel = obs["vel"]
            x0 = np.hstack([pos, vel])



            # Regenerate trajectory if the observation changes
            if self.is_obs_different(gates_pos):
                self.current_gates_pos = np.copy(gates_pos)
                self.gate_quats = obs["gates_quat"]
            
            #Build reference for horizon
            slice_end = self._tick + self.horizon + 1
            ref_h = trajectory.trajectory[self._tick:slice_end, :]

            X_r= ref_h[:, 0:6].T
            U_r = ref_h[0:self.horizon, 6:9].T




            #Set parameters
            self._mpc_solver.set_value(self._X0_param, x0)
            self._mpc_solver.set_value(self.X_ref_param, X_r)
            self._mpc_solver.set_value(self.U_ref_param, U_r)
            

            try:
                sol = self._mpc_solver.solve()
                X_sol = sol.value(self.X_var)
                U_sol = sol.value(self.U_var)

                desired_pos = X_sol[0:3, 1]
                desired_vel = X_sol[3:6, 1]
                desired_acc = U_sol[:, 0]
            except :
                print("MPC solver failed, using default control")
                desired_pos = position
                desired_vel = velocity


            return np.concatenate((desired_pos, desired_vel, desired_acc ,np.zeros(4)), dtype=np.float32)
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
        #print(obs["target_gate"])
        return self._finished
    
    def is_obs_different(self, gates_pos):
        for i in range(len(gates_pos)):
            if not np.array_equal(gates_pos[i], self.current_gates_pos[i]):
                return True
        return False
        
    def regenerate_trajectory(self):
        #print("MINSNAP: Regenerating trajectory")

        trajectory.trajectory = minsnap.generate_trajectory(self.make_refs(), self._t_total)
        trajectory.trajectory = interpolate_trajectory_linear(trajectory.trajectory, self._interpolation_factor)

        return
    
    def make_refs(self):
        #self.current_gates_pos[2][1] += 0.13
        #self.current_gates_pos[3][1] -= 0.2
        waypoint1 = self.initial_pos.copy() # starting point
        # waypoint1[2] += 0.13 # clear the ground
        # waypoint1[1] -= 0.2 # clear the ground

        waypoint2 = self.current_gates_pos[0].copy() # first gate

        waypoint3 = self.current_gates_pos[1].copy() # second gate

        waypoint4 = self.current_gates_pos[2].copy() # third gate
        waypoint4[1] += 0.25 # increased y to "touch gate"

        waypoint5 = self.current_gates_pos[3].copy() # fourth gate
        waypoint5[1] -= 0.2 # increased y to meet velocity threshold
        
        refs = [
            # starting point
            ms.Waypoint(
                time= 0.0,
                position=np.array(waypoint1),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                jerk=np.array([0.0, 0.0, 0.0])
            ),
            # first gate
            ms.Waypoint(  # Any higher-order derivatives
                time= 2.0,
                position=np.array(waypoint2),
                #velocity=np.array([-0.6, -0.6, 0.0]),
            ),
            # intermediary
            # ms.Waypoint(  # Any higher-order derivatives
            #     time= 8.0,
            #     position=np.array([0.35, -1.7, 0.85]),
            # ),
            # second gate
            ms.Waypoint( 
                time= 4.0,
                position=np.array(waypoint3),
                velocity=np.array([0.8, 0.8, 0.0])
            ),
            # third gate
            ms.Waypoint(
                time= 6.7,
                position=np.array(waypoint4), # increased y to "touch gate"
                velocity=np.array([0.0, 0.0, 0.0]),
            ),
            # fourth gate
            ms.Waypoint(
                time= self._t_total,
                position=np.array(waypoint5),
            ),
            # # endpoint
            # ms.Waypoint(
            #     time= self._t_total,
            #     position=np.array([-0.6, -0.4, 1.11]),
            # )
        ]

        return refs

    
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

def interpolate_trajectory_linear(trajectory, interpolation_factor=2):
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