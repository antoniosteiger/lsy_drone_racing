from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import casadi as ca
import minsnap_trajectories as ms
import lsy_drone_racing.utils.minsnap as minsnap
from lsy_drone_racing.control import Controller
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import RotationSpline
import lsy_drone_racing.utils.trajectory as trajectory


if TYPE_CHECKING:
    from numpy.typing import NDArray


class MinSnapMPCController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._tick = 0
        self._finished = False
        # print("MinSnapMPCController initialized")
        self._freq = config.env.freq
        self.dt = 1.0 / self._freq
        # print(f"DEBUG: Controller frequency set to {self._freq} Hz, dt = {self.dt:.3f} seconds.")
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
        self._reference_trajectory = None
        self._trajectory_start_time = 0.0
        self._t_total = 0
        self._approach_distance = 1
        self._exit_distance = 0.5
        self._gates = self._get_gate_positions(config)
        print(f"DEBUG: Gates loaded: {len(self._gates)} gates found.")
        self._obstacles = self._get_obstacles(config)
        print(f"DEBUG: Obstacles loaded: {len(self._obstacles)} obstacles found.")
        self._waypoints = self._generate_waypoints()
        print("DEBUG: waypoints:", self._waypoints)
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
        # print(f"DEBUG: found {len(raw) if raw else 0} raw obstacles in config")
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
        # print(f"DEBUG: Loaded {len(obstacles)} obstacles → {obstacles}")
        return obstacles

    def _generate_waypoints(self) -> list[dict]:
        if not self._gates:
            return []
        way_points = []
        for i, gate in enumerate(self._gates):
            gate_pos = gate["pos"]
            gate_rpy = gate["rpy"]

            way_points.append({"pos": gate_pos, "rpy": gate_rpy, "type": "gate"})

        return way_points

    def _calculate_trajectory_times(
        self, start_pos: np.ndarray, waypoints: list, base_speed: float = 2.0
    ) -> np.ndarray:
        if len(waypoints) < 2:
            return np.array([0.0])
        # print("DEBUG: Calculating trajectory times for waypoints.")
        times = [0.0]
        current_time = 0.0
        min_time_increment = 0.3  # small nonzero duration
        prev_pos = start_pos
        for wp in waypoints:
            curr_pos = wp["pos"]
            distance = np.linalg.norm(curr_pos - prev_pos)
            segment_time = distance / base_speed
            segment_time = max(segment_time, min_time_increment)
            current_time += segment_time
            times.append(current_time)
            prev_pos = curr_pos
        # print(f"DEBUG: Trajectory times: {times}")
        assert np.all(np.diff(times) > 0), f"Times not strictly increasing: {times}"
        return np.array(times)

    def _generate_reference_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        # current_pos = obs["pos"]
        # current_vel = obs.get("vel", np.zeros(3))
        # current_yaw = obs.get("rpy", np.zeros(3))[2]
        # # print(f"DEBUG: Current position: {current_pos}, velocity: {current_vel}, yaw : {current_yaw}")
        # # WAYPOINT TIMES:
        # # times = self._calculate_trajectory_times(obs["pos"], self._waypoints, base_speed=2.0)
        # times = [0.0, 2.0, 4.0, 6.7, 9.0]
        # self._t_total = times[-1]
        # # print(f"DEBUG: Total trajectory time calculated: {self._t_total:.2f} seconds")
        # refs = []
        # refs.append(
        #     ms.Waypoint(
        #         position=current_pos, time=0.0, velocity=current_vel, acceleration=np.zeros(3)
        #     )
        # )
        # for wp, time in zip(self._waypoints, times[1:]):
        #     refs.append(ms.Waypoint(position=wp["pos"], time=time))
        # # print("DEBUG: Waypoints for trajectory generation:", refs)
        t_total = 9.0

        try:
            polys = minsnap.generate_trajectory(self.make_refs(obs), t_total)

            print("DEBUG: Generated trajectory with degree 7 and continuous orders 3.")
        except Exception as e:
            print(f"DEBUG: Trajectory generation failed: {e}")
            raise

        t_eval = np.linspace(0, t_total, int(t_total * self._freq) + 1)
        # pva = ms.compute_trajectory_derivatives(polys, t_eval, 3)
        # print(f"DEBUG: Computed trajectory derivatives at {len(t_eval)} time points.")

        # yaw_refs = [current_yaw]
        # yaw_refs += [wp["rpy"][2] for wp in self._waypoints]
        # yaw_traj = np.interp(t_eval, times, yaw_refs)

        # Let the sim know the trajectory for visualization
        trajectory.trajectory = interpolate_trajectory_linear(polys, 2)

        self._reference_trajectory = {
            "t": t_eval,
            "pos": polys[:, 0:3],
            "vel": polys[:, 7:10],
            "acc": polys[:, 14:17],
        }

    def make_refs(self, obs):
        current_pos = obs["pos"]
        gates = obs["gates_pos"]
        # self.current_gates_pos[2][1] += 0.13
        # self.current_gates_pos[3][1] -= 0.2
        waypoint1 = np.copy(current_pos)  # starting point
        # waypoint1[2] += 0.13 # clear the ground
        # waypoint1[1] -= 0.2 # clear the ground
        print("got pos")
        waypoint2 = np.copy(gates[0])  # first gate

        waypoint3 = np.copy(gates[1])  # second gate

        waypoint4 = np.copy(gates[2])  # third gate
        waypoint4[1] += 0.25  # increased y to "touch gate"

        waypoint5 = np.copy(gates[3])  # fourth gate
        waypoint5[1] -= 0.2  # increased y to meet velocity threshold

        refs = [
            # starting point
            ms.Waypoint(
                time=0.0,
                position=np.array(waypoint1),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                jerk=np.array([0.0, 0.0, 0.0]),
            ),
            # first gate
            ms.Waypoint(  # Any higher-order derivatives
                time=2.0,
                position=np.array(waypoint2),
                # velocity=np.array([-0.6, -0.6, 0.0]),
            ),
            # intermediary
            # ms.Waypoint(  # Any higher-order derivatives
            #     time= 8.0,
            #     position=np.array([0.35, -1.7, 0.85]),
            # ),
            # second gate
            ms.Waypoint(time=4.0, position=np.array(waypoint3), velocity=np.array([0.8, 0.8, 0.0])),
            # third gate
            ms.Waypoint(
                time=6.7,
                position=np.array(waypoint4),  # increased y to "touch gate"
                velocity=np.array([0.0, 0.0, 0.0]),
            ),
            # fourth gate
            ms.Waypoint(time=9.0, position=np.array(waypoint5)),
            # # endpoint
            # ms.Waypoint(
            #     time= self._t_total,
            #     position=np.array([-0.6, -0.4, 1.11]),
            # )
        ]

        return refs

    def _setup_mpc(self) -> None:
        opti = ca.Opti()
        nx = 6
        nu = 3
        N = self._horizon
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        X_ref = opti.parameter(nx, N + 1)
        Q = ca.diag([self._Q_pos, self._Q_pos, self._Q_pos, self._Q_vel, self._Q_vel, self._Q_vel])
        R = ca.diag([self._R_acc, self._R_acc, self._R_acc])
        cost = 0
        for k in range(N):
            state_error = X[:, k] - X_ref[:, k]
            cost += ca.mtimes([state_error.T, Q, state_error])
            cost += ca.mtimes([U[:, k].T, R, U[:, k]])
        terminal_error = X[:, N] - X_ref[:, N]
        Q_terminal = Q * 2.0
        cost += ca.mtimes([terminal_error.T, Q_terminal, terminal_error])
        opti.minimize(cost)
        A = ca.DM(
            [
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        B = ca.DM(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [self.dt, 0, 0], [0, self.dt, 0], [0, 0, self.dt]]
        )
        for k in range(N):
            opti.subject_to(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])
        X0 = opti.parameter(nx)
        opti.subject_to(X[:, 0] == X0)
        for obs in self._obstacles:
            center = obs["pos"]
            radius = obs["radius"]
            safety_margin = 0.1
            for k in range(N + 1):
                dist_xy = ca.sqrt(ca.sumsqr(X[0:2, k] - center[0:2]) + 1e-6)
                opti.subject_to(dist_xy >= radius + safety_margin)
        for k in range(N):
            opti.subject_to(U[:, k] >= -self._u_max)
            opti.subject_to(U[:, k] <= self._u_max)
            v_max = 5.0
            opti.subject_to(X[3:6, k] >= -v_max)
            opti.subject_to(X[3:6, k] <= v_max)
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)
        self._mpc_solver = opti
        self._X0_param = X0
        self._X_ref_param = X_ref
        self._X_var = X
        self._U_var = U

    def _interpolate_reference(self, current_time: float) -> NDArray[np.floating]:
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
            ref_pos[i, :] = np.interp(t_horizon, t_ref, self._reference_trajectory["pos"][i, :])
            ref_vel[i, :] = np.interp(t_horizon, t_ref, self._reference_trajectory["vel"][i, :])

        return np.vstack([ref_pos, ref_vel])

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        if self._mpc_solver is None:
            self._setup_mpc()

        current_time = min(self._tick / self._freq, self._t_total)
        if current_time >= self._t_total - 0.1:
            self._finished = True
            final_pos = self._reference_trajectory["pos"][-1]
            return np.concatenate((final_pos, np.zeros(10)), dtype=np.float32)
        current_pos = obs["pos"]
        current_vel = obs["vel"] if "vel" in obs else np.zeros(3)
        state_vec = np.concatenate([current_pos, current_vel])
        state_vec = np.clip(state_vec, -100.0, 100.0)
        assert not np.isnan(state_vec).any(), "Initial state has NaNs"
        ref_traj = self._interpolate_reference(current_time)
        # print(f"DEBUG: Reference trajectory interpolated at time {current_time:.3f}s, shape: {ref_traj.shape}")
        assert not np.isnan(ref_traj).any(), "Reference trajectory has NaNs"
        try:
            self._mpc_solver.set_value(self._X0_param, state_vec)
            self._mpc_solver.set_value(self._X_ref_param, ref_traj)
            sol = self._mpc_solver.solve()
            u_opt = sol.value(self._U_var[:, 0])
            desired_pos = ref_traj[0:3, 0]
            desired_vel = ref_traj[3:6, 0]

            yaw_cmd = np.interp(
                current_time, self._reference_trajectory["t"], self._reference_trajectory["yaw"]
            )

            return np.concatenate(
                (desired_pos, desired_vel, u_opt, [yaw_cmd, 0.0, 0.0, 0.0]), dtype=np.float32
            )

        except Exception as e:
            print(f"MPC solver failed at t={current_time:.3f}: {e}")
            return self._fallback_control(current_time, current_pos, current_vel)

    def _fallback_control(
        self, current_time: float, current_pos: NDArray, current_vel: NDArray
    ) -> NDArray[np.floating]:
        if self._reference_trajectory is None:
            return np.concatenate((current_pos, np.zeros(10)), dtype=np.float32)
        t_ref = self._reference_trajectory["t"]
        ref_pos = np.zeros(3)
        ref_vel = np.zeros(3)
        ref_acc = np.zeros(3)
        for i in range(3):
            ref_pos[i] = np.interp(current_time, t_ref, self._reference_trajectory["pos"][i, :])
            ref_vel[i] = np.interp(current_time, t_ref, self._reference_trajectory["vel"][i, :])
            ref_acc[i] = np.interp(current_time, t_ref, self._reference_trajectory["acc"][i, :])
        pos_error = ref_pos - current_pos
        vel_error = ref_vel - current_vel
        kp = 8.0
        kd = 4.0
        control_acc = kp * pos_error + kd * vel_error + ref_acc
        control_acc = np.clip(control_acc, -self._u_max, self._u_max)
        return np.concatenate((ref_pos, ref_vel, control_acc, np.zeros(4)), dtype=np.float32)

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
            print(f"[DEBUG] Tick: {self._tick}, Time: {self._tick / self._freq:.2f}s")
        return self._finished

    def reset(self):
        super().reset()
        self._tick = 0
        self._finished = False
        self._mpc_solver = None

    def get_trajectory_info(self) -> dict:
        if self._reference_trajectory is None:
            return {}
        current_time = self._tick / self._freq
        progress = min(current_time / self._t_total, 1.0) if self._t_total > 0 else 0.0
        return {
            "total_time": self._t_total,
            "current_time": current_time,
            "progress": progress,
            "num_waypoints": len(self._waypoints),
            "num_gates": len(self._gates),
            "num_obstacles": len(self._obstacles),
            "max_velocity": np.max(np.linalg.norm(self._reference_trajectory["vel"], axis=1)),
            "max_acceleration": np.max(np.linalg.norm(self._reference_trajectory["acc"], axis=1)),
            "finished": self._finished,
        }


def generate_min_snap_trajectory_standalone(
    gate_points, times, freq=50, start_vel=None, start_acc=None, end_vel=None, end_acc=None
):
    if len(gate_points) != len(times):
        raise ValueError("Number of gate points must match number of times.")
    start_vel = start_vel or [0, 0, 0]
    start_acc = start_acc or [0, 0, 0]
    end_vel = end_vel or [0, 0, 0]
    end_acc = end_acc or [0, 0, 0]
    refs = []
    for i, (point, time) in enumerate(zip(gate_points, times)):
        if i == 0:
            refs.append(
                ms.Waypoint(time=time, position=point, velocity=start_vel, acceleration=start_acc)
            )
        elif i == len(gate_points) - 1:
            refs.append(
                ms.Waypoint(time=time, position=point, velocity=end_vel, acceleration=end_acc)
            )
        else:
            refs.append(ms.Waypoint(time=time, position=point))
    polys = ms.generate_trajectory(
        refs, degree=7, idx_minimized_orders=(4,), num_continuous_orders=3, algorithm="closed-form"
    )
    t = np.linspace(times[0], times[-1], int(freq * (times[-1] - times[0])))
    pva = ms.compute_trajectory_derivatives(polys, t, 3)
    return t, pva[0, ...].T, pva[1, ...].T, pva[2, ...].T


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
