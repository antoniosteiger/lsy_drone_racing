from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import casadi as ca
import minsnap_trajectories as ms
import lsy_drone_racing.utils.trajectory as trajectory
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SegMinSnapMPCController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._tick       = 0
        self._finished   = False
        self._freq       = config.env.freq
        self.dt          = 1.0 / self._freq

        # MPC params
        self._horizon    = 10
        self._Q_pos      = 15
        self._Q_vel      = 2
        self._R_acc      = 0.1
        self._u_max      = 8.0

        # thresholds for segment‑only replan
        self._nominal_speed      = 2.0   # m/s for duration estimates
        self._big_move_threshold = 0.6   # m to trigger segment replan

        # load static track info
        self._gates      = self._get_gate_positions(config)
        print(f"DEBUG: Loaded {len(self._gates)} gates.")
        self._obstacles  = self._get_obstacles(config)
        print(f"DEBUG: Loaded {len(self._obstacles)} obstacles.")

        # 1) build all segments at once from config gates
        self._segments: list[dict] = []
        self._generate_full_segmented_trajectory(obs["pos"])
        print(f"DEBUG: Built {len(self._segments)} segments, total time {self._t_total:.2f}s.")

        # stash for plotting
        trajectory.trajectory = self._stitch_full_trajectory(interpolation=2)

        # 2) one‑time MPC setup
        self._mpc_solver = None
        self._setup_mpc()
        print("DEBUG: MPC setup done.")

    def _get_gate_positions(self, config):
        track = config.env.track
        raw = getattr(track, "gates", None)
        if raw is None and isinstance(track, dict):
            raw = track.get("gates", [])
        gates = []
        for gate in raw or []:
            pos = np.array(getattr(gate, "pos", gate["pos"]))
            rpy = np.array(getattr(gate, "rpy", gate.get("rpy", [0,0,0])))
            gates.append({"pos": pos, "rpy": rpy})
        return gates

    def _get_obstacles(self, config):
        track = config.env.track
        raw = getattr(track, "obstacles", None)
        if raw is None and isinstance(track, dict):
            raw = track.get("obstacles", [])
        obs = []
        for o in raw or []:
            pos = np.array(getattr(o, "pos", o["pos"]))
            obs.append({
                "pos": pos,
                "radius": getattr(o, "radius", o.get("radius", 0.25)),
                "height": getattr(o, "height", o.get("height", 1.4)),
            })
        return obs

    def _generate_full_segmented_trajectory(self, start_pos: np.ndarray):
        # 1) Build waypoints (start + config gates)
        from minsnap_trajectories import Waypoint
        wps = [ Waypoint(0.0, start_pos, np.zeros(3), np.zeros(3), np.zeros(3)) ]
        for g in self._gates:
            wps.append(Waypoint(0.0, g["pos"], np.zeros(3), np.zeros(3), np.zeros(3)))

        # 2) Durations per segment
        durations = []
        for i in range(len(wps)-1):
            d = np.linalg.norm(wps[i+1].position - wps[i].position)
            durations.append(np.clip(d/self._nominal_speed, 0.5, 3.0))

        # 3) Solve each 2‑point min‑snap once
        t0 = 0.0
        self._segments.clear()
        for i, T in enumerate(durations):
            p0 = wps[i].shift_time(-t0)
            p1 = wps[i+1].shift_time(-t0)
            poly = ms.generate_trajectory([p0, p1], T)
            self._segments.append({"t0": t0, "t1": t0+T, "poly": poly})
            t0 += T
        self._t_total = t0

    def _stitch_full_trajectory(self, interpolation: int) -> np.ndarray:
        # Build a big (N×13) array: [pos(3), vel(3), acc(3), zeros(4)]
        points = []
        dt = 1/self._freq/interpolation
        t = 0.0
        while t < self._t_total:
            for seg in self._segments:
                if seg["t0"] <= t <= seg["t1"]:
                    lt = t - seg["t0"]
                    p, v, a = ms.evaluate_polynomial(seg["poly"], lt)
                    points.append(np.hstack((p, v, a, np.zeros(4))))
                    break
            t += dt
        points.append(points[-1])
        return np.array(points)

    def _interpolate_reference(self, current_time: float) -> NDArray[np.floating]:
        t_h = np.linspace(
            current_time,
            min(current_time + self._horizon*self.dt, self._t_total),
            self._horizon+1
        )
        ref = np.zeros((6, len(t_h)))
        for k, τ in enumerate(t_h):
            seg = next(s for s in self._segments if s["t0"] <= τ < s["t1"])
            lt = τ - seg["t0"]
            p, v, _ = ms.evaluate_polynomial(seg["poly"], lt)
            ref[:, k] = np.hstack((p, v))
        return ref

    def _check_gate_changes(self, obs):
        if "gates_pos" not in obs:
            return False, 0.0, None
        cur = obs["gates_pos"]
        if not hasattr(self, "_last_gate_positions"):
            self._last_gate_positions = cur.copy()
            return False, 0.0, None
        diffs = np.linalg.norm(cur - self._last_gate_positions, axis=1)
        idx  = int(np.argmax(diffs))
        md   = float(diffs.max())
        if md > self._big_move_threshold:
            self._last_gate_positions = cur.copy()
            return True, md, idx
        return False, md, None

    def _update_segment(self, idx: int, new_pos: np.ndarray):
        seg = self._segments[idx]
        # continuity: get previous endpoint
        if idx == 0:
            prev_p, prev_v, _ = ms.evaluate_polynomial(seg["poly"], seg["t1"]-seg["t0"])
        else:
            prev = self._segments[idx-1]
            prev_p, prev_v, _ = ms.evaluate_polynomial(prev["poly"], prev["t1"]-prev["t0"])
        from minsnap_trajectories import Waypoint
        p0 = Waypoint(0, prev_p, prev_v, np.zeros(3), np.zeros(3))
        p1 = Waypoint(0, new_pos, np.zeros(3), np.zeros(3), np.zeros(3))
        T  = seg["t1"] - seg["t0"]
        seg["poly"] = ms.generate_trajectory([p0, p1], T)

        # re‑stitch global for plotting
        trajectory.trajectory = self._stitch_full_trajectory(interpolation=2)

    def _setup_mpc(self):
        opti = ca.Opti()
        nx, nu, N = 6, 3, self._horizon
        X = opti.variable(nx, N+1)
        U = opti.variable(nu, N)
        X_ref = opti.parameter(nx, N+1)
        Q = ca.diag([self._Q_pos]*3 + [self._Q_vel]*3)
        R = ca.diag([self._R_acc]*3)

        cost = 0
        for k in range(N):
            e = X[:,k] - X_ref[:,k]
            cost += e.T @ Q @ e + U[:,k].T @ R @ U[:,k]
        eT = X[:,N] - X_ref[:,N]
        cost += eT.T @ (Q*2.0) @ eT
        opti.minimize(cost)

        A = ca.DM([[1,0,0,self.dt,0,0],
                   [0,1,0,0,self.dt,0],
                   [0,0,1,0,0,self.dt],
                   [0,0,0,1,0,0],
                   [0,0,0,0,1,0],
                   [0,0,0,0,0,1]])
        B = ca.DM([[0,0,0],[0,0,0],[0,0,0],
                   [self.dt,0,0],[0,self.dt,0],[0,0,self.dt]])
        for k in range(N):
            opti.subject_to(X[:,k+1] == A @ X[:,k] + B @ U[:,k])

        X0 = opti.parameter(nx)
        opti.subject_to(X[:,0] == X0)

        # obstacle constraints
        for o in self._obstacles:
            c, r = o["pos"], o["radius"]
            m = 0.1
            for k in range(N+1):
                d = ca.sqrt(ca.sumsqr(X[0:2,k] - c[0:2]) + 1e-6)
                opti.subject_to(d >= r + m)

        for k in range(N):
            opti.subject_to(ca.bounded(-self._u_max, U[:,k], self._u_max))
            v_max = 5.0
            opti.subject_to(ca.bounded(-v_max, X[3:6,k], v_max))

        opts = {
            "ipopt.print_level": 0,
            "print_time":         0,
            "ipopt.max_iter":     200,
            "ipopt.tol":          1e-6,
            "ipopt.acceptable_tol":1e-4,
            "ipopt.warm_start_init_point":"yes"
        }
        opti.solver("ipopt", opts)

        self._mpc_solver, self._X0_param, self._X_ref_param, self._X_var, self._U_var = \
            opti, X0, X_ref, X, U

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:
        # possibly patch one segment
        replan, md, idx = self._check_gate_changes(obs)
        if replan and idx is not None:
            self._update_segment(idx, obs["gates_pos"][idx])

        if self._mpc_solver is None:
            self._setup_mpc()

        tcur = min(self._tick/self._freq, self._t_total)
        if tcur >= self._t_total - 0.1:
            self._finished = True
            end = self._interpolate_reference(self._t_total)[:, -1]
            return np.concatenate((end, np.zeros(10)), dtype=np.float32)

        pos = obs["pos"]
        vel = obs.get("vel", np.zeros(3))
        x0  = np.clip(np.hstack((pos, vel)), -100, 100)

        ref = self._interpolate_reference(tcur)
        self._mpc_solver.set_value(self._X0_param, x0)
        self._mpc_solver.set_value(self._X_ref_param, ref)
        sol = self._mpc_solver.solve()

        u   = sol.value(self._U_var[:,0])
        dp, dv = ref[0:3,0], ref[3:6,0]
        self._tick += 1
        return np.concatenate((dp, dv, u, [0,0,0,0]), dtype=np.float32)

    # fallback, reset, step_callback, get_trajectory_info unchanged…
    def _fallback_control(self, current_time: float, current_pos: NDArray, current_vel: NDArray) -> NDArray[np.floating]:
        if self._reference_trajectory is None:
            return np.concatenate((current_pos, np.zeros(10)), dtype=np.float32)
        t_ref = self._reference_trajectory['t']
        ref_pos = np.zeros(3)
        ref_vel = np.zeros(3)
        ref_acc = np.zeros(3)
        for i in range(3):
            ref_pos[i] = np.interp(current_time, t_ref, self._reference_trajectory['pos'][i, :])
            ref_vel[i] = np.interp(current_time, t_ref, self._reference_trajectory['vel'][i, :])
            ref_acc[i] = np.interp(current_time, t_ref, self._reference_trajectory['acc'][i, :])
        pos_error = ref_pos - current_pos
        vel_error = ref_vel - current_vel
        kp = 8.0
        kd = 4.0
        control_acc = kp * pos_error + kd * vel_error + ref_acc
        control_acc = np.clip(control_acc, -self._u_max, self._u_max)
        return np.concatenate((ref_pos, ref_vel, control_acc, np.zeros(4)), dtype=np.float32)

    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        self._tick += 1
        if self._tick % (self._freq * 2) == 0:
            print(f"[DEBUG] Tick: {self._tick}, Time: {self._tick/self._freq:.2f}s")
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
            'total_time': self._t_total,
            'current_time': current_time,
            'progress': progress,
            'num_waypoints': len(self._waypoints),
            'num_gates': len(self._gates),
            'num_obstacles': len(self._obstacles),
            'max_velocity': np.max(np.linalg.norm(self._reference_trajectory['vel'], axis=1)),
            'max_acceleration': np.max(np.linalg.norm(self._reference_trajectory['acc'], axis=1)),
            'finished': self._finished
        }
