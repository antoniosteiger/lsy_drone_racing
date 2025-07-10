from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import casadi as ca
import minsnap_trajectories as ms
import lsy_drone_racing.utils.minsnap as minsnap
from lsy_drone_racing.control import Controller
import lsy_drone_racing.utils.trajectory as trajectory
import traceback
#import lsy_drone_racing.utils.path_planner as PathPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray

class MinSnapMPCController(Controller):
    def __init__(self, obs: dict[str, 'NDArray[np.floating]'], info: dict, config: dict):
        print("=" * 50)
        print("DEBUG: Initializing MinSnapMPCController")
        print(f"DEBUG: obs keys: {list(obs.keys())}")
        print(f"DEBUG: obs shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()]}")
        print(f"DEBUG: info: {info}")
        print(f"DEBUG: config type: {type(config)}")
        
        try:
            super().__init__(obs, info, config)
            print("DEBUG: Super init successful")
        except Exception as e:
            print(f"DEBUG ERROR: Super init failed: {e}")
            traceback.print_exc()
            raise
        
        self._tick = 0
        self._finished = False
        
        try:
            self._freq = config.env.freq
            print(f"DEBUG: Frequency: {self._freq}")
        except Exception as e:
            print(f"DEBUG ERROR: Cannot get frequency from config: {e}")
            print(f"DEBUG: Config structure: {dir(config)}")
            if hasattr(config, 'env'):
                print(f"DEBUG: Config.env structure: {dir(config.env)}")
            self._freq = 50  # Default fallback
            print(f"DEBUG: Using fallback frequency: {self._freq}")
        
        self.dt = 1.0 / self._freq
        self._horizon = int(1.2 / self.dt)  # ~1.2s horizon for speed
        print(f"DEBUG: dt: {self.dt}, horizon: {self._horizon}")

        # Base weights
        self._Q_pos = 50
        self._Q_vel = 15
        self._Q_gate = 50
        self._R_acc = 10
        self._R_acc_ref = 50
        self._u_max = 8.0
        print(f"DEBUG: Weights initialized - Q_pos: {self._Q_pos}, Q_vel: {self._Q_vel}, Q_gate: {self._Q_gate}")

        # Sensing limits
        self._max_obs = 4
        self._max_gates = 4
        print(f"DEBUG: Max obstacles: {self._max_obs}, Max gates: {self._max_gates}")

        # MPC handles
        self._mpc_solver     = None
        self._X0_param       = None
        self._X_ref_param    = None
        self._obs_pos_param  = None
        self._obs_rad_param  = None
        self._gate_pos_param = None
        self._gate_wt_param  = None
        self._U_ref_param    = None
        self._Qp_param       = None
        self._Qv_param       = None
        self._Qg_param       = None
        print("DEBUG: MPC handles initialized to None")

        # Reference trajectory
        self._reference_trajectory = None
        self._t_total = 0.0

        # Gate timing from make_refs
        self._gate_times = [0.0, 2.0, 4.0, 6.7, 9.0]
        print(f"DEBUG: Gate times: {self._gate_times}")

        # Gate progression
        self._current_gate_index = 0
        self._gate_completion_threshold = 0.8
        self._gate_completion_buffer = 3
        self._completion_counter = 0
        self._use_time_based_gates = True
        print(f"DEBUG: Gate progression initialized - current index: {self._current_gate_index}")

        # Static placeholders
        try:
            self._gates = self._get_gate_positions(config)
            print(f"DEBUG: Gates extracted: {len(self._gates)} gates")
            for i, gate in enumerate(self._gates):
                print(f"DEBUG: Gate {i}: {gate}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to get gate positions: {e}")
            traceback.print_exc()
            self._gates = []
            
        try:
            self._obstacles = self._get_obstacles(config)
            print(f"DEBUG: Obstacles extracted: {len(self._obstacles)} obstacles")
            for i, obs in enumerate(self._obstacles):
                print(f"DEBUG: Obstacle {i}: {obs}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to get obstacles: {e}")
            traceback.print_exc()
            self._obstacles = []

        try:
            print("DEBUG: Generating reference trajectory...")
            self._generate_reference_trajectory({'pos':np.array([1.0, 1.5, 0.07])})
            print("DEBUG: Reference trajectory generated successfully")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to generate reference trajectory: {e}")
            traceback.print_exc()
            raise

        print("DEBUG: MinSnapMPCController initialization complete")
        print("=" * 50)

    def _get_gate_positions(self, config):
        print("DEBUG: Getting gate positions")
        print(f"DEBUG: Config type: {type(config)}")
        print(f"DEBUG: Config attributes: {dir(config)}")
        
        try:
            raw = getattr(config.env.track, "gates", None)
            print(f"DEBUG: Raw gates from config.env.track.gates: {raw}")
        except Exception as e:
            print(f"DEBUG: Cannot access config.env.track.gates: {e}")
            raw = None
            
        if raw is None:
            try:
                if isinstance(config.env.track, dict):
                    raw = config.env.track.get("gates", [])
                    print(f"DEBUG: Raw gates from dict access: {raw}")
                else:
                    print(f"DEBUG: config.env.track is not a dict, type: {type(config.env.track)}")
                    raw = []
            except Exception as e:
                print(f"DEBUG ERROR: Failed dict access: {e}")
                raw = []
        
        if not raw:
            print("DEBUG WARNING: No gates found, using empty list")
            return []
            
        gates = []
        for i, g in enumerate(raw):
            try:
                if hasattr(g, 'pos'):
                    pos = np.array(g.pos)
                    print(f"DEBUG: Gate {i} position from attribute: {pos}")
                elif isinstance(g, dict) and 'pos' in g:
                    pos = np.array(g['pos'])
                    print(f"DEBUG: Gate {i} position from dict: {pos}")
                else:
                    print(f"DEBUG ERROR: Gate {i} has no position data: {g}")
                    continue
                gates.append(pos)
            except Exception as e:
                print(f"DEBUG ERROR: Failed to process gate {i}: {e}")
                continue
                
        print(f"DEBUG: Successfully extracted {len(gates)} gates")
        return gates

    def _get_obstacles(self, config):
        print("DEBUG: Getting obstacles")
        
        try:
            raw = getattr(config.env.track, "obstacles", None)
            print(f"DEBUG: Raw obstacles from config.env.track.obstacles: {raw}")
        except Exception as e:
            print(f"DEBUG: Cannot access config.env.track.obstacles: {e}")
            raw = None
            
        if raw is None:
            try:
                if isinstance(config.env.track, dict):
                    raw = config.env.track.get("obstacles", [])
                    print(f"DEBUG: Raw obstacles from dict access: {raw}")
                else:
                    raw = []
            except Exception as e:
                print(f"DEBUG ERROR: Failed dict access: {e}")
                raw = []
        
        if not raw:
            print("DEBUG: No obstacles found, using empty list")
            return []
            
        obstacles = []
        for i, o in enumerate(raw):
            try:
                if hasattr(o, 'pos'):
                    pos = np.array(o.pos)
                    radius = getattr(o, 'radius', 0.25)
                elif isinstance(o, dict):
                    pos = np.array(o['pos'])
                    radius = o.get('radius', 0.25)
                else:
                    print(f"DEBUG ERROR: Obstacle {i} has invalid format: {o}")
                    continue
                    
                obstacle = {"pos": pos, "radius": radius}
                obstacles.append(obstacle)
                print(f"DEBUG: Obstacle {i}: pos={pos}, radius={radius}")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to process obstacle {i}: {e}")
                continue
                
        print(f"DEBUG: Successfully extracted {len(obstacles)} obstacles")
        return obstacles

    def make_refs(self, obs):
        current_pos = obs["pos"]
        gates = self._gates
        
        waypoint1 = np.copy(current_pos)  # starting point
        waypoint2 = np.copy(gates[0])     # first gate
        waypoint3 = np.copy(gates[1])     # second gate
        waypoint4 = np.copy(gates[2])     # third gate
        waypoint4[1] += 0.25              # increased y to "touch gate"
        waypoint5 = np.copy(gates[3])     # fourth gate
        waypoint5[1] -= 0.2               # increased y to meet velocity threshold

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
            ms.Waypoint(
                time=2.0,
                position=np.array(waypoint2),
            ),
            # second gate
            ms.Waypoint(
                time=4.0,
                position=np.array(waypoint3),
                velocity=np.array([0.8, 0.8, 0.0])
            ),
            # third gate
            ms.Waypoint(
                time=6.7,
                position=np.array(waypoint4),
                velocity=np.array([0.0, 0.0, 0.0]),
            ),
            # fourth gate
            ms.Waypoint(
                time=9.0,
                position=np.array(waypoint5)
            ),
        ]
        
        print(f"DEBUG: Generated {len(refs)} waypoints for trajectory")
        for i, ref in enumerate(refs):
            print(f"  Waypoint {i}: t={ref.time:.1f}, pos={ref.position}")

        return refs

    def _generate_reference_trajectory(self, obs):
        print("DEBUG: Generating reference trajectory")
        
        try:
            refs = self.make_refs(obs)
            print(f"DEBUG: Reference waypoints created: {len(refs)}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to make refs: {e}")
            traceback.print_exc()
            raise
            
        try:
            t_total = 9.0
            print(f"DEBUG: Total time: {t_total}")
            
            polys = minsnap.generate_trajectory(refs, t_total)
            print(f"DEBUG: Generated trajectory polynomials shape: {polys.shape}")
            
            t_orig = np.linspace(0, t_total, len(polys))
            t_eval = np.linspace(0, t_total, int(t_total*self._freq)+1)
            print(f"DEBUG: Original time points: {len(t_orig)}, Eval time points: {len(t_eval)}")

            pos = polys[:,0:3]
            vel = polys[:,3:6] 
            acc = polys[:,6:9]
            print(f"DEBUG: Extracted pos shape: {pos.shape}, vel shape: {vel.shape}, acc shape: {acc.shape}")
            
            if len(polys) != len(t_eval):
                print(f"DEBUG: Interpolating trajectory from {len(polys)} to {len(t_eval)} points")
                pos_i = np.zeros((len(t_eval),3))
                vel_i = np.zeros_like(pos_i)
                acc_i = np.zeros_like(pos_i)
                
                for i in range(3):
                    pos_i[:,i] = np.interp(t_eval, t_orig, pos[:,i])
                    vel_i[:,i] = np.interp(t_eval, t_orig, vel[:,i])
                    acc_i[:,i] = np.interp(t_eval, t_orig, acc[:,i])
                    
                pos, vel, acc = pos_i, vel_i, acc_i
                print(f"DEBUG: Interpolated shapes - pos: {pos.shape}, vel: {vel.shape}, acc: {acc.shape}")

            self._t_total = t_eval[-1]
            print(f"DEBUG: Total trajectory time: {self._t_total}")
            
            # Set global trajectory (this line might be problematic)
            try:
                trajectory.trajectory = interpolate_trajectory_linear(polys, 2)
                print("DEBUG: Global trajectory set successfully")
            except Exception as e:
                print(f"DEBUG WARNING: Failed to set global trajectory: {e}")
                
            self._reference_trajectory = {"t":t_eval, "pos":pos.T, "vel":vel.T, "acc":acc.T}
            print(f"DEBUG: Reference trajectory stored with shapes:")
            print(f"DEBUG:   t: {self._reference_trajectory['t'].shape}")
            print(f"DEBUG:   pos: {self._reference_trajectory['pos'].shape}")
            print(f"DEBUG:   vel: {self._reference_trajectory['vel'].shape}")
            print(f"DEBUG:   acc: {self._reference_trajectory['acc'].shape}")
            
        except Exception as e:
            print(f"DEBUG ERROR: Failed to generate trajectory: {e}")
            traceback.print_exc()
            raise

    def _setup_mpc(self):
        print("DEBUG: Setting up MPC")
        
        try:
            opti = ca.Opti()
            nx, nu, N = 6, 3, self._horizon
            print(f"DEBUG: MPC dimensions - nx: {nx}, nu: {nu}, N: {N}")

            # Variables
            X = opti.variable(nx, N+1)
            U = opti.variable(nu, N)
            X_ref = opti.parameter(nx, N+1)
            print("DEBUG: MPC variables created")

            # Adaptive weight params
            Qp = opti.parameter()
            Qv = opti.parameter()
            Qg = opti.parameter()
            print("DEBUG: Weight parameters created")

            # Other params
            obs_pos = opti.parameter(3, self._max_obs)
            obs_rad = opti.parameter(self._max_obs)
            gate_pos = opti.parameter(3, self._max_gates)
            gate_wt = opti.parameter(self._max_gates)
            U_ref = opti.parameter(nu, N)
            print("DEBUG: Other parameters created")

            # Active gate
            active_gate = gate_pos @ gate_wt
            print("DEBUG: Active gate expression created")

            # Control weights
            R = ca.diag([self._R_acc]*3)
            Rr = ca.diag([self._R_acc_ref]*3)
            print("DEBUG: Control weight matrices created")

            # Cost
            print("DEBUG: Building cost function")
            cost = 0
            for k in range(N):
                e = X[:,k]-X_ref[:,k]
                cost += Qp*ca.sumsqr(e[0:3]) + Qv*ca.sumsqr(e[3:6])
                d = X[0:3,k]-active_gate
                cost += Qg*ca.sumsqr(d)
                cost += ca.mtimes([U[:,k].T, R, U[:,k]])
                er = U[:,k]-U_ref[:,k]
                cost += ca.mtimes([er.T, Rr, er])

            eN = X[:,N]-X_ref[:,N]
            cost += Qp*2*ca.sumsqr(eN[0:3]) + Qv*2*ca.sumsqr(eN[3:6])
            opti.minimize(cost)
            print("DEBUG: Cost function built and set")

            # ZOH dynamics
            print("DEBUG: Setting up dynamics")
            I3, dt = ca.DM.eye(3), self.dt
            A = ca.DM.zeros((6,6))
            A[0:3,0:3] = I3
            A[0:3,3:6] = dt*I3
            A[3:6,3:6] = I3
            
            B = ca.DM.zeros((6,3))
            B[0:3,0:3] = 0.5*dt**2*I3
            B[3:6,0:3] = dt*I3
            
            for k in range(N):
                opti.subject_to(X[:,k+1] == A@X[:,k] + B@U[:,k])
            print("DEBUG: Dynamics constraints added")

            # Initial condition
            X0 = opti.parameter(nx)
            opti.subject_to(X[:,0] == X0)
            print("DEBUG: Initial condition constraint added")

            # Obstacles
            print("DEBUG: Adding obstacle constraints")
            margin = 0.1
            for k in range(N+1):
                for i in range(self._max_obs):
                    dxy = ca.sqrt(ca.sumsqr(X[0:2,k]-obs_pos[0:2,i])+1e-8)
                    opti.subject_to(dxy >= obs_rad[i] + margin)
            print("DEBUG: Obstacle constraints added")

            # Limits
            print("DEBUG: Adding input/state limits")
            for k in range(N):
                opti.subject_to(U[:,k] >= -self._u_max)
                opti.subject_to(U[:,k] <= self._u_max)
                vlim = 5.0
                opti.subject_to(X[3:6,k] >= -vlim)
                opti.subject_to(X[3:6,k] <= vlim)
            print("DEBUG: Limits added")

            # Solver options
            opts = {"ipopt.print_level":0,"print_time":0,
                    "ipopt.max_iter":200,"ipopt.tol":1e-6,
                    "ipopt.acceptable_tol":1e-4,
                    "ipopt.warm_start_init_point":"no"}
            opti.solver("ipopt", opts)
            print("DEBUG: Solver configured")

            # Store handles
            self._mpc_solver = opti
            self._X0_param = X0
            self._X_ref_param = X_ref
            self._Qp_param = Qp
            self._Qv_param = Qv
            self._Qg_param = Qg
            self._obs_pos_param = obs_pos
            self._obs_rad_param = obs_rad
            self._gate_pos_param = gate_pos
            self._gate_wt_param = gate_wt
            self._U_ref_param = U_ref
            self._X_var = X
            self._U_var = U
            
            print("DEBUG: MPC setup complete")
            
        except Exception as e:
            print(f"DEBUG ERROR: MPC setup failed: {e}")
            traceback.print_exc()
            raise

    def _interpolate_reference(self, t0):
        print(f"DEBUG: Interpolating reference at t0={t0}")
        
        traj = self._reference_trajectory
        if traj is None:
            print("DEBUG WARNING: No reference trajectory available")
            return np.zeros((6,self._horizon+1)), np.zeros((3,self._horizon))
            
        tfull = np.linspace(t0, t0+self._horizon*self.dt, self._horizon+1)
        tctrl = tfull[:-1]
        T = traj["t"]
        
        print(f"DEBUG: Interpolation time range: {tfull[0]:.3f} to {tfull[-1]:.3f}")
        print(f"DEBUG: Reference time range: {T[0]:.3f} to {T[-1]:.3f}")
        
        try:
            pos = np.vstack([np.interp(tfull, T, traj["pos"][i]) for i in range(3)])
            vel = np.vstack([np.interp(tfull, T, traj["vel"][i]) for i in range(3)])
            acc = np.vstack([np.interp(tctrl, T, traj["acc"][i]) for i in range(3)])
            
            print(f"DEBUG: Interpolated shapes - pos: {pos.shape}, vel: {vel.shape}, acc: {acc.shape}")
            return np.vstack([pos,vel]), acc
            
        except Exception as e:
            print(f"DEBUG ERROR: Interpolation failed: {e}")
            traceback.print_exc()
            return np.zeros((6,self._horizon+1)), np.zeros((3,self._horizon))

    def _get_time_gate_weight(self, t, gates):
        print(f"DEBUG: Getting time-based gate weight at t={t}")
        
        idx = 0
        for i, gt in enumerate(self._gate_times[1:], 1):
            if t < gt + 0.5:
                idx = i - 1
                break
        idx = min(idx, len(gates) - 1)
        
        w = np.zeros(self._max_gates)
        if idx < len(gates):
            w[idx] = 1.0
            
        print(f"DEBUG: Gate weight vector: {w}, active gate index: {idx}")
        return w

    def _get_adaptive_weights(self, pos, vel, gates):
        print(f"DEBUG: Computing adaptive weights")
        print(f"DEBUG: Position: {pos}, Velocity: {vel}")
        
        baseQp, baseQv, baseQg = 50, 15, 50
        
        if gates and self._current_gate_index < len(gates):
            d = np.linalg.norm(pos - gates[self._current_gate_index])
            print(f"DEBUG: Distance to current gate {self._current_gate_index}: {d}")
            
            if d < 2.0:
                Qp, Qv, Qg = baseQp*0.7, baseQv*1.2, baseQg*1.5
                print("DEBUG: Close to gate - reducing position weight, increasing velocity/gate weights")
            elif d > 4.0:
                Qp, Qv, Qg = baseQp*1.3, baseQv, baseQg*0.8
                print("DEBUG: Far from gate - increasing position weight, reducing gate weight")
            else:
                Qp, Qv, Qg = baseQp, baseQv, baseQg
                print("DEBUG: Medium distance - using base weights")
        else:
            Qp, Qv, Qg = baseQp*1.5, baseQv*1.2, 0
            print("DEBUG: No valid gates - focusing on trajectory tracking")
            
        s = np.linalg.norm(vel)
        print(f"DEBUG: Speed: {s}")
        
        if s > 3.0:
            Qv *= 1.3
            Qp *= 1.1
            print("DEBUG: High speed - increasing velocity/position weights")
        elif s < 1.0:
            Qg *= 1.2
            print("DEBUG: Low speed - increasing gate weight")
            
        print(f"DEBUG: Final weights - Qp: {Qp}, Qv: {Qv}, Qg: {Qg}")
        return Qp, Qv, Qg

    def compute_control(self, obs, info=None):
        print(f"\nDEBUG: ===== Computing control at tick {self._tick} =====")

        # RERUN THE PATH PLANNER (PSEUDOCODE):
        # if is_obs_different(gates_pos): # from observation
        #       traj = pp.plan(gates_pos, gates_rpy, obstacles_pos, ...)
        #       trajectory.trajectory = traj
        
        if self._mpc_solver is None:
            print("DEBUG: MPC solver not initialized, setting up...")
            try:
                self._setup_mpc()
                print("DEBUG: MPC solver setup successful")
            except Exception as e:
                print(f"DEBUG ERROR: MPC setup failed: {e}")
                return np.zeros(13, dtype=np.float32)

        t = min(self._tick / self._freq, self._t_total)
        print(f"DEBUG: Current time: {t:.3f}, Total time: {self._t_total:.3f}")
        
        if t >= self._t_total - 0.1:
            print("DEBUG: Trajectory finished, returning final position")
            self._finished = True
            if self._reference_trajectory is not None:
                fpos = self._reference_trajectory["pos"][:, -1]
                return np.concatenate((fpos, np.zeros(10)), dtype=np.float32)
            else:
                return np.zeros(13, dtype=np.float32)

        try:
            pos = obs["pos"]
            vel = obs.get("vel", np.zeros(3))
            print(f"DEBUG: Current state - pos: {pos}, vel: {vel}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to get state from obs: {e}")
            return np.zeros(13, dtype=np.float32)

        x0 = np.concatenate([pos, vel])
        print(f"DEBUG: Initial state x0: {x0}")

        try:
            (traj, acc_ref) = self._interpolate_reference(t)
            print(f"DEBUG: Reference interpolation successful")
        except Exception as e:
            print(f"DEBUG ERROR: Reference interpolation failed: {e}")
            return self._fallback_control(t, pos, vel)

        # Process obstacles
        print("DEBUG: Processing obstacles")
        obs_p = np.zeros((3, self._max_obs))
        obs_r = np.zeros(self._max_obs)
        obstacles = obs.get("obstacles_pos", [])
        print(f"DEBUG: Found {len(obstacles)} obstacles in obs")
        
        for i, o in enumerate(obstacles[:self._max_obs]):
            try:
                obs_p[:, i] = o["pos"]
                obs_r[i] = o.get("radius", 0.25)
                print(f"DEBUG: Obstacle {i}: pos={obs_p[:,i]}, radius={obs_r[i]}")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to process obstacle {i}: {e}")

        # Process gates
        print("DEBUG: Processing gates")
        gate_p = np.zeros((3, self._max_gates))
        gates = self._gates
        
        
        for i, g in enumerate(gates[:self._max_gates]):
            try:
                gate_p[:, i] = g
                print(f"DEBUG: Gate {i}: pos={gate_p[:,i]}")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to process gate {i}: {e}")

        # Compute adaptive weights and gate weights
        try:
            Qp, Qv, Qg = self._get_adaptive_weights(pos, vel, gates)
            gate_w = self._get_time_gate_weight(t, gates)
            print(f"DEBUG: Weights computed - Qp: {Qp}, Qv: {Qv}, Qg: {Qg}")
            print(f"DEBUG: Gate weights: {gate_w}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to compute weights: {e}")
            Qp, Qv, Qg = 50, 15, 50
            gate_w = np.zeros(self._max_gates)

        # Set MPC parameters
        print("DEBUG: Setting MPC parameters")
        try:
            s = self._mpc_solver
            s.set_value(self._X0_param, x0)
            s.set_value(self._X_ref_param, traj)
            s.set_value(self._Qp_param, Qp)
            s.set_value(self._Qv_param, Qv)
            s.set_value(self._Qg_param, Qg)
            s.set_value(self._obs_pos_param, obs_p)
            s.set_value(self._obs_rad_param, obs_r)
            s.set_value(self._gate_pos_param, gate_p)
            s.set_value(self._gate_wt_param, gate_w)
            s.set_value(self._U_ref_param, acc_ref)
            print("DEBUG: MPC parameters set successfully")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to set MPC parameters: {e}")
            traceback.print_exc()
            return self._fallback_control(t, pos, vel)

        # Warm start
        if hasattr(self, "_last_solution"):
            print("DEBUG: Applying warm start")
            try:
                
                Uw = self._last_solution["U"]
                
                Us = np.hstack([Uw[:, 1:], np.zeros((3, 1))])
                
                s.set_initial(self._U_var, Us)
                print("DEBUG: Warm start applied successfully")
            except Exception as e:
                print(f"DEBUG WARNING: Warm start failed: {e}")

        # Solve MPC
        print("DEBUG: Solving MPC")
        try:
            sol = s.solve()
            print("DEBUG: MPC solved successfully")
            
            Uopt = sol.value(self._U_var[:, 0])
            print(f"DEBUG: Optimal control: {Uopt}")
            
            self._last_solution = {"X": sol.value(self._X_var), "U": sol.value(self._U_var)}
            
            pd, pv = traj[0:3, 0], traj[3:6, 0]
            result = np.concatenate((pd, pv, Uopt, np.zeros(4)), dtype=np.float32)
            print(f"DEBUG: Returning control: {result}")
            return result
            
        except Exception as e:
            print(f"DEBUG ERROR: MPC solve failed: {e}")
            traceback.print_exc()
            return self._fallback_control(t, pos, vel)

    def _fallback_control(self, t, pos, vel):
        print(f"DEBUG: Using fallback control at t={t}")
        
        traj = self._reference_trajectory
        if traj is None:
            print("DEBUG WARNING: No reference trajectory for fallback")
            return np.concatenate((pos, np.zeros(10)), dtype=np.float32)
            
        try:
            T = traj["t"]
            pd = np.array([np.interp(t, T, traj["pos"][i]) for i in range(3)])
            vd = np.array([np.interp(t, T, traj["vel"][i]) for i in range(3)])
            ad = np.array([np.interp(t, T, traj["acc"][i]) for i in range(3)])
            
            print(f"DEBUG: Fallback desired state - pos: {pd}, vel: {vd}, acc: {ad}")
            
            e = pd - pos
            f = vd - vel
            u = 8.0 * e + 4.0 * f + ad
            u = np.clip(u, -self._u_max, self._u_max)
            
            print(f"DEBUG: Fallback control - error: {e}, vel_error: {f}, control: {u}")
            
            result = np.concatenate((pd, vd, u, np.zeros(4)), dtype=np.float32)
            print(f"DEBUG: Fallback returning: {result}")
            return result
            
        except Exception as e:
            print(f"DEBUG ERROR: Fallback control failed: {e}")
            traceback.print_exc()
            return np.concatenate((pos, np.zeros(10)), dtype=np.float32)

    def step_callback(self, *args, **kwargs):
        print(f"DEBUG: Step callback called, tick: {self._tick}, finished: {self._finished}")
        self._tick += 1
        return self._finished

    def reset(self):
        print("DEBUG: Resetting controller")
        super().reset()
        self._tick = 0
        self._finished = False
        self._mpc_solver = None
        self._completion_counter = 0
        if hasattr(self, "_last_solution"):
            delattr(self, "_last_solution")
        print("DEBUG: Controller reset complete")

def interpolate_trajectory_linear(trajectory, interpolation_factor=2):
    print(f"DEBUG: Interpolating trajectory with factor {interpolation_factor}")
    print(f"DEBUG: Input trajectory shape: {trajectory.shape}")
    
    try:
        N = trajectory.shape[0]
        result = []
        
        for i in range(N-1):
            a, b = trajectory[i], trajectory[i+1]
            for k in range(interpolation_factor):
                alpha = k / interpolation_factor
                result.append((1-alpha)*a + alpha*b)
        result.append(trajectory[-1])
        
        result_array = np.array(result)
        print(f"DEBUG: Interpolated trajectory shape: {result_array.shape}")
        return result_array
        
    except Exception as e:
        print(f"DEBUG ERROR: Trajectory interpolation failed: {e}")
        traceback.print_exc()
        return trajectory