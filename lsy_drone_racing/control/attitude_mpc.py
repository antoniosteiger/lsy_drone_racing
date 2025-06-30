from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, norm_2
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
import logging

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ml_collections import ConfigDict

# ------------------------------------------------------------------------------
# |  Constants
# ------------------------------------------------------------------------------
MASS        = 0.5
GRAVITY_VEC = np.array([0.0, 0.0, -9.81])
SI_ACC      = [0.0, 5.0]
SI_PARAMS   = np.array([[-2, 3], [-2, 3], [-1, 2]])
THRUST_MIN  = 0.1
THRUST_MAX  = 1.0
OBSTACLE_RADIUS = 0.3  # default obstacle radius
GATE_RADIUS     = 0.3  # gate constraint radius

# ------------------------------------------------------------------------------
# |  Trajectory: Minimum Jerk via CubicSpline
# ------------------------------------------------------------------------------
class MinimumJerkTrajectory:
    def __init__(self, waypoints: NDArray[np.floating], times: NDArray[np.floating]):
        self.dim    = waypoints.shape[1]
        self.times  = times.copy()
        self.splines = [CubicSpline(times, waypoints[:, i], bc_type='clamped') for i in range(self.dim)]
        self.t_start = times[0]
        self.t_end   = times[-1]

    def evaluate(self, t: float) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        t_clipped = float(np.clip(t, self.t_start, self.t_end))
        pos = np.array([s(t_clipped)    for s in self.splines])
        vel = np.array([s(t_clipped, 1) for s in self.splines])
        acc = np.array([s(t_clipped, 2) for s in self.splines])
        return pos, vel, acc

# ------------------------------------------------------------------------------
# |  Quadrotor ODE Model
# ------------------------------------------------------------------------------
def export_quadrotor_ode_model() -> AcadosModel:
    model    = AcadosModel()
    model.name = "dynamic_mpc_quad"

    # states
    px, py, pz       = MX.sym('px'), MX.sym('py'), MX.sym('pz')
    vx, vy, vz       = MX.sym('vx'), MX.sym('vy'), MX.sym('vz')
    r, p, y          = MX.sym('r'), MX.sym('p'), MX.sym('y')
    states           = vertcat(px,py,pz, vx,vy,vz, r,p,y)

    # inputs
    rc, pc, yc, tc   = MX.sym('r_cmd'), MX.sym('p_cmd'), MX.sym('y_cmd'), MX.sym('thrust_cmd')
    inputs           = vertcat(rc, pc, yc, tc)

    # dynamics
    pos_dot = vertcat(vx, vy, vz)
    z_body  = vertcat(
        cos(r)*sin(p)*cos(y) + sin(r)*sin(y),
        cos(r)*sin(p)*sin(y) - sin(r)*cos(y),
        cos(r)*cos(p),
    )
    thrust  = SI_ACC[0] + SI_ACC[1]*tc
    vel_dot = thrust*z_body/MASS + GRAVITY_VEC
    rpy_dot = SI_PARAMS[:,0]*vertcat(r,p,y) + SI_PARAMS[:,1]*vertcat(rc,pc,yc)

    f_expl = vertcat(pos_dot, vel_dot, rpy_dot)
    model.x           = states
    model.u           = inputs
    model.f_expl_expr = f_expl
    model.f_impl_expr = None
    return model

# ------------------------------------------------------------------------------
# |  Build OCP with obstacle slack
# ------------------------------------------------------------------------------
def create_ocp_solver_with_obstacles(
    Tf: float, N: int,
    obstacles: List[Tuple[NDArray[np.floating], float]],
    gate_pos:   NDArray[np.floating]
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    ocp             = AcadosOcp()
    model           = export_quadrotor_ode_model()
    ocp.model       = model
    nx              = model.x.rows()
    nu_base         = model.u.rows()
    n_obs           = len(obstacles)
    nu              = nu_base + n_obs

    # horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = Tf

    # cost
    ny   = nx + nu
    ny_e = nx
    Q    = np.diag([10.0]*3 + [0.0]*3 + [0.0]*3)
    Rw   = np.diag([5.0,5.0,5.0,8.0])
    S    = np.eye(n_obs)*1000.0
    W    = scipy.linalg.block_diag(Q, Rw, S)
    W_e  = Q.copy()

    ocp.cost.cost_type   = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W   = W
    ocp.cost.W_e = W_e

    # selectors
    Vx = np.zeros((ny,nx)); Vx[:nx,:nx]=np.eye(nx)
    Vu = np.zeros((ny,nu)); Vu[nx:nx+nu_base,:nu_base]=np.eye(nu_base)
    Vs = np.zeros((ny,n_obs)); Vs[nx+nu_base:,:]=np.eye(n_obs)
    ocp.cost.Vx, ocp.cost.Vu, ocp.cost.Vs = Vx, Vu, Vs
    ocp.cost.Vx_e = np.vstack((np.eye(nx), np.zeros((ny_e-nx,nx))))

    # state bounds on rpy
    ocp.constraints.idxbx = np.array([6,7,8])
    ocp.constraints.lbx  = -np.pi/3*np.ones(3)
    ocp.constraints.ubx  =  np.pi/3*np.ones(3)

    # input bounds + slack
    idxbu = np.arange(nu)
    lbu   = np.concatenate(([-1.0]*3, [THRUST_MIN], [0.0]*n_obs))
    ubu   = np.concatenate(([ 1.0]*3, [THRUST_MAX], [10.0]*n_obs))
    ocp.constraints.idxbu, ocp.constraints.lbu, ocp.constraints.ubu = idxbu, lbu, ubu

    # initial state placeholder
    ocp.constraints.x0 = np.zeros(nx)

    # nonlinear constraints: gate + obstacles
    pos       = model.x[0:3]
    gate_dist = norm_2(pos - MX(gate_pos))
    nl_list   = []
    lnlb      = []
    lnub      = []
    for i,(c,r) in enumerate(obstacles):
        d_obs = norm_2(pos - MX(c))
        slack = model.u[4+i]
        nl_list.append(d_obs + slack)
        lnlb.append(r)
        lnub.append(1e6)
    ocp.constraints.nl_constr_expr = vertcat(*nl_list)
    ocp.constraints.lnlc           = np.array(lnlb)
    ocp.constraints.unlc           = np.array(lnub)

    # gate constraint
    # Correct terminal constraint
    ocp.constraints.constr_type_e = 'BGH'
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([GATE_RADIUS])
    ocp.constraints.h_e = gate_dist

    # solver opts
    opts = ocp.solver_options
    opts.qp_solver           = 'FULL_CONDENSING_HPIPM'
    opts.hessian_approx      = 'GAUSS_NEWTON'
    opts.integrator_type     = 'ERK'
    opts.nlp_solver_type     = 'SQP'
    opts.tol                 = 1e-5
    opts.qp_solver_iter_max  = 1000
    opts.nlp_solver_max_iter = 1000

    solver = AcadosOcpSolver(ocp, json_file='ocp.json', verbose=False)
    solver.n_obs     = n_obs
    solver.obstacles = obstacles
    solver.gate_pos  = gate_pos
    solver.gate_rad  = GATE_RADIUS
    return solver, ocp

# ------------------------------------------------------------------------------
# |  Dynamic Attitude MPC Controller
# ------------------------------------------------------------------------------
class AttitudeMPC(Controller):
    """
    Controller that tracks a minimum-jerk trajectory, avoids obstacles via slacks,
    and replans dynamically when obstacles or gate updates.
    """
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        super().__init__(obs, info, config)
        # MPC parameters
        self._N    = 30
        self._dt   = 1.0 / config.env.freq
        self._T    = self._N * self._dt
        

        # Extract gates & build waypoints
        gates = config.env.track.gates
        waypoints = np.array([gate.pos for gate in gates])
        times     = np.linspace(0.0, self._T, len(waypoints))

        self._gates = waypoints
        self._current_gate_idx = 0
        self._current_gate = self._gates[0]

        # Extract obstacles & radii
        obstacles_cfg = config.env.track.obstacles
        obstacles = [ ( np.array(o.pos), OBSTACLE_RADIUS ) for o in obstacles_cfg ]

        self._hover_thrust_cmd = (MASS * np.linalg.norm(GRAVITY_VEC) - SI_ACC[0]) / SI_ACC[1]
        

        # Trajectory generator
        self.traj = MinimumJerkTrajectory(waypoints, times)

        # Build solver
        self._solver, self._ocp = create_ocp_solver_with_obstacles(
            Tf = self._T,
            N  = self._N,
            obstacles=obstacles,
            gate_pos = self._current_gate,
        )

        # dims
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._tick = 0
        self._finished = False
        self._last_pos =np.zeros(3)

    def update_obstacles(self, new_obs: List[Tuple[NDArray[np.floating], float]]):
        self._solver, self._ocp = create_ocp_solver_with_obstacles(
            Tf=self._T, N=self._N,
            obstacles=new_obs, gate_pos=self._current_gate
        )

    def update_gate(self, new_gate: NDArray[np.floating]):
        remaining_gates = self._gates[self._current_gate_idx+1:]
        if len(remaining_gates) > 0:
            waypoints = np.vstack([self._last_pos, remaining_gates])
            times     = np.linspace(0.0, self._T, len(waypoints))
            self.traj = MinimumJerkTrajectory(waypoints, times)

        self._solver, self._ocp = create_ocp_solver_with_obstacles(
            Tf=self._T, N=self._N,
            obstacles=self._solver.obstacles, gate_pos=new_gate
        )

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None=None) -> NDArray[np.floating]:

        self._last_pos = obs['pos'].copy()

        #Update gate if needed
        if info and info.get("gates_passed", 0) > self._current_gate_idx:
            self._current_gate_idx = info["gates_passed"]
            if self._current_gate_idx < len(self._gates):
                self._current_gate = self._gates[self._current_gate_idx]
                self.update_gate(self._current_gate)
            else:
                self._finished = True
                return np.zeros(4)  # Return zero control if finished
            
        # state
        obs['rpy'] = R.from_quat(obs['quat']).as_euler('xyz')
        x0 = np.concatenate((obs['pos'], obs['vel'], obs['rpy']))
        self._solver.set(0,'lbx',x0)
        self._solver.set(0,'ubx',x0)

        # build refs
        for j in range(self._N):
            t_f = self._tick*self._dt + j*self._dt
            pos_ref, _, _ = self.traj.evaluate(t_f)
            yref = np.zeros(self._ny)
            yref[0:3] = pos_ref
            yref[8]   = 0.0                      # yaw reference
            yref[12]  = self._hover_thrust_cmd  # thrust reference
            self._solver.set(j,'yref',yref)

        # terminal
        t_f = self._tick*self._dt + self._T
        pos_ref, _, _ = self.traj.evaluate(t_f)
        yref_e = np.zeros(self._nx)
        yref_e[0:3] = pos_ref
        self._solver.set(self._N,'yref',yref_e)

        status = self._solver.solve()
        if status != 0:
            logger.warning(f"MPC solve failed {status}")

        u_opt = self._solver.get(0,'u')
        self._tick += 1
        # return [thrust, r, p, y]
        return np.array([u_opt[3], u_opt[0], u_opt[1], u_opt[2]])

    def step_callback(self, *args, **kwargs) -> bool:
        return self._finished

    def episode_callback(self) -> float:
        self._tick = 0
        self._finished = False

        gates_passed = self._info.get("gates_passed", 0)
        flight_time  = self._info.get("flight_time", 0.0)

        logger.info(f"Gates passed: {gates_passed}")
        logger.info(f"Flight time: {flight_time:.2f} seconds")

        # Return something useful so Fire prints it
        return flight_time
