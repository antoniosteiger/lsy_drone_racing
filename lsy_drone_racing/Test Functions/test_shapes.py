from types import SimpleNamespace
import numpy as np
from lsy_drone_racing.control.min_snap_mpc import MinSnapMPCController

# --- Dummy obs/info/config as before ---
obs = {"pos": np.array([0.0, 0.0, 0.0]), "vel": np.array([0.0, 0.0, 0.0])}
info = {}

env = SimpleNamespace(
    freq=50,
    track=SimpleNamespace(
        gates=[
            SimpleNamespace(pos=[0.45, -0.5, 0.56], rpy=[0.0, 0.0, 2.35]),
            SimpleNamespace(pos=[1.0, -1.05, 1.11], rpy=[0.0, 0.0, -0.78]),
            SimpleNamespace(pos=[0.0, 1.0, 0.56], rpy=[0.0, 0.0, 0.0]),
            SimpleNamespace(pos=[-0.5, 0.0, 1.11], rpy=[0.0, 0.0, 3.14])
        ],
        obstacles=[
            SimpleNamespace(pos=[1.0, 0.0, 1.4]),
            SimpleNamespace(pos=[0.5, -1.0, 1.4]),
            SimpleNamespace(pos=[0.0, 1.5, 1.4]),
            SimpleNamespace(pos=[-0.5, 0.5, 1.4])
        ]
    )
)
config = SimpleNamespace(env=env)

# Instantiate
controller = MinSnapMPCController(obs, info, config)

# 1) Gates, Obstacles, Waypoints
waypoints = controller._generate_waypoints()
print("Waypoints shape:", waypoints.shape)    # (num_gates, 3)

# 2) Trajectory times
times = controller._calculate_trajectory_times(
    np.vstack([obs["pos"], waypoints]), avg_speed=2.0
)
print("Times shape:", times.shape)            # (num_waypoints,)

# 3) Reference trajectory
traj = controller._reference_trajectory
print("Reference trajectory shapes:")
for key in ("t", "pos", "vel", "acc"):
    arr = traj[key]
    print(f"  {key:>3}:", arr.shape)
#    t: (N,)
#    pos: (N, 3)
#    vel: (N, 3)
#    acc: (N, 3)

# 4) MPC interpolation
ref_horizon = controller._interpolate_reference(0.0)
print("Interpolated reference shape:", ref_horizon.shape)  
#    (6, horizon+1) — 3 pos rows + 3 vel rows

# 5) One control call
action = controller.compute_control(obs)
print("Control output shape:", action.shape)  
#    (13,) — [pos(3), vel(3), acc(3), zeros(4)]
