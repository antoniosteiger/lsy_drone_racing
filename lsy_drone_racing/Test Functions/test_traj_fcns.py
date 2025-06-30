import numpy as np
from types import SimpleNamespace
from lsy_drone_racing.control.min_snap_mpc import MinSnapMPCController

# Dummy observation and info
obs = {
    "pos": np.array([0.0, 0.0, 0.0]),
    "vel": np.array([0.5, 0.0, 0.0])
}
info = {}

# Minimal config with gates (to generate waypoints)
env = SimpleNamespace(
    freq=50,
    track=SimpleNamespace(
        gates=[
            SimpleNamespace(pos=[1.0, 0.0, 0.0], rpy=[0, 0, 0]),
            SimpleNamespace(pos=[2.0, 1.0, 0.5], rpy=[0, 0, 0]),
            SimpleNamespace(pos=[3.0, 1.5, 1.0], rpy=[0, 0, 0]),
        ],
        obstacles=[]
    )
)
config = SimpleNamespace(env=env)

# Create controller instance
controller = MinSnapMPCController(obs, info, config)

# Get waypoints and build full waypoint array (including current position)
waypoints = controller._generate_waypoints()
all_wps = np.vstack([obs["pos"], waypoints])

# Test _calculate_trajectory_times and print results
times = controller._calculate_trajectory_times(all_wps)
print("\nCalculated trajectory times:")
print(times)

# Test _generate_reference_trajectory and print samples of outputs
controller._generate_reference_trajectory(obs)
traj = controller._reference_trajectory

print("\nReference trajectory time samples:")
print(traj['t'][:5])

print("\nReference trajectory position samples:")
print(traj['pos'][:5])

print("\nReference trajectory velocity samples:")
print(traj['vel'][:5])

print("\nReference trajectory acceleration samples:")
print(traj['acc'][:5])
