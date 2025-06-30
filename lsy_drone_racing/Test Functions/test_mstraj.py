import numpy as np
import minsnap_trajectories as ms

# Define simple straight-line waypoints
waypoints = [
    ms.Waypoint(position=np.array([0, 0, 0]), time=0.0, velocity=[0, 0, 0], acceleration=[0, 0, 0]),
    ms.Waypoint(position=np.array([1, 1, 1]), time=1.0),
    ms.Waypoint(position=np.array([2, 2, 2]), time=2.0, velocity=[0, 0, 0], acceleration=[0, 0, 0])
]

# Generate trajectory
polys = ms.generate_trajectory(waypoints, degree=7, idx_minimized_orders=(4,), num_continuous_orders=3, algorithm='closed-form')

# Evaluate derivatives
t = np.linspace(0, 2.0, 100)
pva = ms.compute_trajectory_derivatives(polys, t, 3)

print("Trajectory generated successfully.")
print("Positions shape:", pva[0].shape)  # should be (3, len(t))
