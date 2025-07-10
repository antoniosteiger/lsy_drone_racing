import numpy as np
import minsnap_trajectories as ms

freq = 50

def generate_trajectory(refs, duration):
    """
    Generates a trajectory using the minsnap_trajectories library.
    """
    print(f"DEBUG: Generating trajectory with {len(refs)} waypoints, duration={duration}s, freq={freq}Hz")

    polys = ms.generate_trajectory(
        refs,
        degree=8,  # Polynomial degree
        idx_minimized_orders=(3, 4),  
        num_continuous_orders=3,  
        algorithm="closed-form",  # Or "constrained"
    )
    

    t = np.linspace(0, duration, int(freq * duration))
    print(f"DEBUG: Time array created with {len(t)} points")

    states, inputs = ms.compute_quadrotor_trajectory(
        polys,
        t,
        vehicle_mass=0.028,  # Quadrotor weight
        drag_params=ms.RotorDragParameters(0.1, 0.2, 1.0),
    )
    print(f"DEBUG: States shape: {states.shape}")
    print(f"DEBUG: Inputs shape: {inputs.shape}")

    pva = ms.compute_trajectory_derivatives(polys, t, 3)
    
    pos = pva[0].T
    vel = pva[1].T
    acceleration = pva[2,...]
    
    print(f"DEBUG: Acceleration shape: {acceleration.shape}")

    # Assuming states columns: position (3), velocity (3), etc.
    pos = states[:, 0:3]
    vel = states[:, 3:6]
    print(f"DEBUG: Position shape: {pos.shape}")
    print(f"DEBUG: Velocity shape: {vel.shape}")

    ref_traj = np.hstack([pos, vel, acceleration])
    print(f"DEBUG: Final reference trajectory shape: {ref_traj.shape}")

    return ref_traj
