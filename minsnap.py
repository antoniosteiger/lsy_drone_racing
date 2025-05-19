import numpy as np
import minsnap_trajectories as ms

num_waypoints = 5
freq = 20
duration = 16.0

refs = [
    # starting point
    ms.Waypoint(
        time= 0.0,
        position=np.array([1.0, 1.5, 0.07]),
        velocity=np.array([0.0, 0.0, 0.0]),
        acceleration=np.array([0.0, 0.0, 0.0]),
        jerk=np.array([0.0, 0.0, 0.0])
    ),
    # first gate
    ms.Waypoint(  # Any higher-order derivatives
        time= 4.0,
        position=np.array([0.4, -0.5, 0.56]),
    ),
    # intermediary
    # ms.Waypoint(  # Any higher-order derivatives
    #     time= 8.0,
    #     position=np.array([0.35, -1.7, 0.85]),
    # ),
    # second gate
    ms.Waypoint( 
        time= 7.8,
        position=np.array([1.0, -1.05, 1.11]),
        velocity=np.array([0.4, 0.4, 0.0])
    ),
    # third gate
    ms.Waypoint(
        time= 12.0,
        position=np.array([0.0, 1.2, 0.56]), # increased y to "touch gate"
        velocity=np.array([0.0, 0.0, 0.0]),
    ),
    # fourth gate
    ms.Waypoint(
        time= duration,
        position=np.array([-0.6, -0.2, 1.11]),
    )
]


if __name__ == "__main__":
    polys = ms.generate_trajectory(
        refs,
        degree=8,  # Polynomial degree
        idx_minimized_orders=(3, 4),  
        num_continuous_orders=3,  
        algorithm="closed-form",  # Or "constrained"
    )

    t = np.linspace(0, duration, int(freq * duration))
    states, inputs = ms.compute_quadrotor_trajectory(
        polys,
        t,
        vehicle_mass=0.028, # Quadrotor weight
        drag_params=ms.RotorDragParameters(0.1, 0.2, 1.0),
    )

    pva = ms.compute_trajectory_derivatives(polys, t, 3)
    acceleration = pva[2, ...]

    #print("States: ", states[0])
    #print("Inputs: ", inputs)

    t_col = t.reshape(-1, 1)
    np.savetxt('trajectory.csv', np.hstack([states, inputs, acceleration]), delimiter=',', fmt='%.6f')