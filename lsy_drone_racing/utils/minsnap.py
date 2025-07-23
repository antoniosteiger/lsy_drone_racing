"""A simple wrapper for the minsnap-trajectories library.

Author:
    Antonio Steiger
"""

import minsnap_trajectories as ms
import numpy as np
from numpy.typing import NDArray

FREQ = 20  # Not really relevant here, trajectory is interpolated later.


def generate_trajectory(refs: NDArray, duration: float) -> NDArray:
    """Generates a trajectory using the minsnap_trajectories library."""
    polys = ms.generate_trajectory(
        refs,
        degree=8,  # Polynomial degree
        idx_minimized_orders=(3, 4),
        num_continuous_orders=3,
        algorithm="closed-form",  # Or "constrained"
    )

    t = np.linspace(0, duration, int(FREQ * duration))
    states, inputs = ms.compute_quadrotor_trajectory(
        polys,
        t,
        vehicle_mass=0.028,  # Quadrotor weight
        drag_params=ms.RotorDragParameters(0.1, 0.2, 1.0),
    )

    pva = ms.compute_trajectory_derivatives(polys, t, 3)
    acceleration = pva[2, ...]

    # print("States: ", states[0])
    # print("Inputs: ", inputs)

    return np.hstack([states, inputs, acceleration])
