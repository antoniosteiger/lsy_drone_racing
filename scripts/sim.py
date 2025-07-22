"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller, draw_line
import lsy_drone_racing.utils.trajectory as trajectory

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

import numpy as np # needed for np.array


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = None,
    trajectory_visualization: bool | None = None,  ### EXTENDED: visualize trajectory or not
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """

    # Settings
    trajectory_color = np.array([1.0, 1.0, 0, 1])

    if trajectory_visualization:
        print("SIM: Trajectory visualization enabled.")
    
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui

    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance

    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )

    env = JaxToNumpy(env)

    ep_times = []
    
    try:
        for _ in range(n_runs):  # Run n_runs episodes with the controller
            obs, info = env.reset()
            controller: Controller = controller_cls(obs, info, config)
            i = 0
            fps = 60

            while True:
                curr_time = i / config.env.freq

                action = controller.compute_control(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                # Update the controller internal state and models.
                controller_finished = controller.step_callback(
                    action, obs, reward, terminated, truncated, info
                )
                # Add up reward, collisions
                if terminated or truncated or controller_finished:
                    break
                # Synchronize the GUI.
                if config.sim.gui:
                    if ((i * fps) % config.env.freq) < fps:
                        env.render()
                        
                        # EXTENDED: Draw trajectory line
                        if trajectory_visualization:
                            # draw the trajectory line
                            draw_trajectory(env, trajectory.trajectory, trajectory_color)
                i += 1

            controller.episode_callback()  # Update the controller internal state and models.
            log_episode_stats(obs, info, config, curr_time)
            controller.episode_reset()
            ep_times.append(curr_time if obs["target_gate"] == -1 else None)
        
        # EXTENDED: Keep GUI open
        i = 0
        if config.sim.gui:
            while True:
                if ((i * fps) % config.env.freq) < fps:
                    env.render()
                    if trajectory_visualization:
                        # draw the trajectory line
                        draw_trajectory(env, trajectory.trajectory, trajectory_color)
                i += 1
    except KeyboardInterrupt:
        print("Closing visualization.")
    finally:
        # Close the environment
        env.close()
        return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )

def decimate(arr: np.ndarray, max_len: int) -> np.ndarray:
    length = len(arr)
    if length <= max_len:
        return arr
    step = int(np.ceil(length / max_len))
    return arr[::step]

def draw_trajectory(env, trajectory, color):
    # Draw the trajectory line
    trajectory = trajectory[:, 0:3]  # Only take the x, y, z coordinates
    trajectory = decimate(trajectory, 500)
    draw_line(env, trajectory, color, 10.0, 10.0)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
