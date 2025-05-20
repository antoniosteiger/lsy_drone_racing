# Time-Optimal Trajectory Generator
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
import gymnasium
from pathlib import Path
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy



def render_trajectory(
        trajectory = None,
        config: str = "level0.toml",
):
    config = load_config(Path(__file__).parents[1] / "config" / config)
    config.sim.gui = True
    
    # Initialize the racing environment
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

    env.reset()
    
    # Render the environment
    i = 0
    fps = 60
    try:
        while True:
            if ((i * fps) % config.env.freq) < fps:
                env.render()
            i += 1
    except KeyboardInterrupt:
        print("Closing visualization.")
    finally:
        env.close()

    return


if __name__ == "__main__":
    render_trajectory()