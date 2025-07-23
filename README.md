# Submission to LSY Autonomous Drone Racing Course SS25

## Running the submission

Make sure all the usual requirements specified in the forked repo are met.
Activate the race environment: \
`mamba activate race`\

Run one of three controllers:\

Minsnap Tracker (No obstacle Avoidance):\
`python scripts/sim.py -g -t -config "level2.toml" -controller "minsnap_tracker.py"` \
Note that this controller will not be successful with any gate configuration. We suggest updating scripts/kaggle.py to run sim.py with arguments gui=True and trajectory_visualization=True and then running `python scripts/kaggle.py`to see ten consecutive runs.

Spline Tracker (obstacle avoidance)
`python scripts/sim.py -g -t -config "level1.toml" -controller "spline_tracker.py"` \
Note that this controller has issues with level2 at the moment due to the dynamic replacement of the trajectory.

MPC + Minsnap (obstacle avoidance)
`python scripts/sim.py -g -t -config "level2.toml" -controller "minsnap_mpc.py"`

The `-t` flag enables trajectory visualization, the `-g`flag enables the GUI.

## Submission Code Scope

Work submitted is the following:

- All python scripts within lsy_drone_racing/control/ except controller.py
- path_planner.py, trajectory.py and minsnap.py within lsy_drone_racing/utils/
- A slightly extended sim.py script that renders trajectories in a modular way (using trajectories.py)

## References

- Original submission repository: https://github.com/utiasDSL/lsy_drone_racing
- minsnap-trajectories library: https://pypi.org/project/minsnap-trajectories/
